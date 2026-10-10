"""
bdd.py — Couche d'accès PostgreSQL pour la bataille navale.

Principes :
- Pool de connexions thread-safe (Streamlit = une session par navigateur)
- Une transaction courte par appel, commit/rollback automatiques
- Matchmaking atomique (FOR UPDATE SKIP LOCKED) : deux joueurs ne peuvent
  pas rejoindre la même file en même temps
- En PvP, la BDD fait foi : record_shot() calcule le résultat du tir
  côté serveur à partir du plateau adverse stocké en base
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg2
import psycopg2.extras
import psycopg2.pool

import config

logger = logging.getLogger("bataille.bdd")

# Un joueur sans heartbeat depuis 2 min est considéré hors-ligne
STALE_MINUTES = 2

_pool_lock = threading.Lock()
_pool: psycopg2.pool.ThreadedConnectionPool | None = None


# ==================== CONNEXIONS ====================

def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Pool singleton, créé à la première utilisation."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1, maxconn=10,
                    host=config.PG_HOST, port=config.PG_PORT,
                    dbname=config.PG_DB, user=config.PG_USER,
                    password=config.PG_PASSWORD,
                    connect_timeout=5,
                )
                logger.info("Pool PostgreSQL créé (%s:%s/%s)",
                            config.PG_HOST, config.PG_PORT, config.PG_DB)
    return _pool


@contextmanager
def get_conn() -> Iterator[Any]:
    """Connexion du pool avec commit/rollback automatiques."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ==================== SCHÉMA ====================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS players (
    player_id   SERIAL PRIMARY KEY,
    pseudo      TEXT UNIQUE NOT NULL,
    online      BOOLEAN NOT NULL DEFAULT TRUE,
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS games (
    game_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player1     TEXT NOT NULL REFERENCES players(pseudo),
    player2     TEXT REFERENCES players(pseudo),   -- NULL = file d'attente
    status      TEXT NOT NULL DEFAULT 'waiting'
                CHECK (status IN ('waiting', 'playing', 'finished')),
    winner      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_games_waiting
    ON games (created_at) WHERE status = 'waiting';

CREATE TABLE IF NOT EXISTS boards (
    game_id     UUID NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player      TEXT NOT NULL,
    ships       JSONB NOT NULL,  -- [{"x": 0, "y": 0, "ship": "Torpilleur"}, ...]
    PRIMARY KEY (game_id, player)
);

CREATE TABLE IF NOT EXISTS moves (
    move_id     BIGSERIAL PRIMARY KEY,
    game_id     UUID NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player      TEXT NOT NULL,
    turn        INT NOT NULL,
    x           INT NOT NULL CHECK (x BETWEEN 0 AND 9),
    y           INT NOT NULL CHECK (y BETWEEN 0 AND 9),
    result      TEXT NOT NULL CHECK (result IN ('miss', 'hit', 'sunk')),
    ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_moves_game ON moves (game_id, move_id);
"""


def init_schema() -> None:
    """Crée le schéma si absent + répare les tables existantes. Idempotent."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        # Colonnes ajoutées après le 1er déploiement : les anciennes tables
        # n'étaient pas modifiées par CREATE TABLE IF NOT EXISTS.
        for col in ("started_at", "finished_at"):
            cur.execute(
                "ALTER TABLE games ADD COLUMN IF NOT EXISTS %s TIMESTAMPTZ;" % col
            )


# ==================== JOUEURS ====================

def login(pseudo: str) -> int:
    """Upsert du pseudo et marque en ligne. Renvoie player_id."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            """
            INSERT INTO players (pseudo, online, last_seen)
            VALUES (%s, TRUE, now())
            ON CONFLICT (pseudo)
            DO UPDATE SET online = TRUE, last_seen = now()
            RETURNING player_id;
            """,
            (pseudo,),
        )
        return cur.fetchone()[0]


def touch(pseudo: str) -> None:
    """Heartbeat : à appeler à chaque rerun pour rester visible en ligne."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("UPDATE players SET last_seen = now() WHERE pseudo = %s;", (pseudo,))


def set_online(pseudo: str, online: bool) -> None:
    with get_conn() as c, c.cursor() as cur:
        cur.execute("UPDATE players SET online = %s WHERE pseudo = %s;", (online, pseudo))


def logout(pseudo: str) -> None:
    """Déconnexion propre : hors-ligne + sort de file + forfait des PvP en cours."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("UPDATE players SET online = FALSE WHERE pseudo = %s;", (pseudo,))
        cur.execute(
            "DELETE FROM games WHERE status = 'waiting' AND player1 = %s AND player2 IS NULL;",
            (pseudo,),
        )
        cur.execute(
            """
            UPDATE games
            SET status = 'finished', finished_at = now(),
                winner = CASE WHEN player1 = %s THEN player2 ELSE player1 END
            WHERE status = 'playing' AND player2 IS NOT NULL
              AND (player1 = %s OR player2 = %s);
            """,
            (pseudo, pseudo, pseudo),
        )


# ==================== MATCHMAKING ====================

def _match_payload(game: dict, pseudo: str) -> dict:
    role = "player1" if game["player1"] == pseudo else "player2"
    opponent = game["player2"] if role == "player1" else game["player1"]
    return {
        "game_id": str(game["game_id"]),
        "role": role,
        "opponent": opponent,
        "status": game["status"],
    }


def join_matchmaking(pseudo: str) -> dict:
    """
    Matchmaking atomique (une seule transaction). Idempotent : peut être
    rappelé à chaque polling — deux files parallèles finissent toujours
    par fusionner.

    1. déjà dans une partie 'playing' → la renvoie (reconnexion / match trouvé)
    2. nettoie les files fantômes (joueurs hors-ligne)
    3. file d'un autre joueur en attente → je le rejoins (partie 'playing')
       et je supprime mes éventuelles anciennes files
    4. sinon, je réutilise ma file existante ou j'en crée une nouvelle
    """
    with get_conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        # 1. Partie en cours ?
        cur.execute(
            """
            SELECT game_id, player1, player2, status FROM games
            WHERE status = 'playing' AND (player1 = %s OR player2 = %s)
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            (pseudo, pseudo),
        )
        current = cur.fetchone()
        if current:
            return _match_payload(current, pseudo)

        # 2. Files abandonnées (joueur hors-ligne)
        cur.execute(
            """
            DELETE FROM games
            WHERE status = 'waiting' AND player1 IN (
                SELECT pseudo FROM players
                WHERE NOT online OR last_seen < now() - make_interval(mins => %s)
            );
            """,
            (STALE_MINUTES,),
        )

        # 3. Rejoindre la file d'un autre joueur (FOR UPDATE : pas de doublon)
        cur.execute(
            """
            SELECT game_id, player1 FROM games
            WHERE status = 'waiting' AND player2 IS NULL AND player1 != %s
            ORDER BY created_at
            LIMIT 1
            FOR UPDATE SKIP LOCKED;
            """,
            (pseudo,),
        )
        row = cur.fetchone()
        if row:
            # Je sors de mes anciennes files avant de rejoindre celle-ci
            cur.execute(
                "DELETE FROM games WHERE status = 'waiting' AND player1 = %s AND player2 IS NULL;",
                (pseudo,),
            )
            cur.execute(
                """
                UPDATE games
                SET player2 = %s, status = 'playing', started_at = now()
                WHERE game_id = %s AND status = 'waiting' AND player2 IS NULL
                RETURNING game_id;
                """,
                (pseudo, row["game_id"]),
            )
            if cur.fetchone():
                logger.info("Match : %s rejoint %s", pseudo, row["player1"])
                return {"game_id": str(row["game_id"]), "role": "player2",
                        "opponent": row["player1"], "status": "playing"}

        # 4. Ma file existante, sinon nouvelle file
        cur.execute(
            """
            SELECT game_id FROM games
            WHERE status = 'waiting' AND player1 = %s AND player2 IS NULL
            ORDER BY created_at
            LIMIT 1;
            """,
            (pseudo,),
        )
        mine = cur.fetchone()
        if mine:
            return {"game_id": str(mine["game_id"]), "role": "player1",
                    "opponent": None, "status": "waiting"}

        cur.execute(
            "INSERT INTO games (player1, player2, status) VALUES (%s, NULL, 'waiting') RETURNING game_id;",
            (pseudo,),
        )
        game_id = str(cur.fetchone()["game_id"])
        logger.info("%s entre en file d'attente (%s)", pseudo, game_id)
        return {"game_id": game_id, "role": "player1", "opponent": None, "status": "waiting"}


def waiting_count() -> int:
    """Nombre de joueurs en file (affichage de la salle d'attente)."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM games WHERE status = 'waiting';")
        return cur.fetchone()[0]


def cancel_waiting_game(game_id: str, pseudo: str) -> None:
    """Sortie volontaire de la file d'attente."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "DELETE FROM games WHERE game_id = %s AND status = 'waiting' AND player1 = %s AND player2 IS NULL;",
            (game_id, pseudo),
        )


# ==================== PARTIES ====================

def create_ai_game(pseudo: str) -> str:
    """Partie contre l'IA : démarre immédiatement, pas de player2."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO games (player1, player2, status, started_at) VALUES (%s, NULL, 'playing', now()) RETURNING game_id;",
            (pseudo,),
        )
        return str(cur.fetchone()[0])


def get_game(game_id: str) -> dict | None:
    with get_conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM games WHERE game_id = %s;", (game_id,))
        row = cur.fetchone()
        if row:
            row["game_id"] = str(row["game_id"])
        return row


def finish_game(game_id: str, winner: str) -> None:
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "UPDATE games SET status = 'finished', winner = %s, finished_at = now() WHERE game_id = %s;",
            (winner, game_id),
        )


# ==================== PLATEAUX & TIRS ====================

def save_board(game_id: str, player: str, ships: list[dict]) -> None:
    """ships = [{"x": int, "y": int, "ship": str}, ...] (JSON-sérialisable)."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            """
            INSERT INTO boards (game_id, player, ships) VALUES (%s, %s, %s)
            ON CONFLICT (game_id, player) DO UPDATE SET ships = EXCLUDED.ships;
            """,
            (game_id, player, psycopg2.extras.Json(ships)),
        )


def get_board(game_id: str, player: str) -> list[dict] | None:
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT ships FROM boards WHERE game_id = %s AND player = %s;", (game_id, player))
        row = cur.fetchone()
        return row[0] if row else None


def save_move(game_id: str, player: str, turn: int, x: int, y: int, result: str) -> None:
    """Enregistrement brut (mode IA : le résultat est calculé côté client)."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO moves (game_id, player, turn, x, y, result) VALUES (%s, %s, %s, %s, %s, %s);",
            (game_id, player, turn, x, y, result),
        )


def get_moves(game_id: str) -> list[dict]:
    """Tous les tirs d'une partie, dans l'ordre (rejoue la partie)."""
    with get_conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT player, turn, x, y, result, ts FROM moves WHERE game_id = %s ORDER BY move_id;",
            (game_id,),
        )
        return cur.fetchall()


def record_shot(game_id: str, shooter: str, x: int, y: int) -> dict:
    """
    Tir PvP résolu côté serveur (transaction + verrou FOR UPDATE sur la partie).

    Lève ValueError si le tir est illégal (pas ton tour, case déjà visée...).
    Renvoie {"result": "miss"|"hit"|"sunk", "turn": int, "winner": str | None}.
    """
    with get_conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM games WHERE game_id = %s FOR UPDATE;", (game_id,))
        game = cur.fetchone()
        if game is None:
            raise ValueError("Partie introuvable.")
        if game["status"] != "playing":
            raise ValueError("La partie n'est pas en cours.")
        if shooter not in (game["player1"], game["player2"]):
            raise ValueError("Tu ne participes pas à cette partie.")
        opponent = game["player2"] if shooter == game["player1"] else game["player1"]

        cur.execute(
            "SELECT 1 FROM moves WHERE game_id = %s AND player = %s AND x = %s AND y = %s;",
            (game_id, shooter, x, y),
        )
        if cur.fetchone():
            raise ValueError("Case déjà visée.")

        cur.execute(
            "SELECT player, COUNT(*)::int AS n FROM moves WHERE game_id = %s GROUP BY player;",
            (game_id,),
        )
        counts = {r["player"]: r["n"] for r in cur.fetchall()}
        n_me, n_opp = counts.get(shooter, 0), counts.get(opponent, 0)
        my_turn = (shooter == game["player1"] and n_me == n_opp) or \
                  (shooter == game["player2"] and n_me < n_opp)
        if not my_turn:
            raise ValueError("Ce n'est pas ton tour.")

        cur.execute("SELECT ships FROM boards WHERE game_id = %s AND player = %s;", (game_id, opponent))
        row = cur.fetchone()
        ships = row[0] if row else []
        cells = {(s["x"], s["y"]): s["ship"] for s in ships}

        cur.execute("SELECT x, y, result FROM moves WHERE game_id = %s AND player = %s;", (game_id, shooter))
        past = {(r["x"], r["y"]): r["result"] for r in cur.fetchall()}
        past[(x, y)] = "hit"  # le tir courant participe au calcul de "coulé"

        ship = cells.get((x, y))
        if ship is None:
            result = "miss"
        else:
            ship_cells = [c for c, n in cells.items() if n == ship]
            result = "sunk" if all(past.get(c) in ("hit", "sunk") for c in ship_cells) else "hit"

        winner = None
        if cells and all(past.get(c) in ("hit", "sunk") for c in cells):
            winner = shooter

        turn = n_me + n_opp + 1
        cur.execute(
            "INSERT INTO moves (game_id, player, turn, x, y, result) VALUES (%s, %s, %s, %s, %s, %s);",
            (game_id, shooter, turn, x, y, result),
        )
        if winner:
            cur.execute(
                "UPDATE games SET status = 'finished', winner = %s, finished_at = now() WHERE game_id = %s;",
                (winner, game_id),
            )
        return {"result": result, "turn": turn, "winner": winner}


# ==================== VUE COMPLÈTE (polling Streamlit) ====================

def get_game_state(game_id: str, pseudo: str) -> dict:
    """
    Tout ce qu'il faut pour afficher une partie PvP :
    partie, tirs, à qui le tour, et si l'adversaire a placé sa flotte.
    """
    game = get_game(game_id)
    if game is None:
        raise ValueError("Partie introuvable.")
    moves = get_moves(game_id)
    opponent = game["player2"] if pseudo == game["player1"] else game["player1"]
    counts: dict[str, int] = {}
    for m in moves:
        counts[m["player"]] = counts.get(m["player"], 0) + 1
    n_me, n_opp = counts.get(pseudo, 0), counts.get(opponent, 0)
    if game["status"] == "playing" and opponent:
        my_turn = (pseudo == game["player1"] and n_me == n_opp) or \
                  (pseudo == game["player2"] and n_me < n_opp)
    else:
        my_turn = False
    return {
        "game": game,
        "moves": moves,
        "opponent": opponent,
        "my_turn": my_turn,
        "status": game["status"],
        "winner": game["winner"],
        "opponent_ready": bool(opponent and get_board(game_id, opponent)),
    }