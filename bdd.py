import psycopg2
import psycopg2.extras
from config import *
from contextlib import contextmanager

@contextmanager
def get_conn():
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASSWORD
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_schema():
    """À appeler au démarrage de l'app (idempotent)."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id   SERIAL PRIMARY KEY,
                pseudo      TEXT UNIQUE NOT NULL,
                online      BOOLEAN DEFAULT TRUE,
                last_seen   TIMESTAMPTZ DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS games (
                game_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                player1     TEXT NOT NULL,
                player2     TEXT,                    -- NULL = IA
                status      TEXT DEFAULT 'waiting',  -- waiting|playing|finished
                winner      TEXT,
                created_at  TIMESTAMPTZ DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS boards (
                game_id     UUID REFERENCES games(game_id),
                player      TEXT NOT NULL,
                ships       JSONB NOT NULL,
                PRIMARY KEY (game_id, player)
            );
            CREATE TABLE IF NOT EXISTS moves (
                move_id     BIGSERIAL PRIMARY KEY,
                game_id     UUID,
                player      TEXT,
                turn        INT,
                x           INT, y INT,
                result      TEXT,                -- miss|hit|sunk
                ts          TIMESTAMPTZ DEFAULT now()
            );
            CREATE INDEX IF NOT EXISTS idx_moves_game ON moves(game_id);
        """)

def login(pseudo: str):
    """Login = upsert du pseudo. Renvoie l'id joueur."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""
            INSERT INTO players (pseudo, online, last_seen)
            VALUES (%s, TRUE, now())
            ON CONFLICT (pseudo)
            DO UPDATE SET online = TRUE, last_seen = now()
            RETURNING player_id;
        """, (pseudo,))
        return cur.fetchone()[0]

def set_online(pseudo: str, online: bool):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("UPDATE players SET online=%s WHERE pseudo=%s;", (online, pseudo))

def find_opponent(pseudo: str) -> str | None:
    """Cherche un joueur online (autre que moi) qui n'est pas en partie."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""
            SELECT p.pseudo FROM players p
            WHERE p.online AND p.pseudo != %s
              AND NOT EXISTS (
                  SELECT 1 FROM games g
                  WHERE g.status IN ('waiting','playing')
                    AND (g.player1 = p.pseudo OR g.player2 = p.pseudo)
              )
              AND p.last_seen > now() - interval '2 minutes'
            ORDER BY p.last_seen DESC LIMIT 1;
        """, (pseudo,))
        row = cur.fetchone()
        return row[0] if row else None

def create_game(p1: str, p2: str | None) -> str:
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""
            INSERT INTO games (player1, player2, status)
            VALUES (%s, %s, 'playing') RETURNING game_id;
        """, (p1, p2))
        return str(cur.fetchone()[0])

def save_board(game_id: str, player: str, ships: dict):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""
            INSERT INTO boards (game_id, player, ships) VALUES (%s, %s, %s)
            ON CONFLICT (game_id, player) DO UPDATE SET ships = EXCLUDED.ships;
        """, (game_id, player, psycopg2.extras.Json(ships)))

def save_move(game_id: str, player: str, turn: int, x: int, y: int, result: str):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO moves (game_id, player, turn, x, y, result) VALUES (%s,%s,%s,%s,%s,%s)",
            (game_id, player, turn, x, y, result))

def finish_game(game_id: str, winner: str):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("UPDATE games SET status='finished', winner=%s WHERE game_id=%s;",
                    (winner, game_id))

def get_moves(game_id: str) -> list[dict]:
    """Rejoue une partie (utile pour resynchroniser l'affichage)."""
    with get_conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT player, turn, x, y, result FROM moves WHERE game_id=%s ORDER BY move_id;", (game_id,))
        return cur.fetchall()