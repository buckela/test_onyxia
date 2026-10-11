"""manuel_bdd.py — Accès manuel à la base PostgreSQL.

Commandes rapides :
    python manuel_bdd.py stats                  # compteurs globaux
    python manuel_bdd.py players                # joueurs (online, last_seen)
    python manuel_bdd.py games [--status playing] [--limit 20]
    python manuel_bdd.py moves <game_id>        # tirs d'une partie
    python manuel_bdd.py board <game_id>        # plateaux d'une partie
    python manuel_bdd.py sql "SELECT ..."       # requête libre (lecture)
    python manuel_bdd.py sql "DELETE ..." --write   # écriture (avec confirm)

    python manuel_bdd.py sql "SELECT * FROM games WHERE status='finished' ORDER BY finished_at DESC LIMIT 5"
    python manuel_bdd.py sql "DELETE FROM games WHERE status='waiting'" --write

Sans argument : REPL interactif (SQL libre, 'quit' pour sortir).
"""
import argparse
import sys

import psycopg2.extras

import bdd

try:
    import pandas as pd
    def _print_rows(rows: list[dict]) -> None:
        if not rows:
            print("(aucune ligne)")
        else:
            print(pd.DataFrame(rows).to_string(index=False))
        print(f"\n{len(rows)} ligne(s)")
except ImportError:  # pandas absent : affichage minimal
    def _print_rows(rows: list[dict]) -> None:
        for r in rows:
            print(r)
        print(f"\n{len(rows)} ligne(s)")


def query(sql: str, params: tuple = ()) -> list[dict]:
    with bdd.get_conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


# ==================== COMMANDES MÉTIER ====================

def cmd_stats() -> None:
    rows = query("""
        SELECT
            (SELECT COUNT(*) FROM players)                        AS players,
            (SELECT COUNT(*) FROM players WHERE online)           AS online,
            (SELECT COUNT(*) FROM games WHERE status='waiting')   AS waiting,
            (SELECT COUNT(*) FROM games WHERE status='playing')   AS playing,
            (SELECT COUNT(*) FROM games WHERE status='finished')  AS finished,
            (SELECT COUNT(*) FROM moves)                          AS moves;
    """)
    _print_rows(rows)


def cmd_players() -> None:
    _print_rows(query("""
        SELECT pseudo, online,
               last_seen AT TIME ZONE 'Europe/Paris' AS last_seen,
               EXTRACT(EPOCH FROM (now() - last_seen))::int AS idle_sec
        FROM players ORDER BY last_seen DESC;
    """))


def cmd_games(status: str | None, limit: int) -> None:
    sql = """
        SELECT game_id, player1, player2, status, winner,
               created_at AT TIME ZONE 'Europe/Paris' AS created_at,
               (SELECT COUNT(*) FROM moves m WHERE m.game_id = g.game_id) AS nb_moves
        FROM games g
    """
    params: tuple = ()
    if status:
        sql += " WHERE status = %s"
        params = (status,)
    sql += " ORDER BY created_at DESC LIMIT %s;"
    params += (limit,)
    _print_rows(query(sql, params))


def cmd_moves(game_id: str) -> None:
    _print_rows(query("""
        SELECT move_id, player, turn, x, y, result,
               ts AT TIME ZONE 'Europe/Paris' AS ts
        FROM moves WHERE game_id = %s ORDER BY move_id;
    """, (game_id,)))


def cmd_board(game_id: str) -> None:
    rows = query("SELECT player, ships FROM boards WHERE game_id = %s;", (game_id,))
    if not rows:
        print("(aucun plateau pour cette partie)")
        return
    for r in rows:
        cells = {(s["x"], s["y"]): s["ship"] for s in r["ships"]}
        print(f"\n=== Plateau de {r['player']} ({len(cells)} cases) ===")
        header = "    " + " ".join(f"{x:>3}" for x in range(10))
        print(header)
        for y in range(10):
            line = f"{y:>3} " + " ".join(
                f"{cells[(x, y)][:2] if (x, y) in cells else '·':>3}"
                for x in range(10))
            print(line)


# ==================== SQL LIBRE ====================

def cmd_sql(sql: str, write: bool) -> None:
    is_read = sql.lstrip().lower().startswith(("select", "with", "explain"))
    if not is_read and not write:
        print("⚠️  Écriture détectée : ajoute --write pour confirmer.")
        sys.exit(1)
    if is_read:
        _print_rows(query(sql))
    else:
        with bdd.get_conn() as c, c.cursor() as cur:
            cur.execute(sql)
            print(f"✅ {cur.rowcount} ligne(s) affectée(s)")


def repl() -> None:
    print("REPL SQL — tape ta requête, 'quit' pour sortir.")
    print("Raccourcis : .stats .players .games\n")
    shortcuts = {".stats": cmd_stats, ".players": cmd_players,
                 ".games": lambda: cmd_games(None, 20)}
    while True:
        try:
            line = input("sql> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line in ("quit", "exit", "\\q"):
            break
        try:
            if line in shortcuts:
                shortcuts[line]()
            elif line.lstrip().lower().startswith(("select", "with", "explain")):
                _print_rows(query(line))
            else:
                with bdd.get_conn() as c, c.cursor() as cur:
                    cur.execute(line)
                    print(f"✅ {cur.rowcount} ligne(s) affectée(s)")
        except Exception as e:   # on reste dans le REPL
            print(f"❌ {e}")
    print("Bye.")


# ==================== MAIN ====================

def main() -> None:
    p = argparse.ArgumentParser(description="Accès manuel à la BDD bataille navale")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("stats")
    sub.add_parser("players")

    g = sub.add_parser("games")
    g.add_argument("--status", choices=["waiting", "playing", "finished"])
    g.add_argument("--limit", type=int, default=20)

    m = sub.add_parser("moves")
    m.add_argument("game_id")

    b = sub.add_parser("board")
    b.add_argument("game_id")

    s = sub.add_parser("sql")
    s.add_argument("sql")
    s.add_argument("--write", action="store_true")

    args = p.parse_args()

    if args.cmd is None:
        repl()
    elif args.cmd == "stats":
        cmd_stats()
    elif args.cmd == "players":
        cmd_players()
    elif args.cmd == "games":
        cmd_games(args.status, args.limit)
    elif args.cmd == "moves":
        cmd_moves(args.game_id)
    elif args.cmd == "board":
        cmd_board(args.game_id)
    elif args.cmd == "sql":
        cmd_sql(args.sql, args.write)


if __name__ == "__main__":
    main()
