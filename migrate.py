"""migrate.py — À lancer UNE FOIS sur Onyxia pour mettre à jour le schéma existant.

    python migrate.py

Ajoute les colonnes manquantes (started_at, finished_at) aux anciennes tables.
Idempotent : peut être relancé sans risque. Inutile après un reset complet.
"""
import logging

import bdd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def migrate() -> None:
    with bdd.get_conn() as conn, conn.cursor() as cur:
        for col in ("started_at", "finished_at"):
            cur.execute(
                "ALTER TABLE games ADD COLUMN IF NOT EXISTS %s TIMESTAMPTZ;" % col
            )
            print(f"✅ games.{col} OK")
    print("Migration terminée.")


if __name__ == "__main__":
    migrate()
