"""pipeline_batch.py — Enchaîne silver → gold, une fois ou en boucle.

Usage :
    python pipeline_batch.py                  # une seule passe (avant un entraînement)
    python pipeline_batch.py --interval 60    # toutes les 60 min, en continu

Rappel : inutile de les faire tourner en permanence — bronze (S3) est durable,
silver/gold sont des vues recalculables. La boucle n'a de sens que si tu veux
une gold toujours fraîche sans intervention (ex : entraînements réguliers).
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime


def run(script: str) -> bool:
    print(f"\n── {datetime.now():%H:%M:%S} — {script} ──", flush=True)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"❌ {script} a échoué (code {result.returncode})")
        return False
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Orchestrateur batch silver → gold")
    ap.add_argument("--interval", type=int, default=0,
                    help="minutes entre chaque passe (0 = une seule fois)")
    args = ap.parse_args()

    while True:
        ok = run("spark_silver.py") and run("spark_gold.py")
        if args.interval <= 0:
            break
        if not ok:
            print(f"⚠️  Passe échouée, nouvelle tentative dans {args.interval} min")
        print(f"💤 Prochaine passe dans {args.interval} min "
              f"(Ctrl+C pour arrêter)", flush=True)
        try:
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            break
    print("── Terminé ──")
