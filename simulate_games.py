"""simulate_games.py — Génération massive de données d'entraînement.

Simule des parties IA vs IA en réutilisant la VRAIE logique du jeu
(game.random_grid, game.shot_result, game.ai_shot) et envoie tout dans
Kafka, exactement comme une vraie partie : game_start → boards → moves →
game_over. spark_silver.py / spark_gold.py les traitent ensuite normalement.

Les données simulées portent mode="simulation" : elles atterrissent dans
leurs propres partitions Spark, séparées des vraies parties (ai / pvp).

    python simulate_games.py --games 1000 --mirror      # random_v1 vs heuristic_v1
    python simulate_games.py --games 1000 --mix         # corpus mixte (recommandé)
    python simulate_games.py --policy-a heuristic_v1 --policy-b heuristic_v1
    python simulate_games.py --games 100 --sleep 0.2    # débit limité

Pourquoi un mix ? Le label `win` n'est informatif que si la variance des
résultats s'explique par la qualité des coups (r-vs-h) ; le full aléatoire
couvre uniformément les états (bon pour un modèle de probabilité de touché) ;
h-vs-h fournit des états de fin de partie « experts ». Aucun matchup seul
ne suffit — et en imitation, on apprend la policy qui a généré les données.

Volume : 1 partie ≈ 70-100 tirs + 2 flottes + 2 events ≈ ~100 messages.
1000 parties ≈ 100 000 messages — Kafka encaisse sans problème.
"""
import argparse
import logging
import random
import time
import uuid

import game
import game_logic
import kafka_utils

logger = logging.getLogger("bataille.simulation")

GRID = game.GRID
KNOWN_POLICIES = ("random_v1", "heuristic_v1", "tensor_v1")
# Corpus équilibré : 50% r-vs-h (signal qualité), 25% h-vs-h (états experts),
# 25% r-vs-r (exploration non biaisée)
MIX = [("random_v1", "heuristic_v1"),
       ("heuristic_v1", "random_v1"),
       ("heuristic_v1", "heuristic_v1"),
       ("random_v1", "random_v1")]


def choose_shot(policy: str, my_shots: dict) -> tuple[int, int]:
    """Prochain tir selon la policy. Toute nouvelle policy s'ajoute ici."""
    if policy == "heuristic_v1":
        return game_logic.heuristic_shot(my_shots)          # la vraie heuristique du jeu
    if policy == "tensor_v1":
        from use_IA import tensor_shot         # lazy : TF requis seulement si utilisé
        return tensor_shot(my_shots)
    # random_v1 : tir purement aléatoire
    return game_logic.random_shot(my_shots)


def simulate_game(policy_a: str, policy_b: str, first: int = 0) -> str:
    """Une partie complète sim_A vs sim_B. Renvoie la policy du gagnant.

    `first` = index du joueur qui commence (0 ou 1) — l'avantage du premier
    coup est réel à la bataille navale, --mirror permet d'équilibrer.
    """
    policies = (policy_a, policy_b)
    game_id = str(uuid.uuid4())
    players = ["sim_A", "sim_B"]
    grids = {p: game.random_grid() for p in players}          # placement aléatoire
    shots = {p: {} for p in players}

    kafka_utils.send_event("game_start", game_id=game_id,
                           players=players, mode="simulation")
    for p in players:
        kafka_utils.send_board(game_id, p, game.grid_to_ships(grids[p]),
                               policy="random_v1", mode="simulation")

    winner, winner_policy, turn = None, None, 1
    order = [first, 1 - first]
    while winner is None and turn <= GRID * GRID:   # borne de sécurité
        for i in order:
            p, opp = players[i], players[1 - i]
            x, y = choose_shot(policies[i], shots[p])
            result = game.shot_result(grids[opp], shots[p], x, y)
            shots[p][(x, y)] = result
            kafka_utils.send_move(game_id, p, turn, x, y, result,
                                  policy=policies[i],
                                  extra={"mode": "simulation"})
            if game.all_sunk(grids[opp], shots[p]):
                winner, winner_policy = p, policies[i]
                break
        turn += 1

    kafka_utils.send_event("game_over", game_id=game_id, winner=winner)
    return winner_policy


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulateur de parties IA vs IA")
    ap.add_argument("--games", type=int, default=100, help="nombre de parties")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="pause en secondes entre les parties")
    ap.add_argument("--mirror", action="store_true",
                    help="alterne le joueur qui commence (équilibre l'avantage du 1er coup)")
    ap.add_argument("--policy-a", default="random_v1", choices=KNOWN_POLICIES)
    ap.add_argument("--policy-b", default="heuristic_v1", choices=KNOWN_POLICIES)
    ap.add_argument("--mix", action="store_true",
                    help="corpus équilibré : 50%% r-vs-h, 25%% h-vs-h, 25%% r-vs-r")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    kafka_utils.ensure_topics()

    wins: dict[str, int] = {}
    matchups: dict[tuple, int] = {}
    start = time.time()
    for i in range(args.games):
        matchup = MIX[i % len(MIX)] if args.mix else (args.policy_a, args.policy_b)
        matchups[matchup] = matchups.get(matchup, 0) + 1
        winner = simulate_game(*matchup, first=(i % 2) if args.mirror else 0)
        wins[winner] = wins.get(winner, 0) + 1
        if (i + 1) % 100 == 0:
            kafka_utils.flush()      # vide le buffer du producer périodiquement
            dt = time.time() - start
            print(f"{i + 1}/{args.games} parties ({(i + 1) / dt:.1f}/s) — "
                  f"victoires : {wins}")
        elif (i + 1) % 10 == 0:
            print(f"{i + 1}/{args.games}…")
        if args.sleep:
            time.sleep(args.sleep)

    kafka_utils.flush()              # rien ne doit rester dans le buffer
    dt = time.time() - start
    print(f"\n✅ {args.games} parties en {dt:.0f}s "
          f"({args.games / dt:.1f} parties/s)")
    print(f"Matchups : {matchups}")
    print(f"Victoires par policy : {wins}")
    print("Lance ensuite : python spark_silver.py && python spark_gold.py")


if __name__ == "__main__":
    main()
