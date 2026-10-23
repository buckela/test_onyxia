"""use_IA.py — Inférence des modèles entraînés (DQN tirs + régression placement).

CLI :
    python use_IA.py --eval 100        # winrate + tirs moyens vs heuristic_v1
    python use_IA.py --demo            # une partie solo commentée
    python use_IA.py --placement       # suggère un placement optimal (régression)
    python use_IA.py --download        # récupère les modèles depuis S3 d'abord

Fonctions importables ailleurs :
    tensor_shot(shots)  -> (x, y)    # policy 'tensor_v1' (utilisée par simulate_games)
    best_placement()    -> list[dict]

Deps : tensorflow, numpy (+ boto3 uniquement pour --download).
"""
import argparse
import os
import random

import numpy as np
from tensorflow import keras

from game_logic import (GRID, SHIP_CELLS, STATE_SCALE, BattleshipEnv, all_sunk,
                        cell_to_xy, grid_to_features, grid_to_ships,
                        heuristic_shot, random_grid, shot_result)

MODEL_DIR = "models"
SHOOTER_NAME_DQN = "dn_shooter.keras"
PLACEMENT_NAME = "placement_regressor.keras"
SHOOTER_PATH = os.path.join(MODEL_DIR, SHOOTER_NAME_DQN)
PLACEMENT_PATH = os.path.join(MODEL_DIR, PLACEMENT_NAME)

_shooter = None
_regressor = None


# ==================== HELPERS ====================

def greedy_cell(model: keras.Model, state: np.ndarray) -> int:
    """Argmax Q sur les cases encore inconnues (masking des actions invalides)."""
    q = model.predict(state.reshape(1, GRID, GRID, 1), verbose=0)[0]
    q = np.where(state == 0, q, -1e9)   # cases connues interdites
    return int(np.argmax(q))


def _load(path: str, name: str) -> keras.Model:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} introuvable — entraîne d'abord (training_IA.py) "
            f"ou récupère le modèle : python use_IA.py --download")
    return keras.models.load_model(path)


# ==================== TIRS (DQN) ====================

def tensor_shot(shots: dict) -> tuple[int, int]:
    """Policy 'tensor_v1' : état → DQN → meilleure case inconnue."""
    global _shooter
    if _shooter is None:
        _shooter = _load(SHOOTER_PATH, "shooter")
    state = np.array(grid_to_features({}, shots), dtype=np.float32) / STATE_SCALE
    return cell_to_xy(greedy_cell(_shooter, state))


def evaluate(n_games: int = 100) -> None:
    """Duels tensor_v1 vs heuristic_v1 (départ alterné) + efficacité solo."""
    model = _load(SHOOTER_PATH, "shooter")

    # --- Efficacité solo : tirs moyens pour couler la flotte ---
    env = BattleshipEnv()
    solo_turns = []
    for _ in range(n_games):
        state = np.array(env.reset(), dtype=np.float32) / STATE_SCALE
        done = False
        while not done:
            _, _, done, _ = env.step(greedy_cell(model, state))
            state = np.array(env.state(), dtype=np.float32) / STATE_SCALE
        solo_turns.append(env.turns)

    # --- Duels vs heuristique ---
    wins = 0
    for i in range(n_games):
        grids = {"A": random_grid(), "B": random_grid()}
        shots = {"A": {}, "B": {}}
        order = ["A", "B"] if i % 2 == 0 else ["B", "A"]   # départ alterné
        while True:
            for side in order:
                opp = "B" if side == "A" else "A"
                if side == "A":
                    x, y = tensor_shot(shots["A"])
                else:
                    x, y = heuristic_shot(shots["B"])
                shots[side][(x, y)] = shot_result(grids[opp], shots[side], x, y)
                if all_sunk(grids[opp], shots[side]):
                    wins += side == "A"
                    break
            else:
                continue
            break

    print(f"── Évaluation sur {n_games} parties ──")
    print(f"Winrate vs heuristic_v1 : {wins / n_games:.1%}")
    print(f"Tirs moyens pour gagner (solo) : {np.mean(solo_turns):.1f} "
          f"(référence aléatoire ≈ 95, bonne heuristique ≈ 55-60)")


def demo() -> None:
    """Une partie solo commentée coup par coup."""
    env = BattleshipEnv()
    state = np.array(env.reset(), dtype=np.float32) / STATE_SCALE
    done = False
    while not done:
        cell = greedy_cell(_load(SHOOTER_PATH, "shooter"), state)
        x, y = cell_to_xy(cell)
        _, _, done, info = env.step(cell)
        if (info['result'] == "miss"):
            info_p = "m" 
        else:
            info_p = info['result']
        print(f"tour {env.turns:>2} → ({x},{y}) : {info_p}")
        state = np.array(env.state(), dtype=np.float32) / STATE_SCALE
    print(f"✅ Flotte coulée en {env.turns} tirs")


# ==================== PLACEMENT (régression) ====================

def featurize_placement(ships: list[dict]) -> np.ndarray:
    """Même features que le gold : placement[100] + adjacent_pairs + edge_cells
    (normalisés). Doit rester cohérent avec spark_gold.placement_features."""
    cells = {(s["x"], s["y"]) for s in ships}
    vec = [0.0] * (GRID * GRID)
    for (x, y) in cells:
        vec[y * GRID + x] = 1.0
    adj = sum(1 for (x, y) in cells for dx, dy in ((1, 0), (0, 1))
              if (x + dx, y + dy) in cells) / SHIP_CELLS
    edge = sum(1 for (x, y) in cells if x in (0, 9) or y in (0, 9)) / SHIP_CELLS
    return np.array([vec + [adj, edge]], dtype=np.float32)


def best_placement(n_candidates: int = 500) -> list[dict]:
    """Tire n placements aléatoires, garde celui qui minimise les dégâts prédits."""
    global _regressor
    if _regressor is None:
        _regressor = _load(PLACEMENT_PATH, "placement")
    best_ships, best_score = None, np.inf
    for _ in range(n_candidates):
        ships = grid_to_ships(random_grid())
        score = _regressor.predict(featurize_placement(ships), verbose=0)[0, 0]
        if score < best_score:
            best_ships, best_score = ships, score
    return best_ships


def show_placement() -> None:
    ships = best_placement()
    cells = {(s["x"], s["y"]): s["ship"] for s in ships}
    print("Placement suggéré (dégâts prédits minimaux) :\n")
    print("    " + " ".join(str(x) for x in range(GRID)))
    for y in range(GRID):
        row = " ".join("🚢" if (x, y) in cells else "🟦" for x in range(GRID))
        print(f"{y:>3} {row}")
    print(f"\nLégende : {[s['ship'] for s in sorted(ships, key=lambda s: s['ship'])][:5]}")


# ==================== S3 ====================

def download_models() -> None:
    """Récupère les .keras depuis s3://<bucket>/bataille-navale/models/."""
    import boto3
    import config
    s3 = boto3.client("s3", endpoint_url=config.S3_ENDPOINT)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name in (SHOOTER_NAME_DQN, PLACEMENT_NAME):
        key = f"bataille-navale/models/{name}"
        s3.download_file(config.S3_BUCKET, key, os.path.join(MODEL_DIR, name))
        print(f"✅ {name} téléchargé")


# ==================== MAIN ====================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inférence des modèles bataille navale")
    ap.add_argument("--eval", type=int, metavar="N", help="évaluation sur N parties")
    ap.add_argument("--demo", action="store_true", help="partie solo commentée")
    ap.add_argument("--placement", action="store_true", help="suggère un placement")
    ap.add_argument("--download", action="store_true", help="modèles depuis S3")
    args = ap.parse_args()

    if args.download:
        download_models()
    if args.eval:
        evaluate(args.eval)
    elif args.demo:
        demo()
    elif args.placement:
        show_placement()
    else:
        ap.print_help()
