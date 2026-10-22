"""game_logic.py — Logique pure de la bataille navale, SANS Streamlit.

Module partagé entre :
- game.py            (interface Streamlit)
- simulate_games.py  (génération de données)
- training_IA.py     (environnement RL / DQN)
- use_IA.py          (inférence)

Aucune dépendance externe : importable partout (service TF, Spark, tests).
"""
import random

GRID = 10
SHIPS = [("Porte-avions", 5), ("Croiseur", 4), ("Contre-torpilleur", 3),
         ("Sous-marin", 3), ("Torpilleur", 2)]
SHIP_CELLS = sum(size for _, size in SHIPS)   # 17

RESULT_LABEL = {"miss": "🌊 raté", "hit": "💥 touché", "sunk": "☠️ coulé !"}

# Reward shaping du DQN (modifiable)
R_MISS = -0.1
R_HIT = 1.0
R_SUNK = 3.0
R_WIN = 10.0


# ==================== PLACEMENT ====================

def random_grid() -> dict:
    """Génère un placement aléatoire : {(x, y): nom_bateau}."""
    grid, occupied = {}, set()
    for name, size in SHIPS:
        while True:
            horiz = random.choice([True, False])
            x = random.randint(0, GRID - (size if horiz else 1))
            y = random.randint(0, GRID - (size if not horiz else 1))
            cells = ([(x + i, y) for i in range(size)] if horiz
                     else [(x, y + i) for i in range(size)])
            if not any(c in occupied for c in cells):
                occupied.update(cells)
                for c in cells:
                    grid[c] = name
                break
    return grid


def can_place(grid: dict, x: int, y: int, size: int, horiz: bool) -> list | None:
    """Cases occupées par le bateau si le placement est valide, sinon None."""
    cells = ([(x + i, y) for i in range(size)] if horiz
             else [(x, y + i) for i in range(size)])
    if all(0 <= cx < GRID and 0 <= cy < GRID and (cx, cy) not in grid
           for cx, cy in cells):
        return cells
    return None


def grid_to_ships(grid: dict) -> list[dict]:
    """{(x, y): nom} → [{"x", "y", "ship"}] : sérialisable en JSON/JSONB."""
    return [{"x": x, "y": y, "ship": name} for (x, y), name in sorted(grid.items())]


def ships_to_grid(ships: list[dict]) -> dict:
    """Inverse de grid_to_ships (reconnexion : recharge le plateau depuis la BDD)."""
    return {(s["x"], s["y"]): s["ship"] for s in ships}


# ==================== TIRS ====================

def shot_result(grid: dict, shots: dict, x: int, y: int) -> str:
    """Résultat d'un tir sur `grid` (bateaux) sachant `shots` (tirs passés)."""
    if (x, y) not in grid:
        return "miss"
    name = grid[(x, y)]
    cells = [c for c, n in grid.items() if n == name]
    sunk = all(shots.get(c) in ("hit", "sunk") or c == (x, y) for c in cells)
    return "sunk" if sunk else "hit"


def all_sunk(grid: dict, shots: dict) -> bool:
    names = set(grid.values())
    return bool(names) and all(
        all(shots.get(c) in ("hit", "sunk") for c, n in grid.items() if n == name)
        for name in names
    )


def grid_to_features(grid: dict, shots: dict) -> list[int]:
    """Encode l'état (pour l'IA / dataset) : 0=inconnu, 1=raté, 2=touché 3=coulé.

    NB : `grid` n'est pas utilisé (l'agent ne connaît pas la flotte adverse) —
    le paramètre est conservé pour compatibilité, passez {}.
    """
    return [
        {"miss": 1, "hit": 2, "sunk": 3}.get(shots.get((x, y)), 0)
        for y in range(GRID) for x in range(GRID)
    ]


def heuristic_shot(shots: dict) -> tuple[int, int]:
    """Policy 'heuristic_v1' : finir un bateau touché si possible, sinon aléatoire."""
    options = [(x, y) for x in range(GRID) for y in range(GRID)
               if (x, y) not in shots]
    for (x, y), r in shots.items():
        if r == "hit":
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                n = (x + dx, y + dy)
                if n in options:
                    return n
    return random.choice(options)


def random_shot(shots: dict) -> tuple[int, int]:
    """Full aléatoire"""
    options = [(x, y) for x in range(GRID) for y in range(GRID)
               if (x, y) not in shots]
    return random.choice(options)


def dqn1_shot(shots: dict) -> tuple[int, int]:
    """Policy 'tensor_v1' : à implementer lorsque l'IA aura fini de s'entrainer"""
    options = [(x, y) for x in range(GRID) for y in range(GRID)
               if (x, y) not in shots]
    for (x, y), r in shots.items():
        if r == "hit":
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                n = (x + dx, y + dy)
                if n in options:
                    return n
    return random.choice(options)


def cell_to_xy(cell: int) -> tuple[int, int]:
    """Index 0-99 → (x, y). Cohérent avec action_cell = y * GRID + x du gold."""
    return cell % GRID, cell // GRID


def xy_to_cell(x: int, y: int) -> int:
    return y * GRID + x


# ==================== ENVIRONNEMENT RL ====================

class BattleshipEnv:
    """Environnement solo pour le DQN : l'agent tire sur une flotte cachée.

    État   : vecteur 100 (0=inconnu, 1=raté, 2=touché 3=coulé)
    Action : index de case 0-99
    Reward : R_HIT / R_SUNK / R_MISS, + R_WIN à la victoire
    """

    def __init__(self):
        self.reset()

    def reset(self) -> list[int]:
        self.grid = random_grid()
        self.shots: dict = {}
        self.turns = 0
        return self.state()

    def state(self) -> list[int]:
        return grid_to_features({}, self.shots)

    def valid_actions(self) -> list[int]:
        return [c for c in range(GRID * GRID)
                if cell_to_xy(c) not in self.shots]

    def step(self, cell: int) -> tuple[list[int], float, bool, dict]:
        x, y = cell_to_xy(cell)
        if (x, y) in self.shots:
            raise ValueError(f"Case {x, y} déjà jouée (action invalide)")
        result = shot_result(self.grid, self.shots, x, y)
        self.shots[(x, y)] = result
        self.turns += 1
        done = all_sunk(self.grid, self.shots)
        reward = {"miss": R_MISS, "hit": R_HIT, "sunk": R_SUNK}[result]
        if done:
            reward += R_WIN
        return self.state(), reward, done, {"result": result}
