"""Logique de jeu de la bataille navale. Sans Streamlit pour la partie pure,
rendu Streamlit séparé. Hooks appelés par app.py pour persistance/Kafka."""
import random
import streamlit as st
import pandas as pd

GRID = 10
SHIPS = [("Porte-avions", 5), ("Croiseur", 4), ("Contre-torpilleur", 3),
         ("Sous-marin", 3), ("Torpilleur", 2)]

# ================= LOGIQUE PURE =================

def random_grid() -> dict:
    """Génère un placement aléatoire : {(x,y): nom_bateau}."""
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
    return all(
        all(shots.get(c) in ("hit", "sunk") for c, n in grid.items() if n == name)
        for name in names
    )

def grid_to_features(grid: dict, shots: dict) -> list[int]:
    """Encode l'état (pour l'IA / dataset) : 0=inconnu, 1=raté, 2=touché."""
    out = []
    for y in range(GRID):
        for x in range(GRID):
            r = shots.get((x, y))
            out.append({"miss": 1, "hit": 2, "sunk": 2}.get(r, 0))
    return out

# ================= ÉTAT SESSION =================

def init_game(vs_ai: bool = True):
    ss = st.session_state
    ss.phase = "placement"
    ss.vs_ai = vs_ai
    ss.placing = 0
    ss.orientation = "H"
    ss.my_grid = {}          # mes bateaux
    ss.enemy_grid = {}       # bateaux adverses (IA) ou inconnus (humain)
    ss.my_shots = {}         # mes tirs sur l'ennemi
    ss.enemy_shots = {}      # tirs de l'ennemi sur moi
    ss.turn = 1
    ss.my_turn = True
    ss.log = []
    ss.winner = None
    if vs_ai:
        ss.enemy_grid = random_grid()

def log(msg: str):
    st.session_state.log.append(f"[tour {st.session_state.turn}] {msg}")

# ================= RENDU =================

def _cell_label(shot: str | None, has_ship: bool, show_ship: bool) -> str:
    if shot == "miss":
        return "🌊"
    if shot in ("hit", "sunk"):
        return "💥" if shot == "hit" else "☠️"
    return "⚓" if (has_ship and show_ship) else "⬜"

def _render_grid(shots: dict, grid: dict, show_ships: bool,
                 key_prefix: str, clickable: bool = False, on_click=None):
    for y in range(GRID):
        cols = st.columns(GRID)
        for x, c in enumerate(cols):
            shot = shots.get((x, y))
            label = _cell_label(shot, (x, y) in grid, show_ships)
            if clickable and shot is None and on_click:
                if c.button("·" if not shot else label, key=f"{key_prefix}{x}-{y}",
                            use_container_width=True):
                    on_click(x, y)
            else:
                c.button(label, key=f"{key_prefix}{x}-{y}", disabled=True,
                         use_container_width=True)

def _summary_table(shots: dict, grid: dict) -> pd.DataFrame:
    data = [[_cell_label(shots.get((x, y)), (x, y) in grid, True)
             for x in range(GRID)] for y in range(GRID)]
    return pd.DataFrame(data, index=list("ABCDEFGHIJ"),
                        columns=[str(i) for i in range(GRID)])

# ================= IA (temporaire : aléatoire) =================

def ai_shot() -> tuple[int, int]:
    ss = st.session_state
    options = [(x, y) for x in range(GRID) for y in range(GRID)
               if (x, y) not in ss.enemy_shots]
    # Heuristique simple : finir un bateau touché si possible
    for (x, y), r in ss.enemy_shots.items():
        if r == "hit":
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                n = (x + dx, y + dy)
                if n in options:
                    return n
    return random.choice(options)

# ================= HOOKS (définis par app.py) =================
# on_shot(game_id, player, turn, x, y, result)
# on_ships_placed(game_id, player, grid)
# on_finish(game_id, winner)

# ================= PHASE PLACEMENT =================

def render_placement(on_ships_placed):
    ss = st.session_state
    name, size = SHIPS[ss.placing]
    st.subheader(f"Place ton **{name}** ({size} cases) — orientation {ss.orientation}")

    c1, c2, c3 = st.columns(3)
    if c1.button("🔄 Rotation"):
        ss.orientation = "V" if ss.orientation == "H" else "H"
        st.rerun()
    if c2.button("🎲 Placement auto"):
        ss.my_grid = random_grid()
        _finish_placement(on_ships_placed)
    if c3.button("⬅️ Annuler dernier"):
        if ss.my_grid:
            last_name = list(ss.my_grid.values())[-1]
            ss.my_grid = {c: n for c, n in ss.my_grid.items() if n != last_name}
            ss.placing = max(0, ss.placing - 1)
        st.rerun()

    def try_place(x, y):
        cells = can_place(ss.my_grid, x, y, size, ss.orientation == "H")
        if cells:
            for c in cells:
                ss.my_grid[c] = name
            ss.placing += 1
            if ss.placing >= len(SHIPS):
                _finish_placement(on_ships_placed)
            st.rerun()
        else:
            log("Placement invalide ⛔")

    _render_grid({}, ss.my_grid, True, "place_", clickable=True, on_click=try_place)

def _finish_placement(on_ships_placed):
    ss = st.session_state
    ss.phase = "battle"
    if ss.vs_ai and not ss.enemy_grid:
        ss.enemy_grid = random_grid()
    on_ships_placed(ss.my_grid)   # app.py → bdd.save_board + kafka

# ================= PHASE BATAILLE =================

def render_battle(on_shot, on_finish):
    ss = st.session_state

    def player_shot(x, y):
        result = shot_result(ss.enemy_grid, ss.my_shots, x, y)
        ss.my_shots[(x, y)] = result
        log(f"Tu tires en {(x, y)} : {{'miss': '🌊 raté', 'hit': '💥 touché',"
            f" 'sunk': '☠️ coulé !'}}[{result}]")
        on_shot(x, y, result)     # persist + kafka

        if all_sunk(ss.enemy_grid, ss.my_shots):
            ss.winner = "player"
            on_finish("player")
            return

        # Riposte
        if ss.vs_ai:
            ex, ey = ai_shot()
            r = shot_result(ss.my_grid, ss.enemy_shots, ex, ey)
            ss.enemy_shots[(ex, ey)] = r
            log(f"🤖 IA tire en {(ex, ey)} : "
                f"{{'miss': '🌊 raté', 'hit': '💥 touché', 'sunk': '☠️ coulé !'}}[{r}]")
            on_shot(ex, ey, r)
            if all_sunk(ss.my_grid, ss.enemy_shots):
                ss.winner = "IA"
                on_finish("IA")
        else:
            # humain : le tir adverse sera détecté par polling côté app.py
            pass
        ss.turn += 1
        st.rerun()

    colg, cols_ = st.columns([3, 2])

    with colg:
        st.subheader("🎯 Grille ennemie")
        _render_grid(ss.my_shots, {}, False, "shoot_",
                     clickable=ss.vs_ai, on_click=player_shot)

    with cols_:
        st.subheader("🛡️ Ta flotte")
        st.dataframe(_summary_table(ss.enemy_shots, ss.my_grid),
                     use_container_width=True)
        st.subheader("📜 Journal")
        for msg in reversed(ss.log[-10:]):
            st.caption(msg)

# ================= POINT D'ENTRÉE =================

def render(on_shot, on_finish, on_ships_placed):
    ss = st.session_state
    if ss.winner:
        st.balloons()
        won = ss.winner == "player"
        st.success(f"**{'Tu as gagné 🎉' if won else ss.winner + ' a gagné 🤖'}**")
    elif ss.phase == "placement":
        render_placement(on_ships_placed)
    else:
        render_battle(on_shot, on_finish)