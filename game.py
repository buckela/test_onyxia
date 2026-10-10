"""Logique de la bataille navale : logique pure + rendu Streamlit.

- vs_ai : tout est calculé côté client (riposte immédiate de l'IA)
- PvP   : la BDD fait foi — app.py fournit fetch_state / resolve_shot,
          les tirs adverses sont récupérés par polling (auto-refresh)
"""
import random
import time

import pandas as pd
import streamlit as st

import bdd

GRID = 10
SHIPS = [("Porte-avions", 5), ("Croiseur", 4), ("Contre-torpilleur", 3),
         ("Sous-marin", 3), ("Torpilleur", 2)]

RESULT_LABEL = {"miss": "🌊 raté", "hit": "💥 touché", "sunk": "☠️ coulé !"}

# Clés posées par init_game() — utilisées par app.py pour le reset
STATE_KEYS = ["phase", "vs_ai", "placing", "orientation", "my_grid", "enemy_grid",
              "my_shots", "enemy_shots", "turn", "log", "winner"]

# ================= LOGIQUE PURE =================

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
    """Encode l'état (pour l'IA / dataset) : 0=inconnu, 1=raté, 2=touché."""
    return [
        {"miss": 1, "hit": 2, "sunk": 2}.get(shots.get((x, y)), 0)
        for y in range(GRID) for x in range(GRID)
    ]

def grid_to_ships(grid: dict) -> list[dict]:
    """{(x, y): nom} → [{"x", "y", "ship"}] : sérialisable en JSON/JSONB."""
    return [{"x": x, "y": y, "ship": name} for (x, y), name in sorted(grid.items())]

def ships_to_grid(ships: list[dict]) -> dict:
    """Inverse de grid_to_ships (reconnexion : recharge le plateau depuis la BDD)."""
    return {(s["x"], s["y"]): s["ship"] for s in ships}

# ================= ÉTAT SESSION =================

def init_game(vs_ai: bool = True):
    ss = st.session_state
    ss.phase = "placement"
    ss.vs_ai = vs_ai
    ss.placing = 0
    ss.orientation = "H"
    ss.my_grid = {}          # mes bateaux
    ss.enemy_grid = {}       # bateaux adverses (IA uniquement ; inconnu en PvP)
    ss.my_shots = {}         # mes tirs sur l'ennemi
    ss.enemy_shots = {}      # tirs de l'ennemi sur moi
    ss.turn = 1
    ss.log = []
    ss.winner = None
    if vs_ai:
        ss.enemy_grid = random_grid()

def enter_pvp_game(game_id: str, opponent: str):
    """Prépare une partie PvP ; recharge le plateau si on se reconnecte."""
    init_game(vs_ai=False)
    ships = bdd.get_board(game_id, st.session_state.pseudo)
    if ships:
        st.session_state.my_grid = ships_to_grid(ships)
        st.session_state.placing = len(SHIPS)
        st.session_state.phase = "battle"

def log(msg: str):
    st.session_state.log.append(f"[tour {st.session_state.turn}] {msg}")

def _poll(seconds: float = 2.0):
    """Rafraîchissement par polling serveur : fonctionne aussi dans un onglet
    en arrière-plan (les timers JS des composants y sont bridés par le navigateur)."""
    time.sleep(seconds)
    st.rerun()

# ================= RENDU =================

def _cell_label(shot: str | None, has_ship: bool, show_ship: bool) -> str:
    if shot == "miss":
        return "🌊"
    if shot == "hit":
        return "💥"
    if shot == "sunk":
        return "☠️"
    return "⚓" if (has_ship and show_ship) else "⬜"

def _render_grid(shots: dict, grid: dict, show_ships: bool,
                 key_prefix: str, clickable: bool = False, on_click=None):
    for y in range(GRID):
        cols = st.columns(GRID)
        for x, c in enumerate(cols):
            shot = shots.get((x, y))
            label = _cell_label(shot, (x, y) in grid, show_ships)
            if clickable and shot is None and (x, y) not in grid and on_click:
                if c.button("·", key=f"{key_prefix}{x}-{y}", use_container_width=True):
                    on_click(x, y)
            else:
                c.button(label, key=f"{key_prefix}{x}-{y}", disabled=True,
                         use_container_width=True)

def _summary_table(shots: dict, grid: dict) -> pd.DataFrame:
    data = [[_cell_label(shots.get((x, y)), (x, y) in grid, True)
             for x in range(GRID)] for y in range(GRID)]
    return pd.DataFrame(data, index=list("ABCDEFGHIJ"),
                        columns=[str(i) for i in range(GRID)])

def _side_panel(journal: list[str]):
    st.subheader("🛡️ Ta flotte")
    st.dataframe(_summary_table(st.session_state.enemy_shots, st.session_state.my_grid),
                 use_container_width=True)
    st.subheader("📜 Journal")
    for msg in reversed(journal[-10:]):
        st.caption(msg)

def _exit_button(on_exit):
    if on_exit and st.button("🏠 Retour au menu"):
        on_exit()
        st.rerun()

# ================= IA (temporaire : heuristique simple) =================

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
        st.rerun()
    if c3.button("⬅️ Annuler dernier") and ss.my_grid:
        last_name = ss.my_grid[next(reversed(ss.my_grid))]
        ss.my_grid = {c: n for c, n in ss.my_grid.items() if n != last_name}
        ss.placing = max(0, ss.placing - 1)
        st.rerun()

    def try_place(x, y):
        cells = can_place(ss.my_grid, x, y, size, ss.orientation == "H")
        if not cells:
            log("Placement invalide ⛔")
            return
        for c in cells:
            ss.my_grid[c] = name
        ss.placing += 1
        if ss.placing >= len(SHIPS):
            _finish_placement(on_ships_placed)
        st.rerun()

    _render_grid({}, ss.my_grid, True, "place_", clickable=True, on_click=try_place)

def _finish_placement(on_ships_placed):
    ss = st.session_state
    ss.phase = "battle"
    if ss.vs_ai and not ss.enemy_grid:
        ss.enemy_grid = random_grid()
    on_ships_placed(grid_to_ships(ss.my_grid))   # → app.py : bdd.save_board

# ================= PHASE BATAILLE — VS IA =================

def _render_ai_battle(on_shot, on_finish, on_exit):
    ss = st.session_state
    if ss.winner:
        if ss.winner == ss.pseudo:
            st.balloons()
            st.success("**Tu as gagné 🎉**")
        else:
            st.error("**L'IA a gagné 🤖**")
        _exit_button(on_exit)
        return

    def player_shot(x, y):
        result = shot_result(ss.enemy_grid, ss.my_shots, x, y)
        ss.my_shots[(x, y)] = result
        log(f"Tu tires en {(x, y)} : {RESULT_LABEL[result]}")
        on_shot(x, y, result, ss.pseudo)

        if all_sunk(ss.enemy_grid, ss.my_shots):
            ss.winner = ss.pseudo
            on_finish(ss.pseudo)
        else:
            # Riposte de l'IA
            ex, ey = ai_shot()
            r = shot_result(ss.my_grid, ss.enemy_shots, ex, ey)
            ss.enemy_shots[(ex, ey)] = r
            log(f"🤖 L'IA tire en {(ex, ey)} : {RESULT_LABEL[r]}")
            on_shot(ex, ey, r, "IA")
            if all_sunk(ss.my_grid, ss.enemy_shots):
                ss.winner = "IA"
                on_finish("IA")
            else:
                ss.turn += 1
        st.rerun()

    colg, cols_ = st.columns([3, 2])
    with colg:
        st.subheader("🎯 Grille ennemie")
        _render_grid(ss.my_shots, {}, False, "shoot_",
                     clickable=True, on_click=player_shot)
    with cols_:
        _side_panel(ss.log)

# ================= PHASE BATAILLE — PVP =================

def _render_pvp_battle(fetch_state, resolve_shot, on_exit):
    ss = st.session_state
    state = fetch_state()

    if state["status"] == "finished":
        if state["winner"] == ss.pseudo:
            st.balloons()
            st.success("**Tu as gagné 🎉**")
        elif state["winner"]:
            st.error(f"**{state['winner']}** a gagné.")
        else:
            st.info("Partie terminée.")
        _exit_button(on_exit)
        return

    # La BDD fait foi : on reconstruit les tirs à chaque rendu
    ss.my_shots = {(m["x"], m["y"]): m["result"]
                   for m in state["moves"] if m["player"] == ss.pseudo}
    ss.enemy_shots = {(m["x"], m["y"]): m["result"]
                      for m in state["moves"] if m["player"] != ss.pseudo}
    journal = [f"{m['player']} tire en ({m['x']},{m['y']}) : {RESULT_LABEL[m['result']]}"
               for m in state["moves"]]

    st.caption(f"⚔️ Adversaire : **{state['opponent']}**")

    if not state["opponent_ready"]:
        st.warning("L'adversaire place encore sa flotte…")
        _poll()
        return

    colg, cols_ = st.columns([3, 2])
    with colg:
        if state["my_turn"]:
            st.subheader("🎯 Grille ennemie — à toi de tirer !")

            def shoot(x, y):
                try:
                    resolve_shot(x, y)
                except ValueError as e:
                    st.warning(str(e))
                    return
                st.rerun()

            _render_grid(ss.my_shots, {}, False, "shoot_",
                         clickable=True, on_click=shoot)
        else:
            st.subheader("🎯 Grille ennemie — tour adverse…")
            _render_grid(ss.my_shots, {}, False, "shoot_", clickable=False)
    with cols_:
        _side_panel(journal)

    if not state["my_turn"]:
        st.info("En attente du tir adverse — rafraîchissement automatique.")
        _poll()

# ================= POINT D'ENTRÉE =================

def render(*, on_shot=None, on_finish=None, on_ships_placed=None,
           fetch_state=None, resolve_shot=None, on_exit=None):
    """Hooks fournis par app.py (persistance PostgreSQL + events Kafka)."""
    ss = st.session_state
    if ss.phase == "placement":
        render_placement(on_ships_placed)
    elif ss.vs_ai:
        _render_ai_battle(on_shot, on_finish, on_exit)
    else:
        _render_pvp_battle(fetch_state, resolve_shot, on_exit)