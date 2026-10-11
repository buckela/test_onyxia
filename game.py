"""Logique de la bataille navale : logique pure + rendu Streamlit.

- vs_ai : tout est calculé côté client (riposte immédiate de l'IA)
- PvP   : la BDD fait foi — app.py fournit fetch_state / resolve_shot,
          les tirs adverses sont récupérés par polling (auto-refresh)

La logique pure (grille, tirs, placements) vit dans game_logic.py, module
partagé avec les scripts headless (simulation, entraînement, inférence).
"""
import time

import streamlit as st

import bdd
from game_logic import (GRID, RESULT_LABEL, SHIPS, all_sunk, can_place,
                        grid_to_features, grid_to_ships, heuristic_shot,
                        random_grid, ships_to_grid, shot_result)

# Cases déjà tirées : bien visibles (gros symboles, pas de faible nuance)
CELL_MISS = "❌"   # tir raté
CELL_HIT = "💥"    # bateau touché
CELL_SUNK = "☠️"   # bateau coulé
CELL_SHIP = "🚢"   # ton bateau intact (visible sur TA grille)
CELL_SEA = "🟦"    # case d'eau non tirée (visible sur TA grille)
CELL_UNKNOWN = "·" # case ennemie cliquable

# CSS injecté : boutons de grille plus grands et lisibles
_GRID_CSS = """
<style>
  div[data-testid="stButton"] > button {
    min-height: 44px;
    font-size: 1.2rem;
    padding: 0;
  }
</style>
"""

# Clés posées par init_game() — utilisées par app.py pour le reset
STATE_KEYS = ["phase", "vs_ai", "placing", "orientation", "my_grid", "enemy_grid",
              "my_shots", "enemy_shots", "turn", "log", "winner", "place_policy"]

# ================= LOGIQUE PURE =================
# → déplacée dans game_logic.py (importée ci-dessus)

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
    ss.place_policy = "human"   # comment la flotte a été placée (human | random_v1)
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
    """Libellé d'une case, bien contrasté entre cases tirées et non tirées."""
    if shot == "miss":
        return CELL_MISS
    if shot == "hit":
        return CELL_HIT
    if shot == "sunk":
        return CELL_SUNK
    if show_ship:
        return CELL_SHIP if has_ship else CELL_SEA
    return CELL_UNKNOWN

def _render_grid(shots: dict, grid: dict, show_ships: bool,
                 key_prefix: str, clickable: bool = False, on_click=None):
    for y in range(GRID):
        cols = st.columns(GRID)
        for x, c in enumerate(cols):
            shot = shots.get((x, y))
            has_ship = (x, y) in grid
            label = _cell_label(shot, has_ship, show_ships)
            if clickable and shot is None and not has_ship and on_click:
                if c.button(CELL_UNKNOWN, key=f"{key_prefix}{x}-{y}",
                            use_container_width=True):
                    on_click(x, y)
            else:
                c.button(label, key=f"{key_prefix}{x}-{y}", disabled=True,
                         use_container_width=True)

def _side_panel(journal: list[str]):
    """Ta flotte + journal. Grille en boutons (plus grande que le dataframe)."""
    st.subheader("🛡️ Ta flotte")
    _render_grid(st.session_state.enemy_shots, st.session_state.my_grid,
                 True, "own_", clickable=False)
    st.subheader("📜 Journal")
    for msg in reversed(journal[-10:]):
        st.caption(msg)

def _exit_button(on_exit):
    if on_exit and st.button("🏠 Retour au menu"):
        on_exit()
        st.rerun()

def ai_shot(shots: dict | None = None) -> tuple[int, int]:
    """Prochain tir de l'IA. Délègue à game_logic.heuristic_shot ;
    par défaut lit la session Streamlit (comportement du jeu)."""
    if shots is None:
        shots = st.session_state.enemy_shots
    return heuristic_shot(shots)

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
        ss.place_policy = "random_v1"
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
        ss.place_policy = "human"
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
    on_ships_placed(grid_to_ships(ss.my_grid), ss.pseudo, ss.place_policy)
    if ss.vs_ai:
        # la flotte IA part aussi dans Kafka (baseline 'random_v1' pour le gold)
        on_ships_placed(grid_to_ships(ss.enemy_grid), "IA", "random_v1")
    ss.place_policy = "human"

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

    colg, cols_ = st.columns([1, 1])
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

    colg, cols_ = st.columns([1, 1])
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
    st.markdown(_GRID_CSS, unsafe_allow_html=True)
    ss = st.session_state
    if ss.phase == "placement":
        render_placement(on_ships_placed)
    elif ss.vs_ai:
        _render_ai_battle(on_shot, on_finish, on_exit)
    else:
        _render_pvp_battle(fetch_state, resolve_shot, on_exit)