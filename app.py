import streamlit as st
import random
import pandas as pd

st.set_page_config(page_title="Bataille Navale", page_icon="🚢")

GRID = 10
SHIPS = [("Porte-avions", 5), ("Croiseur", 4), ("Contre-torpilleur", 3),
         ("Sous-marin", 3), ("Torpilleur", 2)]

# ---------- État du jeu ----------
def init_game():
    st.session_state.phase = "placement"       # placement | battle | over
    st.session_state.placing = 0               # index du bateau à placer
    st.session_state.orientation = "H"         # H ou V
    st.session_state.my_grid = {}               # {(x,y): nom_bateau}
    st.session_state.enemy_grid = {}            # grille de l'IA
    st.session_state.my_shots = {}             # {(x,y): "miss"/"hit"/"sunk"}
    st.session_state.enemy_shots = {}
    st.session_state.log = []
    st.session_state.turn = "player"

def random_place():
    """Placement aléatoire (utilisé par l'IA et par le bouton auto)."""
    grid, occupied = {}, set()
    for name, size in SHIPS:
        while True:
            horiz = random.choice([True, False])
            x = random.randint(0, GRID - (size if horiz else 1))
            y = random.randint(0, GRID - (size if not horiz else 1))
            cells = {(x+i, y) for i in range(size)} if horiz else {(x, y+i) for i in range(size)}
            if not cells & occupied:
                occupied |= cells
                for c in cells: grid[c] = name
                break
    return grid

if "phase" not in st.session_state:
    init_game()

st.title("🚢 Bataille Navale")

# ---------- Phase placement ----------
if st.session_state.phase == "placement":
    st.subheader(f"Place ton **{SHIPS[st.session_state.placing][0]}** ({SHIPS[st.session_state.placing][1]} cases)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Rotation (H/V)"):
            st.session_state.orientation = "V" if st.session_state.orientation == "H" else "H"
    with col2:
        if st.button("🎲 Placement auto"):
            st.session_state.my_grid = random_place()
            st.session_state.placing = len(SHIPS)
            st.session_state.phase = "battle"
            st.session_state.enemy_grid = random_place()
            st.rerun()
    st.caption(f"Orientation actuelle : {st.session_state.orientation}")

    for y in range(GRID):
        cols = st.columns(GRID)
        for x, c in enumerate(cols):
            if (x, y) in st.session_state.my_grid:
                c.button("⚓", key=f"p{x}-{y}", disabled=True)
            else:
                if c.button("·", key=f"p{x}-{y}", use_container_width=True):
                    size = SHIPS[st.session_state.placing][1]
                    name = SHIPS[st.session_state.placing][0]
                    horiz = st.session_state.orientation == "H"
                    cells = {(x+i, y) for i in range(size)} if horiz else {(x, y+i) for i in range(size)}
                    if all(0 <= cx < GRID and 0 <= cy < GRID and (cx, cy) not in st.session_state.my_grid
                           for cx, cy in cells):
                        for cx, cy in cells:
                            st.session_state.my_grid[(cx, cy)] = name
                        st.session_state.placing += 1
                        if st.session_state.placing >= len(SHIPS):
                            st.session_state.phase = "battle"
                            st.session_state.enemy_grid = random_place()
                        st.rerun()
    st.stop()

# ---------- Helpers d'affichage ----------
def render_grid(shots, grid, is_enemy):
    """Affiche une grille : shots = {(x,y): résultat}, grid = positions bateaux."""
    df = []
    for y in range(GRID):
        row = []
        for x in range(GRID):
            r = shots.get((x, y))
            if r == "miss":     row.append("🌊")
            elif r in ("hit", "sunk"): row.append("💥")
            elif not is_enemy and (x, y) in grid: row.append("⚓")
            else:               row.append("⬜")
        df.append(row)
    return pd.DataFrame(df, index=list("ABCDEFGHIJ"),
                        columns=[str(i) for i in range(10)])

def is_sunk(grid, shots, cell):
    """Vérifie si le bateau touché en `cell` est coulé."""
    name = grid.get(cell)
    if name is None: return False
    return all(shots.get((x, y)) == "hit" or (x, y) == cell
               for (x, y), n in grid.items() if n == name)

def all_sunk(grid, shots):
    return all(any(shots.get((x, y)) in ("hit", "sunk")
               for (x, y) in grid if grid[(x, y)] == name)
               for name, _ in SHIPS)

def ai_shot():
    """L'IA tire au hasard sur une case non touchée (remplacée par TensorFlow plus tard)."""
    options = [(x, y) for x in range(GRID) for y in range(GRID)
               if (x, y) not in st.session_state.enemy_shots]
    return random.choice(options)

# ---------- Phase bataille ----------
colg, coll = st.columns([3, 2])

with colg:
    st.subheader("🎯 Grille ennemie — à toi de tirer")
    for y in range(GRID):
        cols = st.columns(GRID)
        for x, c in enumerate(cols):
            r = st.session_state.my_shots.get((x, y))
            if r:
                c.button({"miss": "🌊", "hit": "💥", "sunk": "☠️"}[r],
                         key=f"s{x}-{y}", disabled=True, use_container_width=True)
            elif c.button("·", key=f"s{x}-{y}", use_container_width=True):
                cell = (x, y)
                if cell in st.session_state.enemy_grid:
                    st.session_state.my_shots[cell] = "hit"
                    if is_sunk(st.session_state.enemy_grid, st.session_state.my_shots, cell):
                        st.session_state.my_shots[cell] = "sunk"
                        st.session_state.log.append("😎 Touché coulé !")
                    else:
                        st.session_state.log.append("💥 Touché !")
                else:
                    st.session_state.my_shots[cell] = "miss"
                    st.session_state.log.append("🌊 Raté...")
                    st.rerun()  # pas de rerun ici : l'IA jouera au prochain render
                # Tour de l'IA (si le joueur a raté, ou après chaque tir)
                ex, ey = ai_shot()
                hit = (ex, ey) in st.session_state.my_grid
                st.session_state.enemy_shots[(ex, ey)] = "hit" if hit else "miss"
                st.session_state.log.append(f"🤖 L'IA tire en {(ex, ey)} : "
                                            "💥 touché !" if hit else "🌊 dans l'eau")
                if all_sunk(st.session_state.enemy_grid, st.session_state.my_shots):
                    st.session_state.phase = "over"; st.session_state.winner = "Toi 🎉"
                elif all_sunk(st.session_state.my_grid, st.session_state.enemy_shots):
                    st.session_state.phase = "over"; st.session_state.winner = "L'IA 🤖"
                st.rerun()

with coll:
    st.subheader("🛡️ Ta flotte")
    st.dataframe(render_grid(st.session_state.enemy_shots, st.session_state.my_grid, False),
                 use_container_width=True, hide_index=False)
    st.subheader("📜 Journal")
    for msg in reversed(st.session_state.log[-8:]):
        st.write(msg)

if st.session_state.phase == "over":
    st.balloons()
    st.success(f"**{st.session_state.winner}** a gagné !")
    if st.button("🔄 Nouvelle partie"):
        init_game()
        st.rerun()

if st.sidebar.button("🔁 Recommencer"):
    init_game()
    st.rerun()