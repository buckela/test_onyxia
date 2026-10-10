import streamlit as st
import game
import bdd, kafka
import uuid

st.set_page_config(page_title="Bataille Navale", page_icon="🚢")

if "initialized" not in st.session_state:
    bdd.init_schema()          # idempotent, safe au reload
    st.session_state.initialized = True

# ---------- 1. LOGIN ----------
if "pseudo" not in st.session_state:
    st.title("🚢 Bataille Navale")
    pseudo = st.text_input("Ton pseudo", max_chars=20)
    if st.button("Jouer", type="primary") and pseudo.strip():
        st.session_state.pseudo = pseudo.strip()
        bdd.login(st.session_state.pseudo)
        kafka.send_event("login", player=st.session_state.pseudo)
        st.rerun()
    st.stop()

# ---------- 2. MENU ----------
if "game_id" not in st.session_state:
    st.title(f"Salut {st.session_state.pseudo} 👋")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🤖 Jouer contre l'IA", type="primary"):
            st.session_state.game_id = bdd.create_game(st.session_state.pseudo, None)
            st.session_state.opponent = "IA"
            game.init_game()
    with c2:
        if st.button("👥 Matchmaking"):
            opp = bdd.find_opponent(st.session_state.pseudo)
            if opp:
                st.session_state.game_id = bdd.create_game(st.session_state.pseudo, opp)
                st.session_state.opponent = opp
                kafka.send_event("match_found", players=[st.session_state.pseudo, opp])
                game.init_game()
            else:
                st.warning("Personne d'autre en ligne... essaie l'IA !")
    with c3:z
        if st.button("🚪 Déconnexion"):
            bdd.set_online(st.session_state.pseudo, False)
            for k in list(st.session_state):
                del st.session_state[k]
            st.rerun()
    st.stop()

# ---------- 3. PARTIE ----------
# (la logique de jeu de la réponse précédente, déplacée dans game.py,
#  avec ce hook sur chaque tir :)
def on_shot(x, y, result, turn):
    bdd.save_move(st.session_state.game_id, st.session_state.pseudo, turn, x, y, result)
    kafka.send_move(st.session_state.game_id, st.session_state.pseudo, turn, x, y, result)

def on_finish(winner):
    bdd.finish_game(st.session_state.game_id, winner)
    kafka.send_event("game_over", game_id=st.session_state.game_id, winner=winner)

game.render(
    on_shot=lambda x, y, result: (
        bdd.save_move(st.session_state.game_id, st.session_state.pseudo,
                      st.session_state.turn, x, y, result),
        kafka.safe_send_move(st.session_state.game_id, st.session_state.pseudo,
                             st.session_state.turn, x, y, result),
    ),
    on_finish=lambda winner: (
        bdd.finish_game(st.session_state.game_id, winner),
        kafka.send_event("game_over", game_id=st.session_state.game_id, winner=winner),
    ),
    on_ships_placed=lambda grid: (
        bdd.save_board(st.session_state.game_id, st.session_state.pseudo, grid),
    ),
)