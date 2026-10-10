"""app.py — Application Streamlit : bataille navale (IA + PvP)."""
import time

import streamlit as st

import bdd
import game
import kafka_utils

st.set_page_config(page_title="Bataille Navale", page_icon="🚢")

GAME_KEYS = ["game_id", "opponent", "match_status"]


def reset_game_state() -> None:
    """Retour au menu : purge l'état de la partie en cours."""
    for key in GAME_KEYS + game.STATE_KEYS:
        st.session_state.pop(key, None)


# ---------- 0. INIT ----------
if "initialized" not in st.session_state:
    bdd.init_schema()              # idempotent, safe au reload
    kafka_utils.ensure_topics()    # idempotent, tolérant aux pannes
    st.session_state.initialized = True

# ---------- 1. LOGIN ----------
if "pseudo" not in st.session_state:
    st.title("🚢 Bataille Navale")
    pseudo_input = st.text_input("Ton pseudo", max_chars=20)
    if st.button("Jouer", type="primary") and pseudo_input.strip():
        st.session_state.pseudo = pseudo_input.strip()
        bdd.login(st.session_state.pseudo)
        kafka_utils.send_event("login", player=st.session_state.pseudo)
        st.rerun()
    st.stop()

pseudo = st.session_state.pseudo
bdd.touch(pseudo)  # heartbeat : reste visible pour le matchmaking

# ---------- 2. MENU ----------
if "game_id" not in st.session_state:
    st.title(f"Salut {pseudo} 👋")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🤖 Jouer contre l'IA", type="primary"):
            st.session_state.game_id = bdd.create_ai_game(pseudo)
            st.session_state.opponent = None
            st.session_state.match_status = "playing"
            game.init_game(vs_ai=True)
            kafka_utils.send_event("game_start",
                                   game_id=st.session_state.game_id,
                                   players=[pseudo], mode="ai")
            st.rerun()
    with c2:
        if st.button("👥 Matchmaking"):
            match = bdd.join_matchmaking(pseudo)
            st.session_state.game_id = match["game_id"]
            st.session_state.opponent = match["opponent"]
            st.session_state.match_status = match["status"]
            if match["status"] == "playing":
                # J'ai rejoint la file d'un autre joueur
                game.enter_pvp_game(match["game_id"], match["opponent"])
                kafka_utils.send_event("match_found",
                                       players=[pseudo, match["opponent"]])
            st.rerun()
    with c3:
        if st.button("🚪 Déconnexion"):
            bdd.logout(pseudo)
            st.session_state.clear()
            st.rerun()
    st.stop()

game_id = st.session_state.game_id

# ---------- 3. SALLE D'ATTENTE (matchmaking) ----------
if st.session_state.match_status == "waiting":
    st.title("🔍 Recherche d'un adversaire…")

    # Idempotent : si une autre file est apparue (ou qu'un joueur m'a rejoint),
    # fusionne les files / détecte le match. Garantit que deux files parallèles
    # finissent toujours par se rejoindre, même en cas de clic simultané.
    match = bdd.join_matchmaking(pseudo)
    st.session_state.game_id = match["game_id"]
    if match["status"] == "playing" and match["opponent"]:
        st.session_state.opponent = match["opponent"]
        st.session_state.match_status = "playing"
        game.enter_pvp_game(match["game_id"], match["opponent"])
        st.rerun()

    st.info(f"Joueurs en file : {bdd.waiting_count()} — la page se rafraîchit toute seule.")
    if st.button("❌ Annuler la recherche"):
        bdd.cancel_waiting_game(match["game_id"], pseudo)
        reset_game_state()
        st.rerun()
    # Polling serveur : fonctionne aussi dans un onglet en arrière-plan,
    # contrairement aux timers JS des composants d'auto-refresh.
    time.sleep(2)
    st.rerun()

opponent = st.session_state.opponent

# ---------- 4. PARTIE ----------
if opponent is None:
    # ----- Contre l'IA : calcul local, persistance simple -----
    game.render(
        on_shot=lambda x, y, result, shooter: (
            bdd.save_move(game_id, shooter, st.session_state.turn, x, y, result),
            kafka_utils.send_move(game_id, shooter, st.session_state.turn, x, y, result,
                                  extra={"mode": "ai"}),
        ),
        on_finish=lambda winner: (
            bdd.finish_game(game_id, winner),
            kafka_utils.send_event("game_over", game_id=game_id, winner=winner),
        ),
        on_ships_placed=lambda ships: bdd.save_board(game_id, pseudo, ships),
        on_exit=reset_game_state,
    )
else:
    # ----- PvP : la BDD fait foi, tirs résolus côté serveur -----
    def resolve_shot(x: int, y: int) -> dict:
        shot = bdd.record_shot(game_id, pseudo, x, y)
        kafka_utils.send_move(game_id, pseudo, shot["turn"], x, y, shot["result"],
                              extra={"mode": "pvp", "opponent": opponent})
        if shot["winner"]:
            kafka_utils.send_event("game_over", game_id=game_id, winner=shot["winner"])
        return shot

    game.render(
        fetch_state=lambda: bdd.get_game_state(game_id, pseudo),
        resolve_shot=resolve_shot,
        on_ships_placed=lambda ships: bdd.save_board(game_id, pseudo, ships),
        on_exit=reset_game_state,
    )