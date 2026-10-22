"""training_IA.py — Entraînement des modèles (service vscode-tensorflow-gpu).

Deux modèles :

1. DQN pour les TIRS (models/dqn_shooter.keras) 
   - Phase OFFLINE : transitions (s, a, r, s', done) reconstruites depuis
     gold/ml_dataset (replay des parties Kafka → S3)
   - Phase ONLINE  : self-play dans BattleshipEnv (epsilon-greedy, target network)
   - Rewards définis dans game_logic.py (hit +1, sunk +3, win +10)

   upgrade : 
   -MLP CNN
   -PER
   -double dqn, dueling dqn
   - n step dqn
   - lyapunov-basd dqn

2. RÉGRESSION pour les PLACEMENTS (models/placement_regressor.keras)
   - Entrée : placement[100] + adjacent_pairs + edge_cells (gold/placements)
   - Sortie : hits_taken prédit (dégâts subis — à minimiser)

Usage :
    python training_IA.py                          # tout : DQN offline + online + régression
    python training_IA.py --episodes 3000          # plus de self-play
    python training_IA.py --placement-only         # que la régression
    python training_IA.py --offline-epochs 0       # sauter le pré-entraînement offline
    python training_IA.py --upload                 # pousser les .keras sur S3 ensuite
    python training_IA.py --offline-epochs 3 --episodes 100 --upload --fresh

Deps : tensorflow, numpy, pandas, pyarrow, boto3 (image Onyxia TF-GPU).

Point de vigilance ML : l'encodage scalaire ordinal (0 < 1 < 2 < 3) fait croire au CNN que "coulé" est une quantité plus grande que "touché" — alors que ce sont des états qualitatifs différents. Ça marche (le réseau apprend à s'adapter), mais l'encodage idéal pour un CNN serait du one-hot par canal : entrée (10, 10, 4) avec un canal par état. Chaque filtre réagit alors à "est-ce un touché ?" plutôt qu'à "valeur ≈ ⅔". C'est le prochain upgrade naturel (et rétro-compatible : la conversion scalaire → canaux se fait dans build_transitions côté offline). À considérer si les résultats plafonnent — garde d'abord l'encodage actuel pour voir la courbe.
"""
import argparse
import os
import random
import sys
import tempfile
from collections import deque
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from botocore.exceptions import ClientError
from tensorflow import keras

import config
from game_logic import (GRID, R_HIT, R_MISS, R_SUNK, R_WIN, SHIP_CELLS,
                        STATE_SCALE, BattleshipEnv)
from use_IA import greedy_cell

SHOOTER_NAME_PRE = "dn_shooter.keras"
SHOOTER_NAME_DQN = "dqn2_shooter.keras"
PLACEMENT_NAME = "placement_regressor.keras"

MODEL_DIR = "models"
LOG_DIR = "logs"
GAMMA = 0.99          # 0.95
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
TARGET_SYNC = 1_000       # steps entre chaque copie model → target
EPS_START = 1.0
EPS_MIN = 0.05
LEARNING_RATE = 0.0003

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # logs TF moins verbeux


def configure_gpu() -> None:
    """TensorFlow utilise le GPU automatiquement ; on configure juste le
    memory growth (n'alloue la VRAM qu'au fur et à mesure, propre sur un
    service mutualisé) et on logge clairement ce qui est détecté.

    NB : pas de mixed_float16 ici — pour un DQN la stabilité numérique des
    Q-values prime, et le modèle est trop petit pour en profiter.
    """
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"✅ GPU actif ({len(gpus)}×) : {gpus[0].name}")
    else:
        print("⚠️  Aucun GPU visible — entraînement CPU "
              "(vérifie que tu es bien sur l'image vscode-tensorflow-GPU)")


# ==================== LOGGING FICHIER ====================

class _Tee:
    """Duplique stdout/stderr vers un fichier, en flushant à chaque écriture.
    Si le service Onyxia redémarre ou que la session shell se perd, le log
    complet reste sur disque (et peut être poussé sur S3 avec --upload)."""

    def __init__(self, path: str):
        self.file = open(path, "w", encoding="utf-8")
        self.stream = sys.__stdout__

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()          # chaque ligne survit à un crash

    def flush(self):
        self.stream.flush()
        self.file.flush()


def start_logging() -> str:
    """Redirige stdout/stderr vers logs/train_<timestamp>.txt. Renvoie le path."""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"train_{datetime.now():%Y%m%d_%H%M%S}.txt")
    sys.stdout = sys.stderr = _Tee(path)
    print(f"📝 Log : {path}")
    return path


# ==================== DONNÉES S3 ====================

def load_gold_table(table: str) -> pd.DataFrame:
    """Télécharge les parquet de gold/<table> depuis S3 → DataFrame pandas."""
    s3 = boto3.client("s3", endpoint_url=config.S3_ENDPOINT)
    prefix = f"bataille-navale/gold/{table}/"
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    s3.download_file(config.S3_BUCKET, obj["Key"],
                                     os.path.join(tmp, f"{n}.parquet"))
                    n += 1
        if n == 0:
            raise FileNotFoundError(
                f"Aucun parquet sous s3://{config.S3_BUCKET}/{prefix} — "
                "lance spark_bronze.py, spark_silver.py puis spark_gold.py")
        print(f"gold/{table} : {n} fichier(s) parquet")
        return pd.read_parquet(tmp)


def build_transitions(df: pd.DataFrame) -> list[tuple]:
    """(s, a, r, s', done) depuis gold/ml_dataset — replay des parties.

    s' = état avant le tir suivant du même joueur ; done = dernier tir.
    Reward : hit +1, sunk +3 (1+2), +10 si dernier tir ET victoire.
    """
    transitions = []
    for _, g in df.groupby(["game_id", "player"]):
        g = g.sort_values("turn")
        states = np.array(g["features"].tolist(), dtype=np.float32) / STATE_SCALE
        actions = g["action_cell"].to_numpy()
        hits = g["label_hit"].to_numpy()
        sunk = (g["result"] == "sunk").to_numpy()
        wins = g["label_win"].to_numpy()
        for k in range(len(g)):
            r = (R_HIT if hits[k] else R_MISS) + (R_SUNK - R_HIT if sunk[k] else 0.0)
            done = k == len(g) - 1
            if done and wins[k] == 1:
                r += R_WIN
            s2 = states[k + 1] if not done else np.zeros(GRID * GRID, np.float32)
            transitions.append((states[k], int(actions[k]), np.float32(r), s2, done))
    return transitions


# ==================== DQN====================

def build_dqn() -> keras.Model:
    """CNN sur la grille 10×10 : la contiguïté des bateaux est spatiale.

    jit_compile=False : XLA fusionne Conv2D+Bias+ReLU en un kernel que
    l'autotuner cuDNN ne configure pas (bug connu T4 / convs minuscules).
    Sans XLA, les ops passent par cuDNN classique — perte de perf négligeable
    à cette taille de modèle.
    """
    inp = keras.Input(shape=(GRID, GRID, 1))
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(GRID * GRID)(x)   # Q(s, a) pour chaque case
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss=keras.losses.Huber(), jit_compile=False)
    return model


class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE):
        self.buf = deque(maxlen=capacity)

    def add(self, *transition):
        self.buf.append(transition)

    def sample(self, n: int) -> list:
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


def train_batch(model, target, batch, gamma: float = GAMMA) -> None:
    states = np.stack([t[0] for t in batch]).reshape(-1, GRID, GRID, 1)
    next_flat = np.stack([t[3] for t in batch])                       # (B, 100)
    actions = np.array([t[1] for t in batch])
    rewards = np.array([t[2] for t in batch], np.float32)
    dones = np.array([t[4] for t in batch], np.float32)

    # Cible : r + γ · max Q(s', a') sur les actions VALIDES (cases inconnues)
    next_q = target.predict(next_flat.reshape(-1, GRID, GRID, 1), verbose=0)
    next_q = np.where(next_flat == 0, next_q, -1e9)    # masking
    targets = rewards + gamma * (1.0 - dones) * next_q.max(axis=1)

    q = model.predict(states, verbose=0)
    q[np.arange(len(batch)), actions] = targets
    model.train_on_batch(states, q)


def act(model, state: np.ndarray, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.choice([c for c in range(GRID * GRID) if state[c] == 0])
    return greedy_cell(model, state)


def evaluate_solo(model, n: int = 50) -> float:
    """Tirs moyens pour couler la flotte (greedy). Métrique clé d'efficacité."""
    env = BattleshipEnv()
    turns = []
    for _ in range(n):
        state = np.array(env.reset(), np.float32) / STATE_SCALE
        done = False
        while not done:
            _, _, done, _ = env.step(greedy_cell(model, state))
            state = np.array(env.state(), np.float32) / STATE_SCALE
        turns.append(env.turns)
    return float(np.mean(turns))


def train_dqn(args) -> keras.Model:
    # Reprise du bon checkpoint : offline → reprendre le pré-entraîné ;
    # online-only → priorité au DQN affiné, sinon le pré-entraîné, sinon neuf.
    if args.offline_epochs == 0:
        candidates = [SHOOTER_NAME_DQN, SHOOTER_NAME_PRE]
    else:
        candidates = [SHOOTER_NAME_PRE]
    resume_path = next(
        (os.path.join(MODEL_DIR, n) for n in candidates
         if os.path.exists(os.path.join(MODEL_DIR, n))), None)

    if resume_path and not args.fresh:
        print(f"♻️  Reprise depuis {resume_path}")
        model = keras.models.load_model(resume_path)
        target = build_dqn()   # recréé puis copié — un seul fichier suffit
    else:
        print("🆕 Initialisation : réseau neuf")
        model, target = build_dqn(), build_dqn()
    target.set_weights(model.get_weights())
    buf = ReplayBuffer()

    # ---------- Phase offline : replay des données réelles/simulées ----------
    if args.offline_epochs > 0:
        df = load_gold_table("ml_dataset")
        transitions = build_transitions(df)
        print(f"Offline : {len(transitions)} transitions, "
              f"{args.offline_epochs} epoch(s)")
        for t in transitions:
            buf.add(*t)
        for epoch in range(args.offline_epochs):
            random.shuffle(transitions)
            for i in range(0, len(transitions) - BATCH_SIZE, BATCH_SIZE):
                train_batch(model, target, transitions[i:i + BATCH_SIZE])
            target.set_weights(model.get_weights())
            print(f"  epoch {epoch + 1}/{args.offline_epochs} — "
                  f"solo moyen : {evaluate_solo(model, 20):.1f} tirs")
        path = os.path.join(MODEL_DIR, SHOOTER_NAME_PRE)
        model.save(path)
        print(f"Pré-learning sauvegardé : {path} — solo moyen final : "
            f"{evaluate_solo(model, 100):.1f} tirs (random ≈ 95, heuristic ≈ 55-60)")

    # ---------- Phase online : self-play dans l'environnement ----------
    print("Phase online started")
    env = BattleshipEnv()
    # Après un pré-entraînement, partir bas : le réseau sait déjà jouer,
    # inutile de re-explorer massivement (ça dilue ce qu'il vient d'apprendre)
    epsilon = 0.3 if args.offline_epochs > 0 else EPS_START
    decay = (epsilon - EPS_MIN) / max(1, int(args.episodes * 0.7))
    steps = 0
    for ep in range(args.episodes):
        state = np.array(env.reset(), np.float32) / STATE_SCALE
        done = False
        while not done:
            a = act(model, state, epsilon)
            ns, r, done, _ = env.step(a)
            ns = np.array(ns, np.float32) / STATE_SCALE
            buf.add(state, a, np.float32(r), ns, done)
            state = ns
            if len(buf) >= BATCH_SIZE:
                train_batch(model, target, buf.sample(BATCH_SIZE))
            steps += 1
            if steps % TARGET_SYNC == 0:
                target.set_weights(model.get_weights())
        epsilon = max(EPS_MIN, epsilon - decay)
        if (ep + 1) % 100 == 0:
            print(f"épisode {ep + 1}/{args.episodes} — ε={epsilon:.2f} — "
                  f"solo moyen : {evaluate_solo(model):.1f} tirs "
                  f"(partie jouée en {env.turns} tirs)")

    path = os.path.join(MODEL_DIR, SHOOTER_NAME_DQN)
    model.save(path)
    print(f"DQN sauvegardé : {path} — solo moyen final : "
          f"{evaluate_solo(model, 100):.1f} tirs (random ≈ 95, heuristic ≈ 55-60)")
    return model


# ==================== RÉGRESSION PLACEMENT ====================

def train_placement() -> keras.Model:
    # PLACEMENT_NAME
    df = load_gold_table("placements")
    print(f"Placement : {len(df)} lignes")
    if len(df) < 200:
        print("Peu de données de placement — génère des parties "
              "(simulate_games.py) pour un signal exploitable.")

    X_place = np.array(df["placement"].tolist(), dtype=np.float32)
    extra = df[["adjacent_pairs", "edge_cells"]].to_numpy(np.float32) / SHIP_CELLS
    X = np.concatenate([X_place, extra], axis=1)
    y = (df["hits_taken"] / SHIP_CELLS).to_numpy(np.float32)

    idx = np.random.permutation(len(X))
    cut = int(len(X) * 0.8)
    Xtr, Xval = X[idx[:cut]], X[idx[cut:]]
    ytr, yval = y[idx[:cut]], y[idx[cut:]]

    model = keras.Sequential([
        keras.Input(shape=(X.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae",
                  jit_compile=False)   # même précaution XLA que le DQN
    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=200, batch_size=64,
              callbacks=[keras.callbacks.EarlyStopping(
                  patience=15, restore_best_weights=True)],
              verbose=0)

    mae = model.evaluate(Xval, yval, verbose=0)
    baseline = float(np.mean(np.abs(yval - ytr.mean())))   # prédire la moyenne
    print(f"Régression placement — MAE : {mae:.4f} | baseline (moyenne) : "
          f"{baseline:.4f}  → le modèle {'apprend 📈' if mae < baseline * 0.95 else 'ne bat pas encore la baseline (il faut plus de données)'}")

    path = os.path.join(MODEL_DIR, PLACEMENT_NAME)
    model.save(path)
    print(f"Régression sauvegardée : {path}")
    return model


# ==================== S3 ====================

def load_models() -> None:
    """Télécharge les .keras depuis s3://<bucket>/bataille-navale/models/."""
    s3 = boto3.client("s3", endpoint_url=config.S3_ENDPOINT)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name in (SHOOTER_NAME_PRE, SHOOTER_NAME_DQN, PLACEMENT_NAME):
        key = f"bataille-navale/models/{name}"
        local = os.path.join(MODEL_DIR, name)
        try:
            s3.head_object(Bucket=config.S3_BUCKET, Key=key)   # test d'existence
        except ClientError:
            print(f"⏭️  {name} absent de S3, ignoré")
            continue
        s3.download_file(config.S3_BUCKET, key, local)   # (Bucket, Key, Filename)
        print(f"✅ {name} ← S3")

def upload_models() -> None:
    """Pousse les .keras vers s3://<bucket>/bataille-navale/models/."""
    s3 = boto3.client("s3", endpoint_url=config.S3_ENDPOINT)
    for name in (SHOOTER_NAME_PRE, SHOOTER_NAME_DQN, PLACEMENT_NAME):
        local = os.path.join(MODEL_DIR, name)
        if os.path.exists(local):
            s3.upload_file(local, config.S3_BUCKET,
                           f"bataille-navale/models/{name}")
            print(f"✅ {name} → S3")


def upload_log(log_path: str) -> None:
    """Copie le log d'entraînement vers s3://<bucket>/bataille-navale/logs/."""
    s3 = boto3.client("s3", endpoint_url=config.S3_ENDPOINT)
    key = f"bataille-navale/logs/{os.path.basename(log_path)}"
    s3.upload_file(log_path, config.S3_BUCKET, key)
    print(f"✅ log → s3://{config.S3_BUCKET}/{key}")


# ==================== MAIN ====================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entraînement IA bataille navale")
    ap.add_argument("--episodes", type=int, default=2000,
                    help="épisodes de self-play DQN")
    ap.add_argument("--offline-epochs", type=int, default=0,
                    help="epochs de pré-entraînement sur gold/ml_dataset (0 = skip)")
    ap.add_argument("--placement-only", action="store_true",
                    help="n'entraîner que la régression de placement")
    ap.add_argument("--upload", action="store_true",
                    help="pousser les modèles sur S3 après entraînement")
    ap.add_argument("--download", action="store_true",
                    help="récupérer les modèles depuis S3 avant entraînement")
    ap.add_argument("--fresh", action="store_true",
                    help="ignorer les checkpoints locaux (réseau neuf)")
    args = ap.parse_args()

    log_path = start_logging()   # tout stdout/stderr part aussi dans le fichier
    os.makedirs(MODEL_DIR, exist_ok=True)
    configure_gpu()

    if args.download:
        load_models()
    if not args.placement_only:
        train_dqn(args)
    #train_placement()
    if args.upload:
        upload_models()
        upload_log(log_path)
