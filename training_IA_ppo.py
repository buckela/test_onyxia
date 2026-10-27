"""training_IA_ppo.py — PPO actor-critic pour les tirs à la bataille navale.

Entraînement on-policy dans BattleshipEnv ; aucun accès à S3 n'est requis.
Les flottes d'entraînement sont aléatoires. La validation utilise un ensemble
fixe et distinct de flottes, comparé aux policies random et heuristic.

Exemples :
    python training_IA_ppo.py --timesteps 500000
    python training_IA_ppo.py --timesteps 1000000 --eval-episodes 200 --upload
    tensorboard --logdir logs/ppo

Mesures enregistrées dans logs/ppo/<run>/ :
- metrics.csv et TensorBoard : pertes acteur/critique, entropie, KL,
  explained variance, retour, tirs et epsilon d'action (aucun epsilon PPO).
- Espérance, variance, écart-type et IC 95 % des tirs sur validation fixe.
- Écart train/validation : indicateur de généralisation, pas une décomposition
  statistique exacte du biais/variance.

Important : le modèle PPO est distinct des modèles DQN utilisés par use_IA.py.
Il est sauvegardé sous models/ppo_shooter.keras ; il ne remplace pas le DQN.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from game_logic import (
    GRID,
    STATE_SCALE,
    BattleshipEnv,
    heuristic_shot,
    random_grid,
    shot_result,
    all_sunk,
)

MODEL_DIR = Path("models")
LOG_ROOT = Path("logs") / "ppo"
MODEL_NAME = "ppo_shooter.keras"


class ShapedEnv(BattleshipEnv):
    # La somme des bonus hit/sunk/win est constante par partie : seul le coût des tirs ratés distingue deux policies.
    rewards = {"miss": -0.1, "hit": 0.0, "sunk": 0.0, "win": 0.0}

    def step(self, cell: int):
        state, _, done, info = super().step(cell)
        reward = self.rewards[info["result"]] + (self.rewards["win"] if done else 0.0)
        return state, reward, done, info

CSV_FIELDS = [
    "update", "timesteps", "train_return_mean", "train_shots_mean",
    "policy_loss", "value_loss", "entropy", "approx_kl",
    "clip_fraction", "explained_variance", "train_eval_shots_mean",
    "validation_shots_mean",
    "validation_shots_std", "validation_shots_variance",
    "validation_shots_ci95_low", "validation_shots_ci95_high",
    "validation_return_mean", "validation_gap_shots",
    "random_shots_mean", "heuristic_shots_mean",
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"⚠️ Impossible de configurer la mémoire de {gpu.name}: {exc}")
    if gpus:
        print("✅ GPU : " + ", ".join(gpu.name for gpu in gpus))
    else:
        print("⚠️ Aucun GPU détecté ; entraînement sur CPU")


def start_logging(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.txt"
    stream = open(log_path, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout

    class Tee:
        def write(self, value):
            original_stdout.write(value)
            stream.write(value)
            stream.flush()

        def flush(self):
            original_stdout.flush()
            stream.flush()

        def isatty(self):
            return original_stdout.isatty()

    sys.stdout = Tee()
    return stream


def build_actor_critic(learning_rate: float) -> keras.Model:
    """Backbone CNN partagé, tête policy (logits 100) et tête valeur."""
    observation = keras.Input(shape=(GRID, GRID, 1), name="observation")
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(observation)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    logits = keras.layers.Dense(GRID * GRID, name="policy_logits")(x)
    value = keras.layers.Dense(1, name="state_value")(x)
    model = keras.Model(observation, [logits, value], name="ppo_actor_critic")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  jit_compile=False)
    return model


def normalized_observation(state: list[int] | np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.float32).reshape(GRID, GRID, 1) / STATE_SCALE


def masked_logits(logits: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
    """Les actions interdites ont une probabilité nulle dans la policy."""
    return tf.where(masks, logits, tf.constant(-1.0e9, dtype=logits.dtype))


def sample_action(model: keras.Model, state: np.ndarray) -> tuple[int, float, float]:
    """Échantillonne une action légale et renvoie action, log-proba, valeur."""
    mask = state.reshape(-1) == 0.0
    if not np.any(mask):
        raise RuntimeError("Aucune action valide alors que l'épisode n'est pas fini")
    logits, value = model(state[None, ...], training=False)
    logits = masked_logits(logits[0], tf.convert_to_tensor(mask))
    action = int(tf.random.categorical(logits[None, :], 1)[0, 0].numpy())
    log_probs = tf.nn.log_softmax(logits)
    return action, float(log_probs[action].numpy()), float(value[0, 0].numpy())


def greedy_action(model: keras.Model, state: np.ndarray) -> int:
    mask = state.reshape(-1) == 0.0
    logits, _ = model(state[None, ...], training=False)
    logits = masked_logits(logits[0], tf.convert_to_tensor(mask))
    return int(tf.argmax(logits).numpy())


def collect_rollout(model: keras.Model, env: BattleshipEnv, state: np.ndarray,
                    horizon: int, training_fleets: list[dict],
                    fleet_cursor: int) -> tuple[dict[str, np.ndarray], np.ndarray,
                                               np.ndarray, float, np.ndarray, int]:
    """Collecte une suite on-policy ; redémarre l'environnement aux terminaisons."""
    obs, masks, actions, old_log_probs = [], [], [], []
    rewards, dones, values = [], [], []
    completed_returns: list[float] = []
    completed_shots: list[int] = []
    episode_return = 0.0

    for _ in range(horizon):
        mask = state.reshape(-1) == 0.0
        action, log_prob, value = sample_action(model, state)
        next_raw, reward, done, _ = env.step(action)

        obs.append(state)
        masks.append(mask)
        actions.append(action)
        old_log_probs.append(log_prob)
        rewards.append(float(reward))
        dones.append(float(done))
        values.append(value)
        episode_return += float(reward)

        if done:
            completed_returns.append(episode_return)
            completed_shots.append(env.turns)
            next_raw = env.reset()
            env.grid = training_fleets[fleet_cursor % len(training_fleets)]
            fleet_cursor += 1
            next_raw = env.state()
            episode_return = 0.0
        state = normalized_observation(next_raw)

    # V(s_{t+1}) bootstrap uniquement à la frontière du rollout.
    _, bootstrap = model(state[None, ...], training=False)
    rollout = {
        "observations": np.asarray(obs, dtype=np.float32),
        "masks": np.asarray(masks, dtype=bool),
        "actions": np.asarray(actions, dtype=np.int32),
        "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
    }
    return (rollout, np.asarray(completed_returns, dtype=np.float32),
            np.asarray(completed_shots, dtype=np.int32),
            float(bootstrap[0, 0].numpy()), state, fleet_cursor)


def compute_gae(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray,
                bootstrap_value: float, gamma: float,
                gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation, avec gestion des resets d'épisode."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = bootstrap_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
    return advantages, advantages + values


def ppo_update(model: keras.Model, rollout: dict[str, np.ndarray],
               advantages: np.ndarray, returns: np.ndarray,
               minibatch_size: int, update_epochs: int, clip_ratio: float,
               value_coef: float, entropy_coef: float, max_grad_norm: float,
               target_kl: float) -> dict[str, float]:
    obs = tf.convert_to_tensor(rollout["observations"], dtype=tf.float32)
    masks = tf.convert_to_tensor(rollout["masks"], dtype=tf.bool)
    actions = tf.convert_to_tensor(rollout["actions"], dtype=tf.int32)
    old_log_probs = tf.convert_to_tensor(rollout["old_log_probs"], dtype=tf.float32)
    adv = tf.convert_to_tensor(advantages, dtype=tf.float32)
    adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)
    target_returns = tf.convert_to_tensor(returns, dtype=tf.float32)
    n = int(obs.shape[0])
    metric_rows = []
    stopped_early = False

    for _ in range(update_epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, minibatch_size):
            idx = indices[start:start + minibatch_size]
            idx_tf = tf.convert_to_tensor(idx, dtype=tf.int32)
            batch_obs = tf.gather(obs, idx_tf)
            batch_masks = tf.gather(masks, idx_tf)
            batch_actions = tf.gather(actions, idx_tf)
            batch_old_log_probs = tf.gather(old_log_probs, idx_tf)
            batch_adv = tf.gather(adv, idx_tf)
            batch_returns = tf.gather(target_returns, idx_tf)

            with tf.GradientTape() as tape:
                logits, values = model(batch_obs, training=True)
                logits = masked_logits(logits, batch_masks)
                log_all = tf.nn.log_softmax(logits, axis=-1)
                probs = tf.nn.softmax(logits, axis=-1)
                chosen_log_probs = tf.gather(
                    log_all, batch_actions, axis=1, batch_dims=1)
                ratio = tf.exp(chosen_log_probs - batch_old_log_probs)
                unclipped = ratio * batch_adv
                clipped = tf.clip_by_value(
                    ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_adv
                policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))
                value_loss = 0.5 * tf.reduce_mean(
                    tf.square(tf.squeeze(values, axis=-1) - batch_returns))
                entropy = -tf.reduce_mean(
                    tf.reduce_sum(probs * log_all, axis=-1))
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            approx_kl = tf.reduce_mean(batch_old_log_probs - chosen_log_probs)
            clip_fraction = tf.reduce_mean(
                tf.cast(tf.abs(ratio - 1.0) > clip_ratio, tf.float32))
            metric_rows.append((float(policy_loss.numpy()), float(value_loss.numpy()),
                                float(entropy.numpy()), float(approx_kl.numpy()),
                                float(clip_fraction.numpy())))
            if target_kl > 0 and float(approx_kl.numpy()) > 1.5 * target_kl:
                stopped_early = True
                break
        if stopped_early:
            break

    metrics = np.asarray(metric_rows, dtype=np.float64)
    return {
        "policy_loss": float(metrics[:, 0].mean()),
        "value_loss": float(metrics[:, 1].mean()),
        "entropy": float(metrics[:, 2].mean()),
        "approx_kl": float(metrics[:, 3].mean()),
        "clip_fraction": float(metrics[:, 4].mean()),
    }


def make_validation_fleets(seed: int, count: int) -> list[dict]:
    """Flottes déterministes, sans altérer l'aléatoire de l'entraînement."""
    random_state = random.getstate()
    random.seed(seed)
    fleets = [random_grid() for _ in range(count)]
    random.setstate(random_state)
    return fleets


def run_policy_on_fleet(fleet: dict, policy: str, model: keras.Model | None,
                        seed: int) -> tuple[int, float]:
    """Évalue une policy sur une flotte fixe : tirs et retour réel."""
    env = ShapedEnv()  #BattleshipEnv()
    env.grid = fleet
    env.shots = {}
    env.turns = 0
    rng = random.Random(seed)
    done = False
    episode_return = 0.0
    while not done:
        if policy == "ppo":
            state = normalized_observation(env.state())
            action = greedy_action(model, state)
            x, y = action % GRID, action // GRID
        elif policy == "heuristic":
            # heuristic_shot choisit au hasard en phase hunt ; isoler cette source
            old_state = random.getstate()
            random.seed(rng.randrange(2**32))
            x, y = heuristic_shot(env.shots)
            random.setstate(old_state)
        else:
            options = env.valid_actions()
            cell = rng.choice(options)
            x, y = cell % GRID, cell // GRID
        _, reward, done, _ = env.step(y * GRID + x)
        episode_return += reward
    return env.turns, episode_return


def evaluate(model: keras.Model, fleets: list[dict], seed: int,
             update: int, include_baselines: bool = True) -> dict[str, float]:
    """Évaluation appariée PPO/random/heuristic sur les mêmes flottes."""
    ppo_results = [run_policy_on_fleet(f, "ppo", model, seed + i)
                   for i, f in enumerate(fleets)]
    ppo = np.asarray([result[0] for result in ppo_results], dtype=np.float64)
    returns = np.asarray([result[1] for result in ppo_results], dtype=np.float64)
    random_mean = float("nan")
    heuristic_mean = float("nan")
    if include_baselines:
        random_scores = np.asarray([
            run_policy_on_fleet(f, "random", None, seed + 10_000 + i)[0]
            for i, f in enumerate(fleets)], dtype=np.float64)
        heuristic_scores = np.asarray([
            run_policy_on_fleet(f, "heuristic", None, seed + 20_000 + i)[0]
            for i, f in enumerate(fleets)], dtype=np.float64)
        random_mean = float(random_scores.mean())
        heuristic_mean = float(heuristic_scores.mean())
    mean = float(ppo.mean())
    std = float(ppo.std(ddof=1)) if len(ppo) > 1 else 0.0
    se = std / math.sqrt(len(ppo)) if len(ppo) else 0.0
    return {
        "validation_shots_mean": mean,
        "validation_shots_std": std,
        "validation_shots_variance": float(ppo.var(ddof=1)) if len(ppo) > 1 else 0.0,
        "validation_shots_ci95_low": mean - 1.96 * se,
        "validation_shots_ci95_high": mean + 1.96 * se,
        "validation_return_mean": float(returns.mean()),
        "random_shots_mean": random_mean,
        "heuristic_shots_mean": heuristic_mean,
    }


def explained_variance(returns: np.ndarray, values: np.ndarray) -> float:
    variance = float(np.var(returns))
    if variance < 1e-8:
        return float("nan")
    return float(1.0 - np.var(returns - values) / variance)


def upload_model(path: Path, key: str) -> None:
    import boto3
    import config

    client = boto3.client("s3", endpoint_url=config.S3_ENDPOINT)
    client.upload_file(str(path), config.S3_BUCKET, key)
    print(f"✅ Modèle → s3://{config.S3_BUCKET}/{key}")


def train(args: argparse.Namespace, run_dir: Path) -> Path:
    set_global_seed(args.seed)
    configure_gpu()
    ShapedEnv.rewards = {"miss": args.r_miss, "hit": args.r_hit,
                         "sunk": args.r_sunk, "win": args.r_win}
    print(f"Récompenses PPO : {ShapedEnv.rewards}")

    run_dir.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(run_dir / "tensorboard"))
    metrics_path = run_dir / "metrics.csv"
    model_path = MODEL_DIR / MODEL_NAME
    best_path = MODEL_DIR / "ppo_shooter_best.keras"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {resume_path}")
        model = keras.models.load_model(resume_path)
        print(f"♻️  Reprise depuis {resume_path} (poids + état de l'optimiseur)")
    else:
        model = build_actor_critic(args.learning_rate)
    env = ShapedEnv()
    state = normalized_observation(env.reset())
    training_fleets = make_validation_fleets(args.seed + 500_000,
                                              args.train_fleet_count)
    env.grid = training_fleets[0]
    state = normalized_observation(env.state())
    validation_fleets = make_validation_fleets(args.seed + 1_000_000,
                                                args.eval_episodes)
    train_eval_fleets = training_fleets[:min(args.train_eval_episodes,
                                              len(training_fleets))]
    fleet_cursor = 1
    best_validation = float("inf")
    if args.resume:
        # Seuil à battre : le checkpoint repris, pas l'infini (sinon il serait écrasé)
        baseline = evaluate(model, validation_fleets, args.seed, 0)
        best_validation = baseline["validation_shots_mean"]
        print(f"Point de départ validation : {best_validation:.2f} tirs "
              f"(random={baseline['random_shots_mean']:.2f}, "
              f"heuristic={baseline['heuristic_shots_mean']:.2f})")
    recent_returns: list[float] = []
    recent_shots: list[int] = []
    timesteps = 0
    update = 0
    no_improvement = 0

    print(f"PPO démarré : {args.timesteps:,} pas, rollout={args.rollout_steps}, "
          f"batch={args.minibatch_size}, epochs={args.update_epochs}")
    print(f"Validation fixe : {len(validation_fleets)} flottes ; logs : {run_dir}")
    print("PPO n'utilise pas les transitions historiques S3 : il collecte des "
          "trajectoires on-policy dans BattleshipEnv.")

    with open(metrics_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()
        csv_file.flush()

        while timesteps < args.timesteps:
            started = time.time()
            horizon = min(args.rollout_steps, args.timesteps - timesteps)
            rollout, episode_returns, episode_shots, bootstrap, state, fleet_cursor = collect_rollout(
                model, env, state, horizon, training_fleets, fleet_cursor)
            timesteps += horizon
            if episode_returns.size:
                recent_returns.extend(episode_returns.tolist())
                recent_shots.extend(episode_shots.tolist())
                recent_returns = recent_returns[-100:]
                recent_shots = recent_shots[-100:]

            advantages, returns = compute_gae(
                rollout["rewards"], rollout["dones"], rollout["values"],
                bootstrap, args.gamma, args.gae_lambda)
            update_metrics = ppo_update(
                model, rollout, advantages, returns,
                args.minibatch_size, args.update_epochs, args.clip_ratio,
                args.value_coef, args.entropy_coef, args.max_grad_norm,
                args.target_kl)
            update += 1
            train_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            train_shots = float(np.mean(recent_shots)) if recent_shots else float("nan")
            val_metrics = {}
            train_eval_shots = float("nan")
            gap = float("nan")

            if update % args.eval_every == 0 or timesteps >= args.timesteps:
                train_eval = evaluate(model, train_eval_fleets,
                                      args.seed + 3_000_000, update,
                                      include_baselines=False)
                train_eval_shots = train_eval["validation_shots_mean"]
                val_metrics = evaluate(model, validation_fleets, args.seed, update)
                # Critère de sélection principal : espérance des tirs (plus bas = mieux).
                if val_metrics["validation_shots_mean"] < best_validation:
                    best_validation = val_metrics["validation_shots_mean"]
                    no_improvement = 0
                    model.save(best_path)
                    print(f"✅ Nouveau meilleur checkpoint : {best_path}")
                else:
                    no_improvement += 1

                # Même policy greedy, jeux fixes distincts : proxy de généralisation,
                # pas une décomposition statistique exacte biais/variance.
                gap = (val_metrics["validation_shots_mean"] - train_eval_shots)
            row = {
                "update": update,
                "timesteps": timesteps,
                "train_return_mean": train_return,
                "train_shots_mean": train_shots,
                **update_metrics,
                "explained_variance": explained_variance(returns, rollout["values"]),
                 "train_eval_shots_mean": train_eval_shots,
                **{field: val_metrics.get(field, float("nan")) for field in CSV_FIELDS
                   if field.startswith("validation_") or field.endswith("_shots_mean")
                   and field in ("random_shots_mean", "heuristic_shots_mean")},
                "validation_gap_shots": gap,
                "random_shots_mean": val_metrics.get("random_shots_mean", float("nan")),
                "heuristic_shots_mean": val_metrics.get("heuristic_shots_mean", float("nan")),
            }
            # Ensure every declared CSV field has a value.
            row = {field: row.get(field, float("nan")) for field in CSV_FIELDS}
            csv_writer.writerow(row)
            csv_file.flush()

            with writer.as_default(step=timesteps):
                for key, value in update_metrics.items():
                    tf.summary.scalar(f"train/{key}", value)
                tf.summary.scalar("train/return_mean_100", train_return)
                tf.summary.scalar("train/shots_mean_100", train_shots)
                tf.summary.scalar("train/explained_variance", row["explained_variance"])
                tf.summary.scalar("train_eval/shots_mean", train_eval_shots)
                for key, value in val_metrics.items():
                    tf.summary.scalar(f"validation/{key}", value)
                tf.summary.scalar("validation/train_gap_shots", gap)
            writer.flush()

            elapsed = time.time() - started
            val_text = (f"train-eval={train_eval_shots:.2f}; "
                        f"val shots={val_metrics['validation_shots_mean']:.2f} "
                        f"±{val_metrics['validation_shots_std']:.2f} "
                        f"(IC95 {val_metrics['validation_shots_ci95_low']:.2f}, "
                        f"{val_metrics['validation_shots_ci95_high']:.2f}); "
                        f"random={val_metrics['random_shots_mean']:.2f}, "
                        f"heuristic={val_metrics['heuristic_shots_mean']:.2f}"
                        if val_metrics else "validation en attente")
            print(f"update {update:>4} | steps {timesteps:>8,}/{args.timesteps:,} | "
                  f"retour train={train_return:.2f} | tirs train={train_shots:.2f} | "
                  f"policy={update_metrics['policy_loss']:.4f} "
                  f"value={update_metrics['value_loss']:.4f} "
                  f"entropy={update_metrics['entropy']:.3f} "
                  f"KL={update_metrics['approx_kl']:.4f} "
                  f"EV={row['explained_variance']:.3f} | {val_text} | "
                  f"{elapsed:.1f}s")

            if args.patience > 0 and no_improvement >= args.patience:
                print(f"Arrêt anticipé : pas d'amélioration validation après "
                      f"{args.patience} évaluations.")
                break

    writer.close()
    model.save(model_path)
    print(f"✅ Dernier modèle : {model_path}")
    print(f"✅ Meilleur modèle validation : {best_path}")
    print(f"📈 CSV : {metrics_path}")
    print(f"📊 TensorBoard : tensorboard --logdir {args.log_dir}")
    return best_path if best_path.exists() else model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement PPO bataille navale")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="nombre total de tirs collectés (défaut: 500000)")
    parser.add_argument("--rollout-steps", type=int, default=2048,
                        help="pas collectés avant chaque mise à jour PPO")
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--r-miss", type=float, default=-0.1)
    parser.add_argument("--r-hit", type=float, default=0.0)
    parser.add_argument("--r-sunk", type=float, default=0.0)
    parser.add_argument("--r-win", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="arrêt anticipé des epochs si KL dépasse 1.5× cette cible; 0 désactive")
    parser.add_argument("--eval-episodes", type=int, default=200,
                        help="nombre de flottes fixes pour la validation")
    parser.add_argument("--train-fleet-count", type=int, default=10_000,
                        help="taille du pool fixe de flottes d'entraînement")
    parser.add_argument("--train-eval-episodes", type=int, default=100,
                        help="flottes du pool train pour mesurer le gap avec validation")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="validation toutes les N mises à jour PPO")
    parser.add_argument("--patience", type=int, default=0,
                        help="arrêt après N validations sans amélioration (0 désactive)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", nargs="?", const=str(MODEL_DIR / "ppo_shooter_best.keras"),
                        default=None, metavar="CHEMIN",
                        help="reprendre depuis un .keras (sans valeur : ppo_shooter_best.keras)")
    parser.add_argument("--log-dir", default=str(LOG_ROOT))
    parser.add_argument("--upload", action="store_true",
                        help="uploader le meilleur checkpoint vers S3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = start_logging(run_dir)
    try:
        best_path = train(args, run_dir)
        if args.upload:
            upload_model(best_path, "bataille-navale/models/ppo_shooter.keras")
            upload_model(run_dir / "metrics.csv",
                         f"bataille-navale/logs/ppo/{run_dir.name}_metrics.csv")
            upload_model(run_dir / "training.txt",
                         f"bataille-navale/logs/ppo/{run_dir.name}_training.txt")
    finally:
        sys.stdout.flush()
        sys.stdout = sys.__stdout__
        log_file.close()


if __name__ == "__main__":
    main()
