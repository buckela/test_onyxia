"""
kafka_utils.py — Couche Kafka pour la bataille navale.

Fonctionnalités :
- Producer singleton avec sérialisation JSON, tolérant aux pannes
- Envoi asynchrone avec callback de logging (n bloque jamais le jeu)
- Gestion auto des topics (création si absents)
- Consumer réutilisable (pour tests, debugging ou matchmaking)
- Mode dégradé : si Kafka est down, le jeu continue (juste un log)

⚠️ Ne pas nommer ce fichier "kafka.py" (conflit avec la librairie).
"""
import json
import logging
import threading
import atexit
from typing import Any, Callable

from kafka import KafkaConsumer, KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import KafkaError, NoBrokersAvailable, TopicAlreadyExistsError

import config

logger = logging.getLogger("bataille.kafka")

# ==================== TOPICS ====================

TOPIC_GAME_MOVES = "game_moves"      # chaque tir joué (dataset IA)
TOPIC_GAME_EVENTS = "game_events"    # login, matchmaking, fin de partie...
TOPIC_CHAT = "game_chat"             # optionnel : chat entre joueurs

ALL_TOPICS = [TOPIC_GAME_MOVES, TOPIC_GAME_EVENTS, TOPIC_CHAT]


# ==================== PRODUCER ====================

class _ProducerHolder:
    """Singleton thread-safe, recréé si le broker revient."""
    _lock = threading.Lock()
    _producer: KafkaProducer | None = None

    @classmethod
    def get(cls) -> KafkaProducer | None:
        if cls._producer is not None:
            return cls._producer
        with cls._lock:
            if cls._producer is None:
                try:
                    cls._producer = KafkaProducer(
                        bootstrap_servers=config.KAFKA_BOOTSTRAP,
                        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                        key_serializer=lambda k: k.encode("utf-8") if k else None,
                        # fiabilité
                        acks="all",
                        retries=3,
                        # un seul message en attente max par partition suffit ici
                        max_in_flight_requests_per_connection=1,
                        request_timeout_ms=10_000,
                        # le jeu doit rester réactif
                        linger_ms=5,
                    )
                    logger.info("Producer Kafka connecté (%s)", config.KAFKA_BOOTSTRAP)
                except (NoBrokersAvailable, KafkaError) as e:
                    logger.warning("Kafka indisponible : %s", e)
            return cls._producer

    @classmethod
    def reset(cls):
        with cls._lock:
            if cls._producer is not None:
                try:
                    cls._producer.close(timeout=2)
                except Exception:
                    pass
            cls._producer = None


@atexit.register
def _cleanup():
    _ProducerHolder.reset()


def _send(topic: str, value: dict, key: str | None = None) -> bool:
    """Envoi asynchrone non bloquant. Renvoie True si accepté par le producer."""
    producer = _ProducerHolder.get()
    if producer is None:
        return False
    try:
        future = producer.send(topic, value=value, key=key)

        def _on_error(err):
            logger.error("Échec envoi Kafka (topic=%s) : %s", topic, err)
            # Forcer une reconnexion au prochain envoi si le broker est mort
            if isinstance(err, KafkaError):
                _ProducerHolder.reset()

        future.add_errback(_on_error)
        return True
    except KafkaError as e:
        logger.error("Kafka send a levé : %s", e)
        _ProducerHolder.reset()
        return False


# ==================== API MÉTIER ====================

def send_move(game_id: str, player: str, turn: int,
              x: int, y: int, result: str,
              extra: dict | None = None) -> bool:
    """
    À appeler à CHAQUE tir. Alimente le dataset IA (via Spark → Parquet).

    result : 'miss' | 'hit' | 'sunk'
    extra  : métadonnées libres (mode de jeu, contre IA ou humain...)
    """
    payload = {
        "game_id": game_id,
        "player": player,
        "turn": turn,
        "x": x,
        "y": y,
        "result": result,
        "mode": (extra or {}).pop("mode", None),
        **(extra or {}),
    }
    return _send(TOPIC_GAME_MOVES, payload, key=game_id)


def send_event(event_type: str, **data) -> bool:
    """Événements de cycle de vie : login, game_start, game_over, match_found..."""
    return _send(TOPIC_GAME_EVENTS,
                 {"event": event_type, **data},
                 key=data.get("game_id"))


def send_chat(game_id: str, player: str, message: str) -> bool:
    return _send(TOPIC_CHAT, {"game_id": game_id, "player": player, "message": message},
                 key=game_id)


# ==================== ADMIN ====================

def ensure_topics():
    """Crée les topics du jeu s'ils n'existent pas. Idempotent."""
    try:
        admin = KafkaAdminClient(bootstrap_servers=config.KAFKA_BOOTSTRAP)
        existing = set(admin.list_topics())
        missing = [t for t in ALL_TOPICS if t not in existing]
        if missing:
            admin.create_topics([
                NewTopic(name=t, num_partitions=1, replication_factor=1)
                for t in missing
            ])
            logger.info("Topics créés : %s", missing)
        admin.close()
        return True
    except (TopicAlreadyExistsError, KafkaError) as e:
        logger.warning("ensure_topics : %s", e)
        return False


def list_topics() -> list[str]:
    try:
        admin = KafkaAdminClient(bootstrap_servers=config.KAFKA_BOOTSTRAP)
        topics = sorted(admin.list_topics())
        admin.close()
        return topics
    except KafkaError as e:
        logger.warning("list_topics : %s", e)
        return []


# ==================== CONSUMER ====================

def consume(topic: str, n: int = 10, timeout_ms: int = 3000) -> list[dict]:
    """
    Lit jusqu'à `n` messages (les plus récents disponibles).
    Usage : debug, tests, ou polling léger côté app.
    """
    out = []
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=config.KAFKA_BOOTSTRAP,
            auto_offset_reset="earliest",
            consumer_timeout_ms=timeout_ms,
            group_id=f"bataille-debug-{topic}",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        for msg in consumer:
            out.append(msg.value)
            if len(out) >= n:
                break
        consumer.close()
    except KafkaError as e:
        logger.warning("consume : %s", e)
    return out


def stream(topic: str, handler: Callable[[dict], None], group_id: str | None = None):
    """
    Boucle bloquante : appelle handler(payload) pour chaque message.
    À lancer dans un thread dédié. Ne retourne qu'en cas d'erreur fatale.
    """
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=config.KAFKA_BOOTSTRAP,
        group_id=group_id or f"bataille-{topic}",
        auto_offset_reset="latest",           # on démarre à neuf
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    logger.info("Consumer actif sur '%s' (group=%s)", topic, group_id)
    try:
        for msg in consumer:
            try:
                handler(msg.value)
            except Exception:
                logger.exception("Handler a échoué sur %s", msg.value)
    finally:
        consumer.close()


# ==================== SELF-TEST ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    print(f"Broker : {config.KAFKA_BOOTSTRAP}")
    ensure_topics()
    print("Topics :", list_topics())

    gid = "test-game-123"
    ok = send_move(gid, "tester", 1, 4, 7, "hit", extra={"mode": "ai"})
    print("send_move :", "OK" if ok else "ÉCHEC")
    send_event("game_over", game_id=gid, winner="tester")

    msgs = consume(TOPIC_GAME_MOVES, n=5)
    print("Messages reçus :", msgs)
    assert any(m.get("game_id") == gid for m in msgs), "round-trip KO"
    print("✅ kafka_utils opérationnel") 