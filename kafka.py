import json, atexit
from kafka import KafkaProducer
from config import KAFKA_BOOTSTRAP, TOPIC_MOVES, TOPIC_EVENTS

_producer = None

def get_producer():
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        atexit.register(lambda: _producer.close() if _producer else None)
    return _producer

def send_move(game_id, player, turn, x, y, result):
    """À appeler à CHAQUE tir — c'est ça qui alimente le futur dataset IA."""
    get_producer().send(TOPIC_MOVES, {
        "game_id": game_id, "player": player, "turn": turn,
        "x": x, "y": y, "result": result,
    })

def send_event(event_type: str, **data):
    """Login, matchmaking, fin de partie..."""
    get_producer().send(TOPIC_EVENTS, {"type": event_type, **data})