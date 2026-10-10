import os

# PostgreSQL (ajuste selon ton service Onyxia)
PG_HOST = os.getenv("PG_HOST", "postgresql-bataille")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "bataille")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")

# Kafka
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka-bataille:9092")
TOPIC_MOVES = "game_moves"
TOPIC_EVENTS = "game_events"   # login, matchmaking, fin de partie

# S3 / MinIO
S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT", "")
S3_BUCKET = "bataille-navale"