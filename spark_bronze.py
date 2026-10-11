"""spark_bronze.py — BRONZE : streaming Kafka → Parquet brut sur S3/MinIO.

Longue durée :  python spark_bronze.py     (ou spark-submit spark_bronze.py)

- Un stream par topic (game_moves, game_events)
- Zéro logique métier : parsing JSON → typage + métadonnées Kafka conservées
  (offset/timestamp : nécessaires à la dédup en silver) + message brut (`raw`)
- Checkpoint S3 : reprise exactly-once après crash/redémarrage
- Partitionné par date — JAMAIS par game_id (cardinalité trop élevée)

⚠️ Token STS Onyxia : expire après quelques heures → relancer le job
   si le stream meurt avec une erreur 403.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, current_date, from_json
from pyspark.sql.streaming import StreamingQuery
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

import config
from spark_utils import checkpoint, get_spark, zone

# Champs absents du JSON → null ; champs inconnus → ignorés (tolérant).
# La colonne `raw` conserve le message brut en secours (sécurité bronze).
MOVE_SCHEMA = StructType([
    StructField("game_id", StringType()),
    StructField("player", StringType()),
    StructField("turn", IntegerType()),
    StructField("x", IntegerType()),
    StructField("y", IntegerType()),
    StructField("result", StringType()),
    StructField("mode", StringType()),      # ai | pvp
    StructField("policy", StringType()),    # human | heuristic_v1 | tensor_v2...
    StructField("opponent", StringType()),
    StructField("ts", StringType()),        # event time ISO — parsé en silver
])

EVENT_SCHEMA = StructType([
    StructField("event", StringType()),     # login|game_start|match_found|game_over
    StructField("game_id", StringType()),
    StructField("player", StringType()),
    StructField("players", ArrayType(StringType())),
    StructField("winner", StringType()),
    StructField("mode", StringType()),
    StructField("ts", StringType()),
])

SHIP_SCHEMA = StructType([
    StructField("x", IntegerType()),
    StructField("y", IntegerType()),
    StructField("ship", StringType()),
])

BOARD_SCHEMA = StructType([
    StructField("game_id", StringType()),
    StructField("player", StringType()),
    StructField("ships", ArrayType(SHIP_SCHEMA)),   # 17 cases
    StructField("mode", StringType()),
    StructField("policy", StringType()),
    StructField("ts", StringType()),
])


def stream_topic(spark: SparkSession, topic: str, schema: StructType,
                 table: str) -> StreamingQuery:
    """Un topic Kafka → une table bronze (append-only, partitionnée par date)."""
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP)
        .option("subscribe", topic)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")   # la rétention Kafka ne doit pas tuer le stream
        .load()
    )
    parsed = (
        raw.select(
            from_json(col("value").cast("string"), schema).alias("d"),
            col("value").cast("string").alias("raw"),
            col("offset").alias("kafka_offset"),
            col("timestamp").alias("kafka_ts"),
        )
        .select("d.*", "raw", "kafka_offset", "kafka_ts")
        .withColumn("date", current_date())
    )
    return (
        parsed.writeStream.format("parquet")
        .option("path", zone("bronze", table))
        .option("checkpointLocation", checkpoint(f"bronze_{table}"))
        .partitionBy("date")
        .trigger(processingTime="1 minute")
        .start()
    )


def main() -> None:
    spark = get_spark("bataille-bronze")
    queries = [
        stream_topic(spark, "game_moves", MOVE_SCHEMA, "game_moves"),
        stream_topic(spark, "game_events", EVENT_SCHEMA, "game_events"),
        stream_topic(spark, "game_boards", BOARD_SCHEMA, "game_boards"),
    ]
    for q in queries:
        print(f"✅ stream '{q.name}' actif")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
