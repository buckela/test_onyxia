"""spark_silver.py — SILVER : nettoyage + dédup + jointure avec les outcomes.

Batch (à la demande) :  python spark_silver.py     (ou spark-submit)

- Dédup des tirs (retries Kafka) : 1re occurrence par (game_id, player, x, y)
- Validation métier : coordonnées 0-9, résultat connu, clés non nulles
- `ts` (event time ISO) parsé en timestamp
- Jointure avec game_over → winner + flag `win` sur chaque tir
- Rebuild complet (idempotent) : à cette échelle, plus simple et plus fiable
  qu'un merge incrémental (passer à Delta Lake si le volume explose)
"""
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, row_number, to_timestamp, when
from pyspark.sql.window import Window

from spark_utils import get_spark, zone

VALID_RESULTS = ("miss", "hit", "sunk")


def clean_moves(moves: DataFrame) -> DataFrame:
    """Dédup (garde la 1re occurrence Kafka) + validation métier."""
    w = Window.partitionBy("game_id", "player", "x", "y") \
              .orderBy("kafka_ts", "kafka_offset")
    return (
        moves
        .withColumn("rn", row_number().over(w))
        .filter(col("rn") == 1).drop("rn")
        .filter(col("x").between(0, 9) & col("y").between(0, 9))
        .filter(col("result").isin(*VALID_RESULTS))
        .filter(col("game_id").isNotNull() & col("player").isNotNull())
        .withColumn("event_ts", to_timestamp("ts"))
        .drop("raw")
    )


def game_outcomes(events: DataFrame) -> DataFrame:
    """Un game_over par partie (le premier, si doublon)."""
    w = Window.partitionBy("game_id").orderBy("kafka_ts", "kafka_offset")
    return (
        events
        .filter((col("event") == "game_over") & col("game_id").isNotNull())
        .withColumn("rn", row_number().over(w))
        .filter(col("rn") == 1)
        .select(col("game_id"), col("winner"),
                to_timestamp("ts").alias("finished_ts"))
    )


def main() -> None:
    spark = get_spark("bataille-silver")
    spark.sparkContext.setLogLevel("WARN")

    moves = spark.read.parquet(zone("bronze", "game_moves"))
    events = spark.read.parquet(zone("bronze", "game_events"))

    outcomes = game_outcomes(events)
    enriched = (
        clean_moves(moves)
        .join(outcomes, "game_id", "left")
        .withColumn(
            "win",
            when(col("winner").isNull(), None)
            .otherwise((col("player") == col("winner")).cast("int")),
        )
    )

    (enriched.write.mode("overwrite").partitionBy("date")
     .parquet(zone("silver", "game_moves")))
    outcomes.write.mode("overwrite").parquet(zone("silver", "game_outcomes"))

    print(f"✅ silver : {enriched.count()} tirs, {outcomes.count()} parties terminées")


if __name__ == "__main__":
    main()
