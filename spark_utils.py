"""spark_utils.py — Streaming Kafka → Parquet (zone bronze) sur S3/MinIO.

À exécuter dans le service Spark Onyxia (PAS dans l'app Streamlit) :
    python spark_utils.py        # ou spark-submit spark_utils.py

Les identifiants AWS/MinIO (dont le token STS) sont injectés par Onyxia.
"""
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_date, from_json
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

import config

# Les champs absents du JSON sont ignorés par from_json → tolérant aux extras
MOVE_SCHEMA = StructType([
    StructField("game_id", StringType()), StructField("player", StringType()),
    StructField("turn", IntegerType()), StructField("x", IntegerType()),
    StructField("y", IntegerType()), StructField("result", StringType()),
    StructField("mode", StringType()),
])


def get_spark() -> SparkSession:
    # s3a attend un endpoint SANS schéma ; le SSL est piloté à part
    endpoint = config.S3_ENDPOINT.replace("https://", "").replace("http://", "")
    return (
        SparkSession.builder.appName("bataille-navale-bronze")
        # MinIO : nécessite hadoop-aws + configuration s3a
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config("spark.hadoop.fs.s3a.endpoint", endpoint)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled",
                "true" if config.S3_ENDPOINT.startswith("https") else "false")
        # Identifiants injectés par Onyxia (token STS temporaire)
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID", ""))
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY", ""))
        .config("spark.hadoop.fs.s3a.session.token", os.getenv("AWS_SESSION_TOKEN", ""))
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider")
        .getOrCreate()
    )


def stream_moves_to_bronze(spark: SparkSession) -> None:
    """Topic game_moves → Parquet bronze partitionné par date."""
    base = f"s3a://{config.S3_BUCKET}/bataille-navale"
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP)
        .option("subscribe", "game_moves")
        .option("startingOffsets", "earliest")
        .load()
    )
    moves = (
        raw.select(from_json(col("value").cast("string"), MOVE_SCHEMA).alias("d"))
        .select("d.*")
        .withColumn("date", current_date())
    )
    (
        moves.writeStream.format("parquet")
        .option("path", f"{base}/bronze/game_moves")
        .option("checkpointLocation", f"{base}/checkpoints/game_moves")
        .partitionBy("date")
        .trigger(processingTime="1 minute")
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    stream_moves_to_bronze(get_spark())