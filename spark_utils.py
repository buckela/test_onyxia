from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_date
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

MOVE_SCHEMA = StructType([
    StructField("game_id", StringType()), StructField("player", StringType()),
    StructField("turn", IntegerType()), StructField("x", IntegerType()),
    StructField("y", IntegerType()), StructField("result", StringType()),
])

def get_spark():
    return (SparkSession.builder.appName("bataille-navale")
            # MinIO : nécessite hadoop-aws + configure s3a
            .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2")
            .config("spark.hadoop.fs.s3a.endpoint", S3_ENDPOINT)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .getOrCreate())

def stream_moves_to_bronze(spark):
    """Streaming Kafka → Parquet bronze (à lancer dans un service Spark à part)."""
    df = (spark.readStream.format("kafka")
          .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
          .option("subscribe", "game_moves").load())
    parsed = df.select(from_json(col("value").cast("string"), MOVE_SCHEMA).alias("d")).select("d.*")
    (parsed.writeStream.format("parquet")
     .option("path", f"s3a://bataille-navale/bronze/game_moves")
     .option("checkpointLocation", f"s3a://bataille-navale/checkpoints/game_moves")
     .partitionBy("game_id")
     .trigger(processingTime="1 minute").start().awaitTermination())

if __name__ == "__main__":
    stream_moves_to_bronze(get_spark())