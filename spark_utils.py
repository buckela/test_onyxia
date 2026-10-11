"""spark_utils.py — Configuration Spark partagée par les 3 jobs du pipeline.

Pipeline médaillon (un seul service Spark suffit) :
    spark_bronze.py  : streaming Kafka → Parquet brut       (longue durée)
    spark_silver.py  : batch nettoyage + dédup + jointures  (à la demande)
    spark_gold.py    : batch dataset ML (features/labels)   (à la demande)

Les identifiants AWS/MinIO (token STS) sont injectés par Onyxia.
⚠️ Ils expirent après quelques heures : relancer le job bronze si le
   stream meurt avec une erreur 403.
"""
import os

import pyspark
from pyspark.sql import SparkSession

import config

BASE = f"s3a://{config.S3_BUCKET}/bataille-navale"

# La source Kafka n'est PAS incluse dans Spark : package externe obligatoire,
# dont la version doit matcher exactement Spark + Scala de l'image Onyxia.
SPARK_VERSION = pyspark.__version__
_SCALA = "2.13" if SPARK_VERSION.startswith("4.") else "2.12"
KAFKA_PACKAGE = f"org.apache.spark:spark-sql-kafka-0-10_{_SCALA}:{SPARK_VERSION}"
# hadoop-aws aligné sur le Hadoop embarqué par Spark
HADOOP_AWS_PACKAGE = ("org.apache.hadoop:hadoop-aws:3.4.1"
                      if SPARK_VERSION.startswith("4.")
                      else "org.apache.hadoop:hadoop-aws:3.3.4")


def get_spark(app_name: str) -> SparkSession:
    """Session Spark configurée pour Kafka + MinIO (s3a, credentials STS Onyxia)."""
    # s3a attend un endpoint SANS schéma ; le SSL est piloté à part
    endpoint = config.S3_ENDPOINT.replace("https://", "").replace("http://", "")
    return (
        SparkSession.builder.appName(app_name)
        # Kafka (source streaming) + hadoop-aws (accès S3A/MinIO)
        .config("spark.jars.packages", f"{KAFKA_PACKAGE},{HADOOP_AWS_PACKAGE}")
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


def zone(zone_name: str, table: str) -> str:
    """Chemin S3 d'une table : s3a://<bucket>/bataille-navale/<zone>/<table>."""
    return f"{BASE}/{zone_name}/{table}"


def checkpoint(name: str) -> str:
    """Checkpoint streaming (offsets Kafka → reprise exactly-once après crash)."""
    return f"{BASE}/checkpoints/{name}"