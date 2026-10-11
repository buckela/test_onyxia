"""reset_data.py — Remise à zéro complète des données du projet.

⚠️  À lancer APRÈS avoir arrêté spark_bronze.py (sinon il réécrit pendant
    la suppression). Supprime :
    1. tout le préfixe S3 bataille-navale/ (bronze, silver, gold, checkpoints)
    2. les topics Kafka (game_moves, game_events, game_chat) puis les recrée
       vides — sans ça, le redémarrage de bronze rejouerait tout l'historique.

    python reset_data.py
"""
import time

import boto3
from kafka.admin import KafkaAdminClient
from kafka.errors import KafkaError, UnknownTopicOrPartitionError

import config
import kafka_utils

S3_PREFIX = "bataille-navale/"


def reset_s3() -> int:
    """Supprime tous les objets sous s3://<bucket>/bataille-navale/."""
    s3 = boto3.client("s3", endpoint_url=config.S3_ENDPOINT)
    bucket = config.S3_BUCKET
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=S3_PREFIX):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            deleted += len(keys)
    print(f"✅ S3 : {deleted} objets supprimés sous s3://{bucket}/{S3_PREFIX}")
    return deleted


def reset_kafka() -> None:
    """Supprime puis recrée les topics (purge instantanée de l'historique)."""
    admin = KafkaAdminClient(bootstrap_servers=config.KAFKA_BOOTSTRAP)
    try:
        admin.delete_topics(kafka_utils.ALL_TOPICS)
        print(f"✅ Kafka : topics supprimés {kafka_utils.ALL_TOPICS}")
        time.sleep(3)   # la suppression est asynchrone côté broker
    except UnknownTopicOrPartitionError:
        print("ℹ️  Kafka : topics déjà absents")
    except KafkaError as e:
        print(f"⚠️  Kafka delete_topics : {e}")
    finally:
        admin.close()
    kafka_utils.ensure_topics()   # recrée les topics vides
    print("✅ Kafka : topics recréés (vides)")


if __name__ == "__main__":
    print("── Reset des données ──")
    print("⚠️  spark_bronze.py doit être arrêté. Ctrl+C pour annuler (5 s)...")
    time.sleep(5)
    reset_s3()
    reset_kafka()
    print("── Terminé : tu peux relancer spark_bronze.py sur des bases saines ──")
