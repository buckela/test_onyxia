# check_services.py
import os
import sys

def check_postgres():
    try:
        import psycopg2
        conn = psycopg2.connect(dbname=os.environ["PGDATABASE"])
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            v = cur.fetchone()[0]
        conn.close()
        return f"✅ Postgres ({os.environ['PGHOST']}:{os.environ['PGPORT']}) — {v.split(',')[0]}"
    except Exception as e:
        return f"❌ Postgres : {e}"

def check_s3():
    try:
        import boto3
        s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
        bucket = os.environ.get("USER", "buckela")
        key = "bataille-navale/check/test.txt"
        s3.put_object(Bucket=bucket, Key=key, Body=b"ping")
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        s3.delete_object(Bucket=bucket, Key=key)
        return f"✅ S3 (bucket '{bucket}') — écriture/lecture/suppression OK"
    except Exception as e:
        return f"❌ S3 : {e}"

def check_kafka(bootstrap: str | None = None):
    try:
        
        from kafka import KafkaProducer, KafkaConsumer
        from kafka.admin import KafkaAdminClient
        # ---- Kafka (auto-injecté par Onyxia) ----
        KAFKA_BOOTSTRAP = (
            os.getenv("KAFKA_BOOTSTRAP")            # si défini dans my-secrets un jour
            or f'{os.getenv("KAFKA_SERVICE_HOST")}:{os.getenv("KAFKA_SERVICE_PORT", "9092")}'
        )
        bootstrap = bootstrap or KAFKA_BOOTSTRAP
        if not bootstrap:
            return "⚠️ Kafka : KAFKA_BOOTSTRAP non défini — déploie le service Kafka et mets l'URL dans my-secrets"

        # 1. Connexion au broker + liste des topics
        admin = KafkaAdminClient(bootstrap_servers=bootstrap)
        topics = admin.list_topics()
        admin.close()

        # 2. Création du topic de test si absent
        if "check_test" not in topics:
            from kafka.admin import NewTopic
            from kafka.errors import TopicAlreadyExistsError
            try:
                admin2 = KafkaAdminClient(bootstrap_servers=bootstrap)
                admin2.create_topics([NewTopic(name="check_test", num_partitions=1, replication_factor=1)])
                admin2.close()
            except TopicAlreadyExistsError:
                pass

        # 3. Round-trip producer → consumer
        producer = KafkaProducer(bootstrap_servers=bootstrap)
        producer.send("check_test", b"ping-streamlit").get(timeout=10)  # .get() = attend l'ACK
        producer.flush()
        producer.close()

        consumer = KafkaConsumer(
            "check_test", bootstrap_servers=bootstrap,
            auto_offset_reset="earliest", consumer_timeout_ms=5000,
        )
        got = None
        for msg in consumer:
            got = msg.value
            break
        consumer.close()

        if got == b"ping-streamlit":
            return f"✅ Kafka ({bootstrap}) — topics: {len(topics)}, round-trip producer→consumer OK"
        return f"❌ Kafka ({bootstrap}) : broker joignable mais message non reçu"
    except ImportError:
        return "⚠️ Kafka : installe le client (pip install kafka-python)"
    except Exception as e:
        return f"❌ Kafka ({bootstrap or 'non défini'}) : {e}"

if __name__ == "__main__":
    print("── Vérification des services ──\n")
    print(check_postgres())
    print(check_s3())
    print(check_kafka())
    print("\n── Fin ──")
    sys.exit(0)