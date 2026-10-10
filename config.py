import os

KAFKA_BOOTSTRAP = (
            os.getenv("KAFKA_BOOTSTRAP")            # si défini dans my-secrets un jour
            or f'{os.getenv("KAFKA_SERVICE_HOST")}:{os.getenv("KAFKA_SERVICE_PORT", "9092")}'
        )

# ---- PostgreSQL (auto-injecté par Onyxia) ----
PG_HOST = os.getenv("PGHOST", "localhost")
PG_PORT = int(os.getenv("PGPORT", "5432"))
PG_DB = os.getenv("PGDATABASE", "defaultdb")
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASSWORD = os.getenv("PGPASSWORD", "")

# ---- S3 / MinIO (auto-injecté, session OIDC) ----
S3_ENDPOINT = os.getenv("AWS_ENDPOINT_URL",
                        os.getenv("AWS_S3_ENDPOINT", "https://minio.lab.sspcloud.fr"))
# Enlève le protocole si la var n'en contient pas
if not S3_ENDPOINT.startswith("http"):
    S3_ENDPOINT = "https://" + S3_ENDPOINT

S3_BUCKET = os.getenv("USER", "buckela")   # ton bucket personnel = ton nom d'utilisateur