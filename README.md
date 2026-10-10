# test_onyxia — Bataille navale sur Onyxia / SSPCloud

Pipeline data complet :

- **Streamlit** ([app.py](app.py)) : front du jeu (contre l'IA ou PvP)
- **PostgreSQL** ([bdd.py](bdd.py)) : joueurs, matchmaking (file atomique),
  plateaux et tirs — en PvP la BDD résout les tirs et fait foi
- **Kafka** ([kafka_utils.py](kafka_utils.py)) : events + chaque tir → dataset
- **Spark** ([spark_utils.py](spark_utils.py), service séparé) : topic `game_moves`
  → Parquet bronze partitionné par date sur S3/MinIO
- **TensorFlow** (plus tard) : entraînement d'une IA sur la zone bronze

## Lancer le jeu

```bash
streamlit run app.py
```

## Vérifier la connectivité des services

```bash
python test_services.py
```

## Lancer le job de streaming (dans le service Spark)

```bash
python spark_utils.py
```