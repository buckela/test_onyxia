# test_onyxia — Bataille navale sur Onyxia / SSPCloud

Pipeline data complet :

- **Streamlit** ([app.py](app.py)) : front du jeu (contre l'IA ou PvP)
- **PostgreSQL** ([bdd.py](bdd.py)) : joueurs, matchmaking (file atomique),
  plateaux et tirs — en PvP la BDD résout les tirs et fait foi
- **Kafka** ([kafka_utils.py](kafka_utils.py)) : events + chaque tir
  (avec `ts` event-time et `policy` : human / heuristic_v1 / futur modèle) → dataset
- **Spark** (service séparé) : pipeline médaillon sur S3/MinIO
  - [spark_bronze.py](spark_bronze.py) : streaming Kafka → Parquet brut (longue durée)
  - [spark_silver.py](spark_silver.py) : dédup + validation + jointure outcomes (batch)
  - [spark_gold.py](spark_gold.py) : dataset ML state→action→label (batch)
- **TensorFlow** (plus tard) : entraînement d'une IA sur la zone gold

## Lancer le jeu

```bash
streamlit run app.py
```
Connection : swijeji5xlodgdtqmmr7

## Vérifier la connectivité des services

```bash
python test_services.py
```

## Pipeline de données (dans le service Spark)

```bash
python spark_bronze.py    # streaming continu : Kafka → bronze
python spark_silver.py    # batch : bronze → silver (dédup, jointure outcomes)
python spark_gold.py      # batch : silver → gold (dataset ML)
```

## Générer des données d'entraînement (simulation)

```bash
python simulate_games.py --games 1000 --mirror
```

Parties IA vs IA (`random_v1` vs `heuristic_v1`) avec la vraie logique du jeu,
envoyées à Kafka avec `mode="simulation"` (séparées des vraies parties dans
silver/gold). Le bronze streaming doit tourner pendant la simulation.