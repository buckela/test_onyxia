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
- **TensorFlow** (service vscode-tensorflow-gpu) :
  - [training_IA.py](training_IA.py) : DQN (tirs, offline gold + self-play) + régression (placements)
  - [use_IA.py](use_IA.py) : inférence — policy `tensor_v1`, évaluation, suggestion de placement

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
```bash
python simulate_games.py --games 1000 --mix --mirror # corpus équilibré (recommandé)
```

Parties IA vs IA (`random_v1` vs `heuristic_v1` vs `tensor_v1`) avec la vraie
logique du jeu, envoyées à Kafka avec `mode="simulation"` (séparées des vraies
parties dans silver/gold). Le bronze streaming doit tourner pendant la simulation.

## Machine Learning (service vscode-tensorflow-gpu)

```bash
python training_IA.py --episodes 3000 --upload   # DQN tirs + régression placements
python use_IA.py --download --eval 100           # récupère les modèles, évalue
python use_IA.py --placement                     # suggère un placement optimal
```

Les modèles (.keras) sont sauvegardés dans `models/` et poussés sur S3
(`bataille-navale/models/`). `tensor_v1` devient ensuite une policy jouable
dans le simulateur : `python simulate_games.py --policy-a tensor_v1 --policy-b heuristic_v1`.