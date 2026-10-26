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

```bash
python pipeline_batch.py                  # une passe silver → gold (avant entraînement)
python pipeline_batch.py --interval 60    # boucle toutes les heures
```

# TODO

- ajout bouton abandon

- qlik traitement statistique -> conversion spark

- calcul biais, variance , esperrance, varaince du bruit

- phase validation, hyperparamètre

- ppo







Le script inclut :

masque des cases déjà tirées, GAE, clipping PPO, limite de gradient et arrêt anticipé sur la divergence KL ;
validation déterministe sur des flottes fixes jamais vues pendant l’entraînement, comparée à random et heuristic ;
moyenne, variance, écart-type et IC 95 % du nombre de tirs ;
retour, pertes acteur/critique, entropie, KL, clip fraction et explained variance ;
logs texte et CSV, plus TensorBoard ; --upload envoie le meilleur modèle et les logs sur S3.
Pour suivre le gap train/validation, l’entraînement utilise un pool fixe de flottes, distinct du pool de validation. Ce gap est un indicateur d’overfitting, pas une décomposition exacte du biais et de la variance du modèle. La variance affichée mesure la dispersion des tirs selon les flottes testées.

Commence par un petit essai sur le service GPU :
python training_IA_ppo.py --timesteps 20000 --eval-episodes 50

Puis, si tout tourne, augmente le budget :
python training_IA_ppo.py --timesteps 500000 --eval-episodes 200 --upload

Les métriques s’affichent dans le terminal et sont conservées sous logs/ppo/<run>/. TensorBoard se lance avec :
tensorboard --logdir logs/ppo

La compilation Python passe. Je n’ai pas pu lancer un entraînement réel ici, car TensorFlow n’est pas installé dans l’environnement local. Le modèle PPO est enregistré séparément sous models/ppo_shooter.keras : il n’est pas directement interchangeable avec le modèle DQN attendu par use_IA.py.