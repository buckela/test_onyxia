"""spark_gold.py — GOLD : dataset ML (state → action → outcome) pour TensorFlow.

Batch (à la demande) :  python spark_gold.py     (ou spark-submit)

Pour chaque tir de chaque partie TERMINÉE, reconstruit la connaissance
du tireur AVANT le tir (replay de ses tirs précédents) :

    features[100] : 0=inconnu, 1=raté, 2=touché/coulé
                    (même encodage que grid_to_features() côté jeu)
    action_cell   : y * 10 + x            (la case jouée)
    label_hit     : 1 si hit/sunk         (récompense immédiate)
    label_win     : 1 si le tireur gagne  (récompense finale)

Le champ `policy` permet de comparer les générations de données
(heuristic_v1 vs futur modèle TensorFlow).

Sorties :
- gold/ml_dataset  : où TIRER   (partitionné par mode)
- gold/placements  : où PLACER  (1 ligne par joueur/partie terminée)

Nécessite pandas + pyarrow sur le service Spark (présents sur Onyxia).
"""
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

from spark_utils import get_spark, zone

GRID = 10
CODE = {"miss": 1, "hit": 2, "sunk": 2}

GOLD_SCHEMA = StructType([
    StructField("game_id", StringType()),
    StructField("player", StringType()),
    StructField("mode", StringType()),
    StructField("policy", StringType()),
    StructField("turn", IntegerType()),
    StructField("action_x", IntegerType()),
    StructField("action_y", IntegerType()),
    StructField("action_cell", IntegerType()),
    StructField("result", StringType()),
    StructField("label_hit", IntegerType()),
    StructField("label_win", IntegerType()),
    StructField("features", ArrayType(IntegerType())),
])


def build_rows(pdf: pd.DataFrame) -> pd.DataFrame:
    """Replay des tirs d'UN joueur d'UNE partie (ordre = turn)."""
    pdf = pdf.sort_values("turn")
    state: dict[tuple[int, int], int] = {}
    rows = []
    for r in pdf.itertuples():
        # Connaissance du tireur AVANT ce tir
        features = [state.get((x, y), 0) for y in range(GRID) for x in range(GRID)]
        rows.append({
            "game_id": r.game_id, "player": r.player, "mode": r.mode,
            "policy": r.policy, "turn": int(r.turn),
            "action_x": int(r.x), "action_y": int(r.y),
            "action_cell": int(r.y) * GRID + int(r.x),
            "result": r.result,
            "label_hit": 1 if r.result in ("hit", "sunk") else 0,
            "label_win": int(r.win),
            "features": features,
        })
        state[(int(r.x), int(r.y))] = CODE[r.result]
    return pd.DataFrame(rows)


# ---------- Dataset placement : où POSITIONNER sa flotte ----------

PLACEMENT_UDF_SCHEMA = StructType([
    StructField("placement", ArrayType(IntegerType())),
    StructField("adjacent_pairs", IntegerType()),
    StructField("edge_cells", IntegerType()),
])


@udf(PLACEMENT_UDF_SCHEMA)
def placement_features(ships):
    """Vecteur de placement + features tactiques simples :
    - placement[100] : 1 = case occupée par un bateau
    - adjacent_pairs : paires de cases bateau voisines (coller ses bateaux
      est risqué : un touché révèle les cases adjacentes)
    - edge_cells     : cases bateau sur les bords de la grille
    """
    cells = {(s["x"], s["y"]) for s in ships}
    vec = [0] * 100
    for (x, y) in cells:
        vec[y * 10 + x] = 1
    adj = sum(1 for (x, y) in cells for dx, dy in ((1, 0), (0, 1))
              if (x + dx, y + dy) in cells)
    edge = sum(1 for (x, y) in cells if x in (0, 9) or y in (0, 9))
    return (vec, adj, edge)


def build_placements(spark):
    """1 ligne par joueur/partie terminée : placement + dégâts subis + issue."""
    boards = spark.read.parquet(zone("silver", "game_boards"))
    moves = spark.read.parquet(zone("silver", "game_moves"))
    outcomes = spark.read.parquet(zone("silver", "game_outcomes"))

    # Dégâts subis : tirs ADVERSES tombés sur mes cases bateau
    cells = (
        boards.select("game_id", "player", explode("ships").alias("s"))
        .select("game_id", "player", col("s.x").alias("x"), col("s.y").alias("y"))
    )
    c, m = cells.alias("c"), moves.alias("m")
    damage = (
        c.join(m, (col("c.game_id") == col("m.game_id"))
               & (col("c.x") == col("m.x")) & (col("c.y") == col("m.y"))
               & (col("c.player") != col("m.player")))
        .groupBy(col("c.game_id").alias("game_id"), col("c.player").alias("player"))
        .agg(
            F.count("*").alias("hits_taken"),
            F.min("turn").alias("first_hit_turn"),   # quand la flotte est repérée
            F.sum(F.when(col("result") == "sunk", 1).otherwise(0)).alias("ships_lost"),
        )
    )

    finished = boards.join(outcomes, "game_id")   # parties terminées uniquement
    return (
        finished.join(damage, ["game_id", "player"], "left")
        .fillna({"hits_taken": 0, "ships_lost": 0})
        .withColumn("pf", placement_features("ships"))
        .select(
            "game_id", "player", "mode", "policy",
            col("pf.placement").alias("placement"),
            col("pf.adjacent_pairs").alias("adjacent_pairs"),
            col("pf.edge_cells").alias("edge_cells"),
            "hits_taken", "ships_lost", "first_hit_turn",
            (col("player") == col("winner")).cast("int").alias("label_win"),
        )
    )


def main() -> None:
    spark = get_spark("bataille-gold")
    spark.sparkContext.setLogLevel("WARN")

    # Dataset 1 : où TIRER (state → action → outcome)
    moves = (
        spark.read.parquet(zone("silver", "game_moves"))
        .filter(col("winner").isNotNull())   # parties terminées uniquement
    )

    dataset = (
        moves.groupBy("game_id", "player")
        .applyInPandas(build_rows, schema=GOLD_SCHEMA)
    )

    (dataset.write.mode("overwrite").partitionBy("mode")
     .parquet(zone("gold", "ml_dataset")))

    # Dataset 2 : où PLACER sa flotte (placement → dégâts → victoire)
    placements = build_placements(spark)
    (placements.write.mode("overwrite").partitionBy("mode")
     .parquet(zone("gold", "placements")))

    print(f"✅ gold : {dataset.count()} exemples de tirs, "
          f"{placements.count()} placements")


if __name__ == "__main__":
    main()
