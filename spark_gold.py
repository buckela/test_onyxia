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

Sortie : gold/ml_dataset (partitionné par mode : ai / pvp)
Nécessite pandas + pyarrow sur le service Spark (présents sur Onyxia).
"""
import pandas as pd
from pyspark.sql.functions import col
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


def main() -> None:
    spark = get_spark("bataille-gold")
    spark.sparkContext.setLogLevel("WARN")

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

    print(f"✅ gold : {dataset.count()} exemples (state, action, label)")


if __name__ == "__main__":
    main()
