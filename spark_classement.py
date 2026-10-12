"""

"""
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

from spark_utils import get_spark, zone






def main() -> None:
    spark = get_spark("bataille-silver")
    spark.sparkContext.setLogLevel("WARN")

    moves = spark.read.parquet(zone("bronze", "game_moves"))
    events = spark.read.parquet(zone("bronze", "game_events"))
    boards = spark.read.parquet(zone("bronze", "game_boards"))



if __name__ == "__main__":
    main()