from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.master("local[*]")\
    .config("spark.driver.memory", "4g") \
    .appName("yfinance-stream") \
    .getOrCreate()

schema = 