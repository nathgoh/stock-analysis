from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

spark = SparkSession.builder.master("local[*]")\
    .config("spark.driver.memory", "4g") \
    .appName("yfinance-stream") \
    .getOrCreate()

schema = StructType([
    StructField("id", StringType(), False),       # ticker symbol
    StructField("price", DoubleType(), True),
    StructField("time", StringType(), True),       
    StructField("day_volume", StringType(), True),
    StructField("change", DoubleType(), True),
    StructField("change_percent", DoubleType(), True)
])

lines = spark.readStream.format("json") \
    .schema(schema) \
    .option("path", "../data/yfinance") \
    .option("maxFilesPerTrigger", 5) \
    .load()

query = lines.writeStream.format("console") \
    .outputMode("append") \
    .option("checkpointLocation", "../data/yfinance_checkpoint") \
    .start()

query.awaitTermination()