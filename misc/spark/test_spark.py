from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()

print(f"Spark version: {spark.version}")

df = spark.range(10)
df.show()

spark.stop()

# Streaming
spark = SparkSession.builder.master("local[*]").appName("simple_stream").getOrCreate()

stream_df = spark.readStream.format("rate").option("rowsPerSecond", 1).load()

result_df = stream_df.withColumn("result", F.col("value") + F.lit(1))

result_df.writeStream.outputMode("append").option("truncate", False).format(
    "console"
).start().awaitTermination()
