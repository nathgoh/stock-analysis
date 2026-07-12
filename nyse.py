import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pyspark
    from pyspark.sql import SparkSession

    return (SparkSession,)


@app.cell
def _(SparkSession):
    # localhost:4040
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName('nyse') \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    return (spark,)


@app.cell
def _(spark):
    spark.stop()
    return


@app.cell
def _(spark):
    nyse_df = spark.read.csv("nyse/*/*.csv", header=True, inferSchema=True)
    return (nyse_df,)


@app.cell
def _(nyse_df):
    nyse_df.printSchema()
    nyse_df.count()
    return


@app.cell
def _(nyse_df):
    from pyspark.sql import functions

    df = nyse_df

    df = df.withColumn("source_file", functions.input_file_name())
    df = df.withColumn("symbol", functions.regexp_extract(functions.col("source_file"), r"([^/]+)\.csv$", 1))
    df = df.drop("source_file")

    df = df.repartition("symbol")

    df.show(10)

    return df, functions


@app.cell
def _(df, functions):
    # Daily returns per symbol
    from pyspark.sql.window import Window

    window = Window.partitionBy("Symbol").orderBy("timestamp")

    df_windowed = df.withColumn("prev_close", functions.lag("close").over(window)).withColumn("daily_return", functions.try_divide(functions.col("close") - functions.col("prev_close"), functions.col("prev_close")))

    df_windowed.select("symbol", "timestamp", "close", "daily_return").show(10)
    return (df_windowed,)


@app.cell
def _(df_windowed, functions):
    # Volatility
    volatility = df_windowed.groupBy("symbol").agg(functions.stddev("daily_return").alias("volatility")).orderBy(functions.desc("volatility"))

    volatility.show(20)
    return


if __name__ == "__main__":
    app.run()
