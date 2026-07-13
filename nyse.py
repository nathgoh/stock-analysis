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
    df = df.dropna(subset=["timestamp", "close", "volume", "symbol"])
    df = df.dropDuplicates(["symbol", "timestamp"])


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
    return Window, df_windowed


@app.cell
def _(df_windowed, functions):
    # Volatility
    volatility = df_windowed.groupBy("symbol").agg(functions.stddev("daily_return").alias("volatility")).orderBy(functions.desc("volatility"))

    volatility.show(20)
    return


@app.cell
def _(Window, df, functions):
    # Moving averages 20, 50, 200 days

    # Golden cross, when sma_50 is above sma_200, bullish signal

    def trailing(n):
        """Rolling window over the last n rows, row inclusive"""
        return Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-(n - 1), 0)

    df_sma = df
    df_sma = df_sma.withColumn("sma_20", functions.avg("close").over(trailing(20)))
    df_sma = df_sma.withColumn("sma_50", functions.avg("close").over(trailing(50)))
    df_sma = df_sma.withColumn("sma_200", functions.avg("close").over(trailing(200)))
    df_sma = df_sma.withColumn(
        "golden_cross",
        functions.when(functions.col("sma_50") > functions.col("sma_200"), 1).otherwise(0),
    )


    df_sma.show(20)
    return


if __name__ == "__main__":
    app.run()
