import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from pyspark.sql import functions as F
    from pyspark.sql import SparkSession

    return F, SparkSession


@app.cell
def _(SparkSession):
    # localhost:4040
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("nyse")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    return (spark,)


@app.cell
def _(spark):
    spark.stop()
    return


@app.cell
def _(F, spark):
    # Add symbols to the df and then save in parquet files

    nyse_df = spark.read.csv("../data/nyse/*/*.csv", header=True, inferSchema=True)
    nyse_df = nyse_df.withColumn("source_file", F.input_file_name())
    nyse_df = nyse_df.withColumn(
        "symbol",
        F.regexp_extract(F.col("source_file"), r"([^/]+)\.csv$", 1),
    )
    nyse_df = nyse_df.drop("source_file")
    nyse_df = nyse_df.dropna(subset=["timestamp", "close", "volume", "symbol"])
    nyse_df = nyse_df.dropDuplicates(["symbol", "timestamp"])
    nyse_df.repartition("symbol")

    nyse_df.write.parquet("../data/nyse-parquet/raw_data/", mode="overwrite")

    nyse_df.printSchema()
    nyse_df.count()
    return


@app.cell
def _(spark):
    df = spark.read.parquet("../data/nyse-parquet/raw_data/*")

    df.show(10)
    return (df,)


@app.cell
def _():
    from pyspark.sql.window import Window

    def trailing(n):
        """Rolling window over the last n rows, row inclusive"""
        return (
            Window.partitionBy("symbol").orderBy("timestamp").rowsBetween(-(n - 1), 0)
        )

    window = Window.partitionBy("symbol").orderBy("timestamp")
    return trailing, window


@app.cell
def _(F, df, window):
    # Daily returns per symbol
    df_daily_returns = df.withColumn("prev_close", F.lag("close").over(window))
    df_daily_returns = df_daily_returns.withColumn(
        "daily_return",
        F.try_divide(
            F.col("close") - F.col("prev_close"),
            F.col("prev_close"),
        ),
    )

    df_daily_returns.select("symbol", "timestamp", "close", "daily_return").show(10)
    return (df_daily_returns,)


@app.cell
def _(F, df, trailing):
    # Moving averages 20, 50, 200 days
    # Golden cross, when sma_50 is above sma_200, bullish signal

    df_sma = df
    df_sma = df_sma.withColumn("sma_20", F.avg("close").over(trailing(20)))
    df_sma = df_sma.withColumn("sma_50", F.avg("close").over(trailing(50)))
    df_sma = df_sma.withColumn("sma_200", F.avg("close").over(trailing(200)))
    df_sma = df_sma.withColumn(
        "golden_cross",
        F.when(F.col("sma_50") > F.col("sma_200"), 1).otherwise(
            0
        ),
    )

    df_sma.show(20)
    return


@app.cell
def _(F, df_daily_returns, trailing):
    # Volatility
    df_volatility = df_daily_returns.withColumn(
        "volatility_14d", F.stddev("daily_return").over(trailing(14))
    )
    df_volatility = df_volatility.withColumn(
        "volatility_21d", F.stddev("daily_return").over(trailing(21))
    )

    df_volatility.show(20)
    return


@app.cell
def _(F, df, window):
    # Momentum, 7 and 14 day price change
    df_momentum = df.withColumn("close_7d_ago", F.lag("close", 7).over(window))
    df_momentum = df_momentum.withColumn("close_14d_ago", F.lag("close", 14).over(window))
    df_momentum = df_momentum.withColumn(
        "momentum_7d",
        F.try_divide(
            F.col("close") - F.col("close_7d_ago"), F.col("close_7d_ago")
        ),
    )
    df_momentum = df_momentum.withColumn(
        "momentum_14d",
        F.try_divide(
            F.col("close") - F.col("close_14d_ago"), F.col("close_14d_ago")
        ),
    )

    df_momentum.show(20)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
