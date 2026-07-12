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
    df = df.withColumn("Symbol", functions.regexp_extract(functions.col("source_file"), r"([^/]+)\.csv$", 1))
    df = df.drop("source_file")

    df.head(1)

    return


if __name__ == "__main__":
    app.run()
