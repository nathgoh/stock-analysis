from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from forecasting.indicators import compute_stock_indicators

RAW_STOCK_PRICES_DIR = Path("../data/fnspid/stock_prices/*.csv")
CACHE_RAW_STOCK_PRICES_DIR = Path("../data/fnspid/stock_prices.parquet")
CACHE_METRICS_DATASET_DIR = Path("../data/fnspid/stock_features_dataset.parquet")

STOCK_PRICE_COLUMNS = ["date", "volume", "open", "high", "low", "close", "adj close"]


def load_or_build_raw_stock_prices(force_build=False) -> pd.DataFrame:
    """
    Combine all stock price CSVs into one Dataframe and then save the result to a parquet file
    """

    if CACHE_RAW_STOCK_PRICES_DIR.exists() and not force_build:
        return pd.read_parquet(CACHE_RAW_STOCK_PRICES_DIR)

    query = f"""
        SELECT
            upper(regexp_extract(filename, '([^/]+)\\.csv$', 1)) AS symbol,
            date::DATE AS date,
            volume::BIGINT AS volume,
            open::DOUBLE AS open,
            high::DOUBLE AS high,
            low::DOUBLE AS low,
            close::DOUBLE AS close,
            "adj close"::DOUBLE AS "adj_close",
        FROM read_csv('{RAW_STOCK_PRICES_DIR}', filename=True)
    """
    stock_prices_df = duckdb.execute(query).df()

    # Save to parquet file
    stock_prices_df.to_parquet(CACHE_RAW_STOCK_PRICES_DIR)

    return stock_prices_df


def load_or_build_metrics_dataset(force_build=False):
    """
    Build the metrics based on the raw stock prices, metrics building will be done per ticker symbol.

    Save the resulting dataset as a parquet file so we don't need to keep rebuilding it
    unless we choose to.
    """

    if CACHE_METRICS_DATASET_DIR.exists() and not force_build:
        return pd.read_parquet(CACHE_METRICS_DATASET_DIR)

    stock_prices_df = load_or_build_raw_stock_prices()

    stocks = []
    for _, group in stock_prices_df.groupby("symbol", sort=False):
        group = compute_stock_indicators(group)
        group = label_stock(group)
        stocks.append(group)
    dataset_df = pd.concat(stocks, ignore_index=True)
    dataset_df = dataset_df.dropna(subset=dataset_df.columns)

    dataset_df.to_parquet(CACHE_METRICS_DATASET_DIR)

    return dataset_df


def label_stock(stock_df: pd.DataFrame, horizon: int = 7, k: float = 0.5):
    """
    Add labeling of either UP, DOWN, or HOLD depending on a threshold based on a
    forward return (log) calculation.

    i.e threshold = k * rolling standard deviation (std) * sqrt(horizon)
    rolling std is a recent volatility of the stock (i.e ATR or rolling std of returns over n days).
    """

    forward_return_log = np.log(stock_df["close"].shift(-horizon) / stock_df["close"])
    stock_df["forward_return"] = forward_return_log

    rolling_std = stock_df["close"].pct_change().ewm(span=21, adjust=False).std()
    stock_df["threshold"] = k * rolling_std * np.sqrt(horizon)
    stock_df = stock_df.dropna(subset=["forward_return", "threshold"]).reset_index(
        drop=True
    )

    stock_df["label"] = np.select(
        condlist=[
            stock_df["forward_return"] > stock_df["threshold"],
            stock_df["forward_return"] < -stock_df["threshold"],
        ],
        choicelist=["UP", "DOWN"],
        default="HOLD",
    )

    return stock_df


def build_split_datasets(
    features_dataset: pd.DataFrame,
    train_split: float = 0.72,
    validation_split: float = 0.13,
    days_gap: int = 14,
):
    """
    Build the training, validation, and testing datasets from the features dataset.
    Split will be based on datetime.

    Have a set days gap at each boundary of the split dataset so the forward windows from each set label's
    don't overlap each other.
    """

    unique_dates = np.sort(features_dataset["date"].unique())
    dataset_size = len(unique_dates)
    date_gap = pd.Timedelta(days=days_gap)

    train_end_date = unique_dates[int(dataset_size * train_split)]
    val_end_date = unique_dates[int(dataset_size * (train_split + validation_split))]

    train_df = features_dataset[
        features_dataset["date"] < train_end_date - date_gap
    ].reset_index(drop=True)
    validation_df = features_dataset[
        features_dataset["date"] >= train_end_date & val_end_date - date_gap
    ].reset_index(drop=True)
    test_df = features_dataset[features_dataset["date"] >= val_end_date].reset_index(
        drop=True
    )

    return train_df, validation_df, test_df
