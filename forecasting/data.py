from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from forecasting.indicators import compute_stock_indicators
from forecasting.models import StockRating

FNSPID_DIR = Path(__file__).resolve().parent.parent / "data" / "fnspid"
RAW_STOCK_PRICES_DIR = FNSPID_DIR / "stock_prices" / "*.csv"
CACHE_RAW_STOCK_PRICES_DIR = FNSPID_DIR / "stock_prices.parquet"
CACHE_METRICS_DATASET_DIR = FNSPID_DIR / "stock_features_dataset.parquet"

STOCK_PRICE_COLUMNS = ["date", "volume", "open", "high", "low", "close", "adj close"]

# A label looks `horizon` rows ahead. Rows whose lookahead spans more calendar days
# than this (missing price data) are dropped, and splits leave a gap this wide.
MAX_HORIZON_SPAN_DAYS = 14
LABEL_HORIZON = 7


def load_or_build_raw_stock_prices(force_build=False) -> pd.DataFrame:
    """
    Combine all stock price CSVs into one Dataframe and then save the result to a parquet file
    """

    if CACHE_RAW_STOCK_PRICES_DIR.exists() and not force_build:
        return pd.read_parquet(CACHE_RAW_STOCK_PRICES_DIR)

    # Rescale open, high and low by the adjusted close value
    query = f"""
        WITH raw AS (
            SELECT
                upper(regexp_extract(filename, '([^/]+)\\.csv$', 1)) AS symbol,
                date::DATE AS date,
                volume::BIGINT AS volume,
                open::DOUBLE AS open,
                high::DOUBLE AS high,
                low::DOUBLE AS low,
                close::DOUBLE AS close,
                "adj close"::DOUBLE AS adj_close,
            FROM read_csv('{RAW_STOCK_PRICES_DIR}', filename=True, union_by_name=true)
        )
        SELECT
            *,
            open * adj_close / nullif(close, 0) AS adj_open,
            high * adj_close / nullif(close, 0) AS adj_high,
            low * adj_close / nullif(close, 0) AS adj_low,
        FROM raw
        WHERE adj_close > 0 and close > 0
        ORDER BY symbol, date
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


def label_stock(stock_df: pd.DataFrame, horizon: int = LABEL_HORIZON, k: float = 0.5):
    """
    Add labeling of either UP, DOWN, or HOLD depending on a threshold based on a
    forward return (log) calculation, measured on split adjusted closes.

    Expects rows of a single ticker symbol, already sorted by date ascending.

    i.e threshold = k * ewm_std * sqrt(horizon)
    Where ewm_std is the exponentially weighted moving standard deviation of the daily log
    returns within a given 'span'.

    Rows whose `horizon` rows ahead span more than MAX_HORIZON_SPAN_DAYS calendar days
    are dropped, since missing price data means they are not a true `horizon`-day return.
    """

    forward_return_log = np.log(
        stock_df["adj_close"].shift(-horizon) / stock_df["adj_close"]
    )
    stock_df["forward_return"] = forward_return_log

    span_days = (stock_df["date"].shift(-horizon) - stock_df["date"]).dt.days
    stock_df.loc[span_days > MAX_HORIZON_SPAN_DAYS, "forward_return"] = np.nan

    # Date when the forward window closes, used to avoid lookahead leakage
    stock_df["label_end_date"] = stock_df["date"].shift(-horizon)

    # Trailing horizon-day return for the baseline, we will mask if the span exceeds MAX_HORIZON_SPAN_DAYS
    trailing_span_days = (stock_df["date"] - stock_df["date"].shift(horizon)).dt.days
    trailing_return_log = np.log(
        stock_df["adj_close"] / stock_df["adj_close"].shift(horizon)
    )
    stock_df["trailing_return"] = trailing_return_log.where(
        trailing_span_days <= MAX_HORIZON_SPAN_DAYS
    )

    daily_log_return = np.log(stock_df["adj_close"] / stock_df["adj_close"].shift(1))

    ewm_std = daily_log_return.ewm(span=21, adjust=False, min_periods=21).std()
    stock_df["threshold"] = k * ewm_std * np.sqrt(horizon)
    stock_df = stock_df.dropna(subset=["forward_return", "threshold"]).reset_index(
        drop=True
    )

    stock_df["label"] = np.select(
        condlist=[
            stock_df["forward_return"] > stock_df["threshold"],
            stock_df["forward_return"] < -stock_df["threshold"],
        ],
        choicelist=[StockRating.UP, StockRating.DOWN],
        default=StockRating.HOLD,
    )

    return stock_df


def class_weights(labels: pd.Series) -> dict[str, float]:
    """
    Inverse frequency weights to have the signals (UP, DOWN, HOLD)
    contribute more equally to the loss.
    """

    counts = labels.value_counts()

    return (len(labels) / (len(counts) * counts)).to_dict()


def build_split_datasets(
    dataset: pd.DataFrame,
    train_split: float = 0.72,
    validation_split: float = 0.13,
    days_gap: int = MAX_HORIZON_SPAN_DAYS,
):
    """
    Split the full metrics dataset (output of `load_or_build_metrics_dataset`, still
    containing `date` and `label`) into training, validation, and testing datasets.
    Split will be based on datetime.

    Split first, then call `compute_model_features` on each split, since that
    function drops `date`.

    Have a set days gap at each boundary of the split dataset so the forward windows from each set label's
    don't overlap each other.
    """

    unique_dates = np.sort(dataset["date"].unique())
    dataset_size = len(unique_dates)
    date_gap = pd.Timedelta(days=days_gap)

    train_end_date = unique_dates[int(dataset_size * train_split)]
    val_end_date = unique_dates[int(dataset_size * (train_split + validation_split))]

    train_df = dataset[dataset["date"] < train_end_date - date_gap].reset_index(
        drop=True
    )
    validation_df = dataset[
        (dataset["date"] >= train_end_date)
        & (dataset["date"] < val_end_date - date_gap)
    ].reset_index(drop=True)
    test_df = dataset[dataset["date"] >= val_end_date].reset_index(drop=True)

    return train_df, validation_df, test_df
