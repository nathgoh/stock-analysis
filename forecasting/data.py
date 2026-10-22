from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from forecasting.indicators import compute_market_indicators, compute_stock_indicators
from forecasting.models import StockRating

FNSPID_DIR = Path(__file__).resolve().parent.parent / "data" / "fnspid"
RAW_STOCK_PRICES_DIR = FNSPID_DIR / "stock_prices" / "*.csv"
CACHE_RAW_STOCK_PRICES_DIR = FNSPID_DIR / "stock_prices.parquet"
CACHE_METRICS_DATASET_DIR = FNSPID_DIR / "stock_features_dataset.parquet"

STOCK_PRICE_COLUMNS = ["date", "volume", "open", "high", "low", "close", "adj close"]

# A label enters at the next row's open and exits `horizon` rows after that. Rows whose
# exit is more calendar days ahead than this (missing price data) are dropped, and
# splits leave a gap this wide.
MAX_HORIZON_SPAN_DAYS = 14
LABEL_HORIZON = 7


def copy_to_parquet(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    """
    Write a DuckDB query result straight to parquet. It goes to a temporary file first
    and is only moved into place once finished, so a failed build never leaves a partial
    file that looks like a cache.
    """

    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    con.execute(f"COPY ({query}) TO '{tmp_path}' (FORMAT parquet, COMPRESSION zstd)")
    tmp_path.replace(path)


def load_or_build_raw_stock_prices(force_build=False) -> pd.DataFrame:
    """
    Combine all stock price CSVs into one Dataframe and then save the result to a parquet file
    """

    if CACHE_RAW_STOCK_PRICES_DIR.exists() and not force_build:
        return pd.read_parquet(CACHE_RAW_STOCK_PRICES_DIR)

    # Rescale open, high and low by the adjusted close value. date is a TIMESTAMP rather
    # than a DATE so pandas reads it as datetime64, which the labeling relies on.
    query = f"""
        WITH raw AS (
            SELECT
                upper(regexp_extract(filename, '([^/]+)\\.csv$', 1)) AS symbol,
                date::TIMESTAMP AS date,
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
    copy_to_parquet(duckdb.connect(), query, CACHE_RAW_STOCK_PRICES_DIR)

    return pd.read_parquet(CACHE_RAW_STOCK_PRICES_DIR)


def load_or_build_metrics_dataset(force_build=False, start_date: str | None = None):
    """
    Build the metrics based on the raw stock prices, see `build_metrics_dataset`.

    Save the resulting dataset as a parquet file so we don't need to keep rebuilding it
    unless we choose to. The cache always holds the full history, `start_date` only
    filters which rows are returned.
    """

    if CACHE_METRICS_DATASET_DIR.exists() and not force_build:
        filters = (
            None if start_date is None else [("date", ">=", pd.Timestamp(start_date))]
        )
        return pd.read_parquet(CACHE_METRICS_DATASET_DIR, filters=filters)

    dataset_df = build_metrics_dataset(load_or_build_raw_stock_prices())

    # Write to a temporary file first so a failed write never leaves a partial cache.
    tmp_path = CACHE_METRICS_DATASET_DIR.with_suffix(".tmp.parquet")
    dataset_df.to_parquet(tmp_path)
    tmp_path.replace(CACHE_METRICS_DATASET_DIR)

    if start_date is not None:
        dataset_df = dataset_df[dataset_df["date"] >= start_date].reset_index(drop=True)

    return dataset_df


def build_metrics_dataset(stock_prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the indicators per ticker symbol, then the cross-sectional market indicators,
    then the labels per ticker symbol.

    The market indicators are computed before labeling, so the rows labeling drops (no
    complete forward window) still count towards that date's market.
    """

    stocks = [
        compute_stock_indicators(group)
        for _, group in stock_prices_df.groupby("symbol", sort=False)
    ]
    dataset_df = pd.concat(stocks, ignore_index=True)
    del stocks

    dataset_df = compute_market_indicators(dataset_df)

    stocks = [
        label_stock(group) for _, group in dataset_df.groupby("symbol", sort=False)
    ]
    dataset_df = pd.concat(stocks, ignore_index=True)
    del stocks

    return dataset_df.dropna(subset=dataset_df.columns).reset_index(drop=True)


def label_stock(stock_df: pd.DataFrame, horizon: int = LABEL_HORIZON, k: float = 0.5):
    """
    Add labeling of either UP, DOWN, or HOLD depending on a threshold based on a
    forward return (log) calculation, measured on split adjusted opens.

    The decision is made after the close of each row's date, so the trade enters at the
    next row's open and exits `horizon` rows later, also at the open:
    forward_return = log(adj_open[t + 1 + horizon] / adj_open[t + 1])

    Expects rows of a single ticker symbol, already sorted by date ascending.

    i.e threshold = k * ewm_std * sqrt(horizon)
    Where ewm_std is the exponentially weighted moving standard deviation of the daily log
    returns within a given 'span'.

    Rows whose exit is more than MAX_HORIZON_SPAN_DAYS calendar days after the row's
    date are dropped, since missing price data means they are not a true `horizon`-day
    return. So are rows whose entry or exit open is missing (non-positive in old data).
    """

    adj_open = stock_df["adj_open"].where(stock_df["adj_open"] > 0)
    entry_open = adj_open.shift(-1)
    exit_open = adj_open.shift(-(horizon + 1))
    stock_df["forward_return"] = np.log(exit_open / entry_open)

    # Date of the exit open, the label is known from then on. Used to avoid lookahead leakage
    stock_df["label_end_date"] = stock_df["date"].shift(-(horizon + 1))

    span_days = (stock_df["label_end_date"] - stock_df["date"]).dt.days
    stock_df.loc[span_days > MAX_HORIZON_SPAN_DAYS, "forward_return"] = np.nan

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


def sample_stock_symbols(
    dataset: pd.DataFrame, fraction: float, seed: int = 0
) -> pd.DataFrame:
    """
    Sample a seeded subset of stock symbols, keep all the time series data of one stock symbol together.
    """

    symbols = np.sort(dataset["symbol"].unique())
    n_keep = max(1, round(len(symbols) * fraction))
    keep_symbols = np.random.default_rng(seed).choice(
        symbols, size=n_keep, replace=False
    )

    return dataset[dataset["symbol"].isin(keep_symbols)].reset_index(drop=True)


def walk_forward_splits(
    dataset: pd.DataFrame,
    n_folds: int,
    initial_train_fraction: float = 0.5,
    train_split: float = 0.72,
    validation_split: float = 0.13,
    days_gap: int = MAX_HORIZON_SPAN_DAYS,
    boundary_dates: np.ndarray | None = None,
):
    """
    To be used in walk forward validation, rolling window approach where the model is trained and
    tested on consecutive periods. Allows the model to be updated continuously,
    closely simulating real-world forecasting scenarios.

    We will fold over the period everything before the test set, get an initial fraction as our training
    set, the the rest if divided cross n_folds consecutive validation blocks. Each fold then trains on
    all the dates before its block (excluding days_gap), prevent leakage into validation set.

    The fold boundaries come from the dates of `dataset`, or from `boundary_dates` when
    given. Pass the full dataset's dates when `dataset` is a symbol subset, so the
    boundaries (and the test cut) match the ones from `build_split_datasets`.
    """

    if boundary_dates is None:
        boundary_dates = dataset["date"].unique()
    unique_dates = np.sort(boundary_dates)
    date_gap = pd.Timedelta(days=days_gap)

    test_start = unique_dates[int(len(unique_dates) * (train_split + validation_split))]
    validation_dates = unique_dates[unique_dates < test_start - date_gap]

    # n_folds + 1 spacing
    # i.e. 1000 validations dates, initial_train_fraction=0.5
    # dates 0 - 500 (train) 500 - 625 (val 0) ... 875 - 1000 (val 3)
    spaces = np.linspace(
        int(len(validation_dates) * initial_train_fraction),
        len(validation_dates),
        n_folds + 1,
    ).astype(int)

    folds = []
    for start, end in zip(spaces[:-1], spaces[1:]):
        val_start = validation_dates[start]

        in_val = dataset["date"] >= val_start
        # boolean mask, keep a row True only if was already True in_val and
        # its date is on or vefore validateion_dates[end - 1]
        in_val &= dataset["date"] <= validation_dates[end - 1]
        train_df = dataset[dataset["date"] < val_start - date_gap]
        folds.append(
            (train_df.reset_index(drop=True), dataset[in_val].reset_index(drop=True))
        )

    return folds


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


def split_date_ranges(**splits: pd.DataFrame) -> dict[str, list[str]]:
    """
    First and last date of each split, e.g. `split_date_ranges(train=train_df, ...)`,
    saved alongside a model so its evaluation can be reproduced.
    """

    return {
        name: [df["date"].min().date().isoformat(), df["date"].max().date().isoformat()]
        for name, df in splits.items()
    }
