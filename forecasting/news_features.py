from pathlib import Path

import duckdb
import pandas as pd

from forecasting.data import (
    CACHE_RAW_STOCK_PRICES_DIR,
    FNSPID_DIR,
    START_DATE,
    copy_to_parquet,
    load_or_build_metrics_dataset,
)
from forecasting.models import NewsFeatureName, SummaryNewsFeatureName
from forecasting.news import CACHE_NEWS_DIR, SENTIMENT_CACHES, load_or_build_news

CACHE_NEWS_FEATURES_DIR = FNSPID_DIR / "news_features.parquet"

# Same features, but an article dated D already counts on row D. A UTC date ends at
# 20:00 New York time, before the next open where the labels enter, so for these labels
# this isn't lookahead, just a day fresher than the default. It would be for a trade
# entered at D's close, since most timestamps are dates only.
CACHE_SAME_DAY_NEWS_FEATURES_DIR = FNSPID_DIR / "news_features_same_day.parquet"

# Articles and rows from here on are used, a year before START_DATE so the windows of
# early START_DATE rows are full. It also drops the few articles with broken dates.
NEWS_HISTORY_START = "2009-01-01"

# Recency-weighted sentiment: half-life and how far back it looks, in trading days.
EWM_HALF_LIFE = 3
EWM_WINDOW = 30

DUCKDB_MEMORY_LIMIT = "12GB"

# Features that are 0 rather than missing on rows without any news.
NEWS_COUNT_FEATURES = [
    NewsFeatureName.NEWS_COUNT_1D,
    NewsFeatureName.NEWS_COUNT_7D,
    NewsFeatureName.NEGATIVE_COUNT_1D,
    SummaryNewsFeatureName.SUMMARY_COUNT_7D,
]


def news_features_query(same_day: bool = False) -> str:
    """
    One row per stock price row (symbol, date) from NEWS_HISTORY_START, with its
    `NewsFeatureName` and `SummaryNewsFeatureName` features.

    An article dated D (its UTC date, most timestamps carry no time of day) becomes
    available on the first trading day after D, so a row only sees news published before
    its date (conservative). With `same_day`, it's available on D itself if D is a
    trading day: still before the label's entry at the next open.
    Weekend and holiday articles pile up onto the next trading day.

    Windows are in trading days of the market calendar (every date with prices), so a
    stock's missing price rows don't stretch them. Sentiment features are missing when
    their window has no articles, counts are 0.
    """

    available = ">=" if same_day else ">"
    decay = f"pow(0.5, lag / {EWM_HALF_LIFE})"
    return f"""
        WITH calendar AS (
            SELECT date, row_number() OVER (ORDER BY date) AS day_idx
            FROM (
                SELECT DISTINCT date FROM '{CACHE_RAW_STOCK_PRICES_DIR}'
                WHERE date >= '{NEWS_HISTORY_START}'
            )
        ),
        articles AS (
            SELECT
                news.symbol,
                news.published_at::DATE::TIMESTAMP AS published_date,
                -- In double precision, the std below loses too much from float32
                -- rounding when the variance is close to 0.
                titles.sentiment::DOUBLE AS sentiment,
                titles.p_positive > greatest(titles.p_negative, titles.p_neutral)
                    AS is_positive,
                titles.p_negative > greatest(titles.p_positive, titles.p_neutral)
                    AS is_negative,
                summaries.sentiment::DOUBLE AS summary_sentiment,
            FROM '{CACHE_NEWS_DIR}' AS news
            JOIN '{SENTIMENT_CACHES["title"]}' AS titles USING (title)
            LEFT JOIN '{SENTIMENT_CACHES["lsa_summary"]}' AS summaries
                USING (lsa_summary)
            WHERE news.published_at >= '{NEWS_HISTORY_START}'
                AND news.symbol IN (
                    SELECT DISTINCT symbol FROM '{CACHE_RAW_STOCK_PRICES_DIR}'
                )
        ),
        daily AS (
            -- Articles per symbol and the trading day they become available.
            SELECT
                articles.symbol,
                calendar.day_idx,
                count(*) AS n,
                sum(sentiment) AS s,
                sum(sentiment * sentiment) AS s2,
                count(*) FILTER (is_positive) AS n_positive,
                count(*) FILTER (is_negative) AS n_negative,
                count(summary_sentiment) AS n_summary,
                sum(summary_sentiment) AS s_summary,
            FROM articles
            ASOF JOIN calendar ON calendar.date {available} articles.published_date
            GROUP BY ALL
        ),
        windows AS (
            -- Each day's articles are spread over the days whose windows include them,
            -- `lag` trading days later.
            SELECT
                symbol,
                day_idx + lag AS day_idx,
                sum(n) FILTER (lag = 0) AS n_1d,
                sum(s) FILTER (lag = 0) AS s_1d,
                sum(n_positive) FILTER (lag = 0) AS n_positive_1d,
                sum(n_negative) FILTER (lag = 0) AS n_negative_1d,
                sum(n) FILTER (lag < 3) AS n_3d,
                sum(s) FILTER (lag < 3) AS s_3d,
                sum(n) FILTER (lag BETWEEN 3 AND 9) AS n_prev_7d,
                sum(s) FILTER (lag BETWEEN 3 AND 9) AS s_prev_7d,
                sum(n) FILTER (lag < 7) AS n_7d,
                sum(s) FILTER (lag < 7) AS s_7d,
                sum(s2) FILTER (lag < 7) AS s2_7d,
                sum(n * {decay}) AS n_ewm,
                sum(s * {decay}) AS s_ewm,
                sum(n_summary) FILTER (lag = 0) AS n_summary_1d,
                sum(s_summary) FILTER (lag = 0) AS s_summary_1d,
                sum(n_summary) FILTER (lag < 7) AS n_summary_7d,
                sum(s_summary) FILTER (lag < 7) AS s_summary_7d,
                sum(n_summary * {decay}) AS n_summary_ewm,
                sum(s_summary * {decay}) AS s_summary_ewm,
            FROM daily, range({EWM_WINDOW}) AS lags(lag)
            GROUP BY ALL
        ),
        price_rows AS (
            SELECT symbol, date, day_idx
            FROM '{CACHE_RAW_STOCK_PRICES_DIR}'
            JOIN calendar USING (date)
        ),
        last_news AS (
            SELECT price_rows.symbol, price_rows.day_idx, daily.day_idx AS last_day_idx
            FROM price_rows
            ASOF JOIN daily
                ON daily.symbol = price_rows.symbol
                AND daily.day_idx <= price_rows.day_idx
        )
        SELECT
            symbol,
            date,
            coalesce(n_1d, 0) AS {NewsFeatureName.NEWS_COUNT_1D},
            coalesce(n_7d, 0) AS {NewsFeatureName.NEWS_COUNT_7D},
            s_1d / n_1d AS {NewsFeatureName.SENTIMENT_MEAN_1D},
            s_7d / n_7d AS {NewsFeatureName.SENTIMENT_MEAN_7D},
            CASE WHEN n_7d >= 2
                THEN sqrt(greatest(s2_7d / n_7d - pow(s_7d / n_7d, 2), 0))
            END AS {NewsFeatureName.SENTIMENT_STD_7D},
            n_positive_1d / n_1d AS {NewsFeatureName.POSITIVE_RATIO_1D},
            coalesce(n_negative_1d, 0) AS {NewsFeatureName.NEGATIVE_COUNT_1D},
            -- Last 3 trading days against the 7 before them.
            s_3d / n_3d - s_prev_7d / n_prev_7d AS {NewsFeatureName.SENTIMENT_CHANGE_3D},
            s_ewm / n_ewm AS {NewsFeatureName.SENTIMENT_EWM},
            day_idx - last_day_idx AS {NewsFeatureName.DAYS_SINCE_LAST_NEWS},
            coalesce(n_summary_7d, 0) AS {SummaryNewsFeatureName.SUMMARY_COUNT_7D},
            s_summary_1d / nullif(n_summary_1d, 0)
                AS {SummaryNewsFeatureName.SUMMARY_SENTIMENT_MEAN_1D},
            s_summary_7d / nullif(n_summary_7d, 0)
                AS {SummaryNewsFeatureName.SUMMARY_SENTIMENT_MEAN_7D},
            s_summary_ewm / nullif(n_summary_ewm, 0)
                AS {SummaryNewsFeatureName.SUMMARY_SENTIMENT_EWM},
        FROM price_rows
        LEFT JOIN windows USING (symbol, day_idx)
        LEFT JOIN last_news USING (symbol, day_idx)
        ORDER BY symbol, date
    """


def load_or_build_news_features(
    force_build=False, same_day: bool = False, start_date: str | None = START_DATE
) -> pd.DataFrame:
    """
    Build the news features, see `news_features_query`, and cache them as parquet.
    Needs the FinBERT scores, run `forecasting.sentiment` first. `start_date` only
    filters which rows are returned.
    """

    path = CACHE_SAME_DAY_NEWS_FEATURES_DIR if same_day else CACHE_NEWS_FEATURES_DIR
    if not path.exists() or force_build:
        missing = [str(p) for p in SENTIMENT_CACHES.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing FinBERT scores {missing}, run "
                "`uv run --group news-features python -m forecasting.sentiment`"
            )
        if not CACHE_NEWS_DIR.exists():
            load_or_build_news(columns=["symbol"])
        build_news_features(path, same_day)

    filters = None if start_date is None else [("date", ">=", pd.Timestamp(start_date))]
    return pd.read_parquet(path, filters=filters)


def build_news_features(path: Path, same_day: bool = False) -> None:
    # The query inner-joins the headline scores, so unscored headlines would silently
    # drop out of the features.
    (unscored,) = (
        duckdb.connect()
        .execute(
            f"""
            SELECT count(DISTINCT title) FROM '{CACHE_NEWS_DIR}'
            ANTI JOIN '{SENTIMENT_CACHES["title"]}' USING (title)
            """
        )
        .fetchall()[0]
    )
    if unscored:
        raise RuntimeError(
            f"{unscored} news headlines have no FinBERT score (the news was rebuilt?), "
            "run `uv run --group news-features python -m forecasting.sentiment`"
        )

    query = news_features_query(same_day)
    # Features as float32, like the price indicators.
    columns = [str(c) for c in [*NewsFeatureName, *SummaryNewsFeatureName]]
    casts = ", ".join(f"{c}::FLOAT AS {c}" for c in columns)
    con = duckdb.connect(config={"memory_limit": DUCKDB_MEMORY_LIMIT})
    copy_to_parquet(con, f"SELECT symbol, date, {casts} FROM ({query})", path)


def load_fusion_dataset(
    start_date: str | None = START_DATE, same_day: bool = False
) -> pd.DataFrame:
    """
    The metrics dataset with the news features joined on (symbol, date). It has the
    same rows, in the same order. Rows without any news have counts of 0 and missing
    sentiment features.
    """

    dataset = load_or_build_metrics_dataset(start_date=start_date)
    news = load_or_build_news_features(same_day=same_day, start_date=start_date)
    fused = dataset.merge(news, on=["symbol", "date"], how="left", validate="1:1")
    del dataset, news

    counts = [str(c) for c in NEWS_COUNT_FEATURES]
    fused[counts] = fused[counts].fillna(0)
    return fused


if __name__ == "__main__":
    load_or_build_news_features()
    load_or_build_news_features(same_day=True)
