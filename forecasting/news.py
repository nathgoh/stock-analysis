from pathlib import Path

import duckdb
import pandas as pd

from forecasting.data import CACHE_RAW_STOCK_PRICES_DIR, FNSPID_DIR, copy_to_parquet

RAW_NEWS_CSV = FNSPID_DIR / "nasdaq_external_data.csv"
CACHE_RAW_NEWS_DIR = FNSPID_DIR / "news_raw.parquet"
CACHE_NEWS_DIR = FNSPID_DIR / "news.parquet"
# CSV rows DuckDB could not parse, kept for inspection.
NEWS_REJECTS_DIR = FNSPID_DIR / "news_raw_rejects.parquet"


def build_raw_news(force_build=False) -> Path:
    """
    Parse the FNSPID news CSV into a parquet file, keeping every row so the parsing can be
    profiled. The full article bodies are left out, only titles and LSA summaries are kept.

    The CSV is ~23GB, so the result is written straight to parquet instead of going
    through pandas. Article bodies span many lines so that will cause issues with
    DuckDB's parallel CSV reader (it can start a chunk inside a quoted field), 
    so the file is read on a single thread. Rows that still fail to 
    parse are saved to NEWS_REJECTS_DIR.
    """

    if CACHE_RAW_NEWS_DIR.exists() and not force_build:
        return CACHE_RAW_NEWS_DIR

    # Timestamps are in UTC, e.g. "2023-12-16 23:00:00 UTC". raw_date is kept so dates
    # that fail to parse can be inspected.
    query = f"""
        SELECT
            upper(trim(Stock_symbol)) AS symbol,
            try_strptime(Date, '%Y-%m-%d %H:%M:%S UTC') AS published_at,
            nullif(trim(Article_title), '') AS title,
            Url AS url,
            nullif(trim(Lsa_summary), '') AS lsa_summary,
            Date AS raw_date,
        FROM read_csv(
            '{RAW_NEWS_CSV}',
            header=true,
            all_varchar=true,
            parallel=false,
            store_rejects=true
        )
    """
    con = duckdb.connect()
    copy_to_parquet(con, query, CACHE_RAW_NEWS_DIR)

    (n_rejects,) = con.execute("SELECT count(*) FROM reject_errors").fetchall()[0]
    if n_rejects:
        con.execute(
            f"COPY reject_errors TO '{NEWS_REJECTS_DIR}' (FORMAT parquet, COMPRESSION zstd)"
        )
        print(f"{n_rejects} CSV rows could not be parsed, see {NEWS_REJECTS_DIR}")

    return CACHE_RAW_NEWS_DIR


def load_or_build_news(
    force_build=False, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Clean the parsed news, drop rows missing a symbol, timestamp or title, and dedupe
    repeated headlines (same symbol, same title ignoring case, same UTC day).

    Save the result as a parquet file, `columns` selects which columns to load from it.
    """

    if not CACHE_NEWS_DIR.exists() or force_build:
        raw_news = build_raw_news()
        query = f"""
            SELECT symbol, published_at, title, url, lsa_summary
            FROM '{raw_news}'
            WHERE symbol IS NOT NULL
                AND symbol != ''
                AND published_at IS NOT NULL
                AND title IS NOT NULL
            QUALIFY row_number() OVER (
                PARTITION BY symbol, lower(title), published_at::DATE
                ORDER BY published_at, url
            ) = 1
            ORDER BY symbol, published_at
        """
        copy_to_parquet(duckdb.connect(), query, CACHE_NEWS_DIR)

    return pd.read_parquet(CACHE_NEWS_DIR, columns=columns)


def profile_news() -> None:
    """
    Print the news coverage stats used to pick the start date of the news experiments.
    """

    # Builds both parquet files if they don't exist yet.
    load_or_build_news(columns=["symbol"])

    raw, news, prices = (
        f"'{CACHE_RAW_NEWS_DIR}'",
        f"'{CACHE_NEWS_DIR}'",
        f"'{CACHE_RAW_STOCK_PRICES_DIR}'",
    )
    con = duckdb.connect()

    def show(title: str, query: str) -> None:
        print(f"\n== {title}")
        print(con.execute(query).df().to_string(index=False))

    show(
        "Parsing (raw rows)",
        f"""
        SELECT
            count(*) AS rows,
            count(*) FILTER (published_at IS NULL) AS bad_date,
            count(*) FILTER (symbol IS NULL OR symbol = '') AS no_symbol,
            count(*) FILTER (title IS NULL) AS no_title,
            count(*) FILTER (lsa_summary IS NULL) AS no_summary,
            (SELECT count(*) FROM {news}) AS clean_rows,
        FROM {raw}
        """,
    )
    show(
        "Unparseable date samples",
        f"SELECT raw_date, count(*) AS n FROM {raw} WHERE published_at IS NULL "
        "GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
    )
    show(
        "Date range",
        f"SELECT min(published_at) AS first, max(published_at) AS last FROM {news}",
    )

    # Timestamps exactly at midnight UTC probably only carry a date, not a real time.
    show(
        "Timestamps exactly at midnight UTC",
        f"""
        SELECT
            count(*) FILTER (published_at::TIME = TIME '00:00:00') AS midnight,
            round(100 * midnight / count(*), 2) AS pct,
        FROM {news}
        """,
    )
    show(
        "Hour of day (America/New_York), non-midnight-UTC rows",
        f"""
        SELECT
            hour(timezone('America/New_York', timezone('UTC', published_at))) AS hour_et,
            count(*) AS n,
        FROM {news}
        WHERE published_at::TIME != TIME '00:00:00'
        GROUP BY 1 ORDER BY 1
        """,
    )

    price_symbols = f"(SELECT DISTINCT symbol FROM {prices})"
    show(
        "Symbol overlap with stock prices",
        f"""
        SELECT
            count(DISTINCT symbol) AS news_symbols,
            count(DISTINCT symbol) FILTER (symbol IN {price_symbols}) AS matched_symbols,
            (SELECT count(*) FROM {price_symbols}) AS price_symbols,
            round(100 * count(*) FILTER (symbol IN {price_symbols}) / count(*), 2)
                AS pct_rows_matched,
        FROM {news}
        """,
    )

    # Coverage per year, only for symbols that also have prices.
    show(
        "Coverage per year (matched symbols)",
        f"""
        WITH per_symbol AS (
            SELECT year(published_at) AS year, symbol, count(*) AS n
            FROM {news}
            WHERE symbol IN {price_symbols}
            GROUP BY ALL
        ),
        prices_per_year AS (
            SELECT year(date) AS year, count(DISTINCT symbol) AS price_symbols
            FROM {prices}
            GROUP BY 1
        )
        SELECT
            year,
            sum(n) AS articles,
            count(*) AS symbols_with_news,
            any_value(price_symbols) AS price_symbols,
            round(100 * count(*) / any_value(price_symbols), 1) AS pct_symbols_covered,
            median(n) AS median_per_symbol,
        FROM per_symbol
        JOIN prices_per_year USING (year)
        GROUP BY year
        ORDER BY year
        """,
    )
    show(
        "Most covered symbols",
        f"SELECT symbol, count(*) AS n FROM {news} GROUP BY 1 ORDER BY 2 DESC LIMIT 10",
    )


if __name__ == "__main__":
    profile_news()
