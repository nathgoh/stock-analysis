from pathlib import Path

import duckdb
import pandas as pd

RAW_STOCK_PRICES_DIR = Path("../data/fsid/stock_prices/*.csv")

STOCK_PRICE_COLUMNS = ["date", "volume", "open", "high", "low", "close", "adj close"]

def load_stock_prices() -> pd.DataFrame:
    """
    Combine all stock price CSVs into one Dataframe and then save the result to a parquet file
    """

    if Path("../data/fsid/stock_prices.parquet").exists():
        return pd.read_parquet("../data/fsid/stock_prices.parquet")

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
    stock_prices_df.to_parquet("../data/fsid/stock_prices.parquet")

    return stock_prices_df

