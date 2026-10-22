# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Overview

A personal data-processing playground with two real components plus scratch work:

- `forecasting/` — XGBoost classifier that predicts a 7-trading-day UP/HOLD/DOWN rating per stock (the main, most actively developed part).
- `news_service/` — Alpaca financial news → Redpanda (Kafka) producer (historical backfill, then live stream).
- `misc/` — Spark / yfinance / marimo experiments (`pyspark`, `yfinance`); not part of the main pipelines.

## Commands

Python 3.13, managed with `uv`. Dependencies are split into groups in `pyproject.toml` (`forecasting`, `misc`, `news-service`); the base set has only duckdb/pandas/pyarrow/pydantic. Use `uv sync --group forecasting` (etc.) to install the group you need.

```bash
uv run ruff check .          # lint (ruff and ty are base dependencies)
uv run ruff format .
uv run ty check              # type check

# Forecasting (run as modules from repo root — imports are absolute, e.g. `forecasting.data`)
uv run python -m forecasting.training.xgboost_model   # train once with hand-picked params, print test comparison
uv run python -m forecasting.training.tuning          # Optuna search -> refit -> evaluate -> save model

# News service
make up                      # start Redpanda (2 brokers) + console via docker compose
make run-news                # uv run python -m news_service
make down | logs | ps | clean
```

There is no test suite (`.pytest_cache` exists but there is no `tests/` directory).

Config/secrets: `configs/config.yaml` (Kafka bootstrap server + topic names), `configs/stocks.yaml` (symbols for the news service), `.env` holds `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`. `news_service` reads configs via relative paths, so run it from the repo root.

## Forecasting architecture

Data flow: raw CSVs → cached parquet → per-symbol indicators + labels → time-based splits → XGBoost.

- **`data.py`** owns the dataset. `load_or_build_raw_stock_prices` uses DuckDB to read `data/fnspid/stock_prices/*.csv` (FNSPID dataset; symbol is taken from the filename) into `data/fnspid/stock_prices.parquet`. `load_or_build_metrics_dataset` runs per-symbol indicators, then cross-sectional market indicators, then per-symbol labeling (`build_metrics_dataset`) and caches `data/fnspid/stock_features_dataset.parquet` (full history; its `start_date` argument only filters the returned rows). **Both loaders return the cached parquet if it exists** — pass `force_build=True` (or delete the parquet) after changing indicators, labeling, or feature columns, otherwise you'll silently train on stale data.
- **`indicators.py` / `models.py`** — indicators are a registry: add an `IndicatorName` enum member in `models.py` and a matching `IndicatorSpec` entry in the `INDICATORS` dict in `indicators.py`. All indicators use split-adjusted prices (`adj_*` columns, computed in the DuckDB query), so raw price levels never become features. Cross-sectional indicators (market median return, excess return, per-date percentile ranks) live in `MarketIndicatorName` / `compute_market_indicators`, since they need every symbol's row for a date.
- **`features.py`** — `compute_model_features` derives the feature matrix by *dropping* `NON_FEATURE_COLUMNS` from the dataset. Any new non-feature column added to the dataset (label helpers, price levels, anything that looks ahead) **must be added to that list** or it leaks into the model as a feature.
- **Labels** (`label_stock`) — UP/DOWN/HOLD by comparing the forward 7-row log return, entering at the next row's open and exiting at the open 7 rows later (`log(adj_open[t+8] / adj_open[t+1])`, decision after the close of `t`), to `k * ewm_std * sqrt(horizon)` (volatility-scaled threshold). Rows whose exit is >14 calendar days ahead (missing data) or whose entry/exit open is missing (`open <= 0`, common before ~1990) are dropped. `label_end_date` is the exit date. It also emits `label_end_date` and `trailing_return`, which are used only by evaluation baselines, not as features.
- **Leakage control is central.** Splits are chronological by *date* (not by row or symbol), with a `days_gap` (14 days) at each boundary so forward label windows don't overlap. `build_split_datasets` gives train/val/test (72/13/15); `walk_forward_splits` carves expanding-window folds out of the pre-test region. When searching on a symbol subset, pass `boundary_dates=dataset["date"].unique()` so fold and test boundaries match the full dataset. `tuning.py` asserts the no-overlap invariants — keep them if you change splitting.
- **`training/xgboost_model.py`** — `fit_xgboost` applies inverse-frequency class weights to train *and* eval sets; `BASE_PARAMS` targets GPU (`device: cuda`, with `cupy-cuda13x` in the group), so training needs a CUDA GPU or a changed `device`. `predict_proba` caps prediction at the best iteration. `evaluate_model` compares against baselines (majority class, persistence via `resolved_label_lag`, trailing-return momentum, class prior) on the test rows where a persistence label is available. Model is saved to `data/artifacts/xgboost_model.ubj`.
- **`training/tuning.py`** — multi-objective Optuna (minimize log loss, maximize macro F1) with walk-forward CV on a 30% symbol sample, stored in SQLite at `data/artifacts/optuna_study.db` with `load_if_exists=True`, so runs resume. A resumed study must use the same `SEARCH_SYMBOL_FRACTION`/`SEARCH_SEED`/`N_FOLDS` (enforced via `search_config` user attr) — use a new study name or storage if you change them. `min_child_weight` is scaled by the symbol fraction only inside the objective; the top Pareto trials are then refit on all symbols and picked by validation macro F1.

`data/` is untracked (large datasets, parquet caches, Optuna DB, model artifacts), as is `.env`.

## News service

`news_service/__main__.py` creates two topics (live + historical, 2 partitions, RF 1), runs `produce_historical_news` once, then blocks in `news_stream` for live news. Messages are modeled with `utils/models.py:NewsMessage` (pydantic); `utils/text_utils.py` cleans HTML/markdown from article text. Redpanda is exposed to the host on `localhost:19092` (matches `bootstrap_servers` in config).
