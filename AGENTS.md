# AGENTS.md

Guidance for coding agents working in this repository.

## The project

A personal playground for data processing and stock forecasting. Most of the work is in `forecasting/`:
- An XGBoost model rates each stock **UP / HOLD / DOWN** over the next 7 trading days, trained on the FNSPID dataset (prices for ~7,700 symbols plus ~15M news articles).
- The news experiments are done: FinBERT news features did **not** improve the 7-day model, because news is priced in by the next open. See `docs/news-experiments.md` before doing more news work.

| Path | Contents |
|---|---|
| `forecasting/data.py` | price dataset: loading, labels, bad-price filter, splits |
| `forecasting/indicators.py`, `models.py` | feature definitions (enums + indicator registry) |
| `forecasting/features.py` | feature matrix (`NON_FEATURE_COLUMNS`) |
| `forecasting/news.py` | news parsing and cleaning, sentiment cache paths |
| `forecasting/sentiment.py` | FinBERT scoring (the only module needing torch) |
| `forecasting/news_features.py` | news features per stock and day, joined onto the price dataset |
| `forecasting/training/` | `xgboost_model.py` (fit, evaluate, save), `tuning.py` (Optuna), `evaluation.py` (ranking metrics), `experiments.py` (news experiments) |
| `news_service/`, `utils/` | Alpaca news → Redpanda (Kafka) producer, independent of `forecasting/` |
| `misc/` | Spark and yfinance experiments, not part of either pipeline |
| `docs/` | `stock-analysis-planning.md` (original plan), `news-experiments.md` (results) |

## Commands

Python 3.13 with `uv`. Base dependencies: duckdb, pandas, pyarrow, pydantic, python-dotenv, pyyaml, ruff, ty. Groups in `pyproject.toml`:
- `forecasting`: xgboost, optuna, scikit-learn, cupy
- `news-features`: torch, transformers
- `news-service`: alpaca-py, kafka-python
- `misc`: pyspark, marimo, yfinance

```bash
uv run ruff check . && uv run ruff format .
uv run --group forecasting --group news-features ty check
```

Run modules from the repo root (imports are absolute). The forecasting pipeline, in build order:

```bash
uv run python -m forecasting.news                                         # parse + clean news, print coverage
uv run --group news-features python -m forecasting.sentiment              # FinBERT scores (GPU, ~35 min first time, then only new texts)
uv run python -m forecasting.news_features                                # news features (~1 min)
uv run --group forecasting python -m forecasting.training.xgboost_model   # one fit with hand-picked params
uv run --group forecasting python -m forecasting.training.tuning          # Optuna search, refit, evaluate, save the model
uv run --group forecasting python -m forecasting.training.experiments     # news experiments (~11 min, resumable)
```

The price dataset builds on first use (about 4 minutes). News service: `make up` (Redpanda + console via docker compose), `make run-news`, `make down | logs | ps | clean`.

There is no test suite. Verify changes with ruff, ty, and a script against the real data. Do it on a symbol sample first, since full builds are slow.

## Rules that matter

1. **Cached loaders go stale.** Every `load_or_build_*` returns its parquet if the file exists. After changing the code that builds a cache, rebuild it (`force_build=True`) and everything downstream of it, or you'll silently use old data. The chain is listed under [Data files](#data-files).
2. **New dataset columns become features unless listed.** `compute_model_features` uses every column not in `features.NON_FEATURE_COLUMNS`. Add any non-feature column (prices, label helpers, anything forward-looking) there. `experiments.py` lists its features explicitly, and checks they match this set.
3. **No lookahead.**
   - Splits are by date, with a 14-day gap at each boundary.
   - News must be available before the label's entry at the next open (see the timing rule under [News](#news)).
   - `tuning.py` asserts the split invariants; keep them.
4. **`START_DATE` (`2010-01-01`, in `data.py`) is the single definition of the experiment period.** It is the loaders' default and every entrypoint uses it.
5. **Change `tuning.STUDY_NAME` whenever the dataset (labels, features, filtering) or the search settings change.** A resumed study refuses different search settings, but nothing detects a changed dataset.
6. **Sentiment scores are joined on the exact headline or summary text.**
   - Keep the news dedupe deterministic, with its total ordering.
   - After rebuilding `news.parquet`, rerun `forecasting.sentiment`; it only scores texts it hasn't seen.
   - `build_news_features` refuses to run while any headline is unscored.
7. **Resources: ~30 GB RAM, one 11 GB CUDA GPU.**
   - Full-dataset steps peak at 15–22 GB, so avoid extra full-size pandas copies.
   - Do big scans and joins in DuckDB, written straight to parquet with `data.copy_to_parquet`, which writes atomically.
   - Store features as float32.
   - XGBoost (`device: cuda`) and FinBERT expect the GPU.
8. **Untracked:** `data/fnspid/*`, `data/artifacts/*`, `.env` (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`), `uv.lock`.

## Forecasting

### Data files

All under `data/fnspid/`. Each file is built from the ones above it:

| File | Built by |
|---|---|
| `stock_prices.parquet` (from `stock_prices/*.csv`, symbol from the filename) | `data.load_or_build_raw_stock_prices` |
| `stock_features_dataset.parquet` | `data.load_or_build_metrics_dataset` |
| `news_raw.parquet` → `news.parquet` (from `nasdaq_external_data.csv`, 23 GB) | `news.build_raw_news`, `news.load_or_build_news` |
| `news_sentiment.parquet`, `news_summary_sentiment.parquet` | `sentiment.load_or_build_news_sentiment` |
| `news_features.parquet`, `news_features_same_day.parquet` | `news_features.load_or_build_news_features` |

Models, the Optuna study (`optuna_study.db`) and experiment results (`experiments/`) live in `data/artifacts/`.

### Price dataset and labels (`data.py`)

- **The cache keeps the full history (1962–2023).** `start_date` only filters what is returned, so early-2010 indicators still look back into 2009.
- **Adjusted prices.** Prices are split-adjusted (`adj_*`), so raw price levels never become features.
- **Build order.** `build_metrics_dataset` computes the per-symbol indicators and labels in one pass, then the cross-sectional market indicators. Unlabeled rows are dropped only at the end.
- **Bad prices.** A close or open more than +100% above the previous close (`MAX_DAILY_MOVE`) is treated as bad data. Rows whose 60-row lookback or label window touches such a day are dropped.
- **Label** (`label_stock`):
  - Decision after the close of row `t`; enter at the next open; exit at the open 7 rows later: `forward_return = log(adj_open[t+8] / adj_open[t+1])`.
  - UP / DOWN beyond ±`0.5 × ewm_std × √7` (`threshold`), HOLD otherwise. Roughly 27% DOWN, 43% HOLD, 31% UP.
  - Rows are dropped if the exit is more than 14 calendar days ahead or an open is missing.
  - `label_end_date` and `trailing_return` only feed evaluation baselines.
- **Splits.** `build_split_datasets` gives train / val / test by date (72 / 13 / 15): on the 2010+ data, 2010-01 → 2020-01, 2020-01 → 2021-10, 2021-11 → 2023-12. `walk_forward_splits` makes expanding folds before the test period. When working on a symbol sample, pass the full dataset's dates as `boundary_dates`.

### Features

- **Price indicators** (`IndicatorName`) form a registry: add the enum member in `models.py` and its `IndicatorSpec` in `indicators.INDICATORS`.
- **Cross-sectional indicators** (`MarketIndicatorName`, `compute_market_indicators`) compare each stock with the others on the same date: market median return, excess return, percentile ranks.
- **News features:** `NewsFeatureName` (10, from headlines) and `SummaryNewsFeatureName` (4, from article summaries).
- **Counts:** 32 price features, 46 with news.

### Training and evaluation (`training/`)

- **`xgboost_model.py`**
  - `fit_xgboost` uses inverse-frequency class weights on train and eval sets, so predicted probabilities lean toward 1/3 and aren't calibrated.
  - `predict_proba` caps prediction at the best iteration.
  - `evaluate_model` compares macro F1 and log loss against the baselines: majority class, persistence, trailing momentum, class prior.
  - `save_model` writes `xgboost_model.ubj` plus `xgboost_model.json` (features, params, study, splits).
- **`tuning.py`**
  - Multi-objective Optuna (log loss, macro F1), with walk-forward CV on a 30% symbol sample; `min_child_weight` is scaled by that fraction inside the objective.
  - The top Pareto trials are refit on all symbols and picked by validation macro F1.
  - The current study is `xgboost-2010-open-to-open`. The older `xgboost-tuning` study in the same DB used a different dataset.
- **`evaluation.py`**
  - The ranking metrics are per-date rank IC of `P(UP) − P(DOWN)`: overall, within volatility terciles ("volatility-neutral"), by liquidity quintile, and with / without news. Also decile returns (simple returns) and calibration error.
  - t-stats use every 8th date, because label windows overlap.
  - **Judge new features on volatility-neutral and liquid-stock IC, not macro F1.** The price model's score is about −0.8 rank-correlated with volatility, and most of its edge is "volatile, illiquid stocks fall".
- **`experiments.py`**
  - Runs A (prices), B (+ headline news), C (+ summary news), `B_placebo` (news shuffled between stocks within each date) and `B_same_day`.
  - Every run uses the same rows, splits and tuned params (`load_tuned_params`), with 3 seeds each.
  - Each run saves `experiments/<name>/seed<k>.{json,npy}` and is skipped when rerun with the same config. `summarize` writes `summary.json`.

### News

- **`news.py`**
  - Reads the CSV single-threaded, because article bodies are multi-line quoted fields.
  - Keeps symbol, `published_at` (UTC), title, url and `lsa_summary`.
  - Removes duplicates (same symbol, same title ignoring case, same UTC day).
  - Drops symbol-days with more than 200 articles (`MAX_ARTICLES_PER_SYMBOL_DAY`). These come from a generic feed scraped under ~50 tickers in Dec 2023.
- **`sentiment.py`**
  - Scores each distinct headline and summary with `ProsusAI/finbert` (fp16): `p_positive`, `p_negative`, `p_neutral`, `sentiment = p_positive − p_negative`.
  - Writes resumable chunks to `<cache>_chunks/` while scoring.
- **`news_features.py`**
  - One row per price row from 2009, built in DuckDB. Windows (1 day, 7 days, recency-weighted with a 3-day half-life) count trading days of the market calendar.
  - Counts are 0 and sentiment is NaN when a window has no articles. `load_fusion_dataset` left-joins the features onto the price dataset, keeping the same rows in the same order.
- **Timing rule.** Most timestamps are dates only, so news timing goes by the article's UTC date D.
  - By default, articles count from the **first trading day after D**.
  - `same_day=True` counts them on D. That is **also safe for these labels**: D ends at 20:00 New York time, before the next open where trades enter. It is one day fresher, and made no difference in the experiments. It would be lookahead for a trade entered at D's close.
- **Coverage.**
  - ~39% of rows have a headline in the last 7 days.
  - Summaries exist for only 10–40% of articles before 2021 and all of them after: 15% of train rows against 39% of test rows. Keep summary features NaN where missing, with no has-summary flag.
  - ~5% of articles are roundups tagged to 50+ stocks.

## News service

- **Flow.** `news_service/__main__.py` creates live and historical topics (2 partitions, replication factor 1), backfills with `produce_historical_news`, then streams live news with `news_stream`.
- **Messages.** `utils/models.py:NewsMessage` (pydantic); `utils/text_utils.py` strips HTML and markdown.
- **Config.** `configs/config.yaml` holds Kafka (`localhost:19092`) and the topic names; `configs/stocks.yaml` the symbols. They are read by relative path, so run the service from the repo root.
- **Known bugs:**
  - `producer.py` sends only the third article of each response, and fails with fewer than 3.
  - `timestamp_ms` is the run time, not the article's `created_at`.
  - `NewsMessage` has no `symbols` field.
