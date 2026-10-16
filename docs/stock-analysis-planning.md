# Stock Analysis Planning

## 1. Objective

Build a stock-analysis system that combines historical **OHLCV** (open, high, low, close, volume) data with financial news to produce evidence-backed **BUY / HOLD / SELL** signals. Separate numerical forecasting, decision policy, risk management, and natural-language explanations so each can be evaluated independently.

**Core recommendation:** Start with OHLCV features + news features in LightGBM, compare against an OHLCV-only XGBoost baseline, and only then add a two-stage decision model and LLM agent.

> This is a research and paper-trading architecture, not a validated investment strategy. Predicted returns and model confidence do not guarantee profitable trades.

## 2. System Architecture

```text
                 Historical / live data
                         |
            +------------+------------+
            |                         |
       OHLCV prices                Financial news
            |                         |
     Feature engineering      Headline / article encoder
     (momentum, RSI,          (FinBERT, embeddings,
      volatility, volume)      or small instruct LLM)
            |                         |
      XGBoost baseline        Time-aware aggregation
      (future returns)         (sentiment, topics,
            |                   event types, counts)
            +------------+------------+
                         |
                Feature fusion table
                         |
              LightGBM decision model
              (net return / trade value)
                         |
             BUY / HOLD / SELL policy
                         |
                Deterministic risk layer
                (limits, sizing, costs)
                         |
                 LLM analysis agent
              (explanations + citations)
                         |
                Backtest / paper trade
```

**Simpler alternative:** Train a single LightGBM model directly on engineered OHLCV and aggregated news features, without stacking XGBoost predictions. This is the recommended first fusion experiment.

## 3. Quantitative Model: XGBoost

### Why XGBoost?

XGBoost is a strong baseline for tabular, engineered financial features. It learns nonlinear feature interactions and supports model interpretation through SHAP. Other candidates:

| Model | Best reason to test | Notes |
|---|---|---|
| XGBoost | Strong tabular baseline | Interpretable via SHAP |
| LightGBM | Efficient large tabular datasets | Good candidate for feature fusion |
| CatBoost | Categorical ticker/sector features | Good alternative |
| Random forest / linear model | Simple baselines | Helps measure genuine improvement |
| TCN / LSTM | Learn from raw historical sequences | More data and tuning needed |
| Transformer / TFT | Large multivariate temporal datasets | Complexity may not pay off |

### Predict returns, not absolute prices

For a five-day horizon:

\[
R_{t,t+5} = \frac{P_{t+5}}{P_t} - 1
\]

Prefer a return target, direction probability, or cross-sectional ranking over predicting an exact future price. Define prices consistently with the execution policy: if the decision occurs after market close and execution at the next open, evaluate returns from realistically executable entry and exit prices.

Example OHLCV features:

- 1-, 5-, and 20-day returns
- RSI, MACD, moving-average gaps
- 20-day realized volatility
- Volume z-score and relative volume
- Sector and broad-market returns
- Market regime or VIX features

Potential model outputs: predicted 1-, 5-, and 20-day returns; probability of a positive return; cross-sectional percentile ranking. These are estimates requiring calibration and out-of-sample testing.

## 4. Financial News Dataset: FNSPID

Dataset: https://huggingface.co/datasets/Zihan1004/FNSPID

FNSPID combines historical stock-price and financial-news information. Relevant news fields include publication date, article title, stock symbol, publisher, URL, and sometimes article body. Inspect the exact downloaded schema and missing values before implementation; some article bodies may be unavailable.

For each ticker and prediction timestamp:

1. Collect articles available **before** the prediction cutoff.
2. Associate each article with the correct affected ticker(s).
3. Deduplicate syndicated or repeated headlines.
4. Encode each headline/article into sentiment, embeddings, or event labels.
5. Aggregate news features over windows such as 1, 3, and 7 days.
6. Join the aggregated features to OHLCV rows using point-in-time timestamps.

The historical dataset does not replace a live news/price feed for production inference.

## 5. News Representation Options

| Approach | Output | Strength | Limitation |
|---|---|---|---|
| FinBERT | Positive / neutral / negative probabilities | Finance-specific, interpretable sentiment baseline | Limited semantic detail |
| Small embedding model | Dense semantic vector | Captures topic, event, and wording differences | Not inherently a sentiment score |
| Small instruct LLM | Structured event and sentiment labels | Flexible, interpretable extraction | Slower and harder to validate |
| Hybrid | Sentiment + embeddings / event labels | Potential complementary information | More complexity |

### 5.1 FinBERT sentiment baseline

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    top_k=None,
)

def score_headline(headline: str):
    output = classifier(headline)[0]
    probs = {item["label"].lower(): item["score"] for item in output}
    return {
        "sentiment": probs["positive"] - probs["negative"],
        "positive_prob": probs["positive"],
        "negative_prob": probs["negative"],
    }
```

Aggregate features by ticker and cutoff:

- `sentiment_mean_1d`, `sentiment_mean_7d`
- `sentiment_std_7d`
- `positive_news_ratio_1d`
- `negative_news_count_1d`
- `news_count_1d`
- `sentiment_change_3d`

Optional recency-weighted sentiment:

\[
S_t = \frac{\sum_i e^{-\lambda(t-t_i)}s_i}{\sum_i e^{-\lambda(t-t_i)}}
\]

where each `t_i` is an article publication time available by the decision cutoff.

### 5.2 Embedding models (recommended experiment)

Candidate models:

- `BAAI/bge-small-en-v1.5`: compact 384-dimensional English embeddings.
- `sentence-transformers/all-MiniLM-L6-v2`: fast 384-dimensional baseline.
- `Alibaba-NLP/gte-base-en-v1.5`: larger 768-dimensional alternative.

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
headlines = [
    "Apple reports record quarterly revenue",
    "Apple faces regulatory investigation",
    "Apple announces new AI features",
]
embeddings = encoder.encode(headlines, normalize_embeddings=True)
print(embeddings.shape)  # (3, 384)
```

Aggregate article embeddings per ticker and time window (simple or recency-weighted mean), and retain news count and a missing-news indicator. To reduce feature dimensionality, fit PCA **only on training-period data**:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=32)
train_reduced = pca.fit_transform(train_embeddings)
test_reduced = pca.transform(test_embeddings)
```

PCA should be fit on the representation used in the training pipeline, with transformations applied unchanged to validation and test periods. Do not fit PCA on all historical data before splitting.

### 5.3 Small instruct LLM for structured events

Instead of a numeric embedding, extract structured fields such as:

```json
{
  "ticker": "AAPL",
  "event_type": "earnings",
  "sentiment": "positive",
  "magnitude": "high",
  "uncertainty": "medium",
  "time_horizon": "short_term"
}
```

Event categories can include earnings, guidance, regulation, product launch, litigation, layoffs, acquisitions, and macroeconomic news. Encode labels as categorical/numeric features. Treat the LLM's impact judgment as a hypothesis, not a forecast validated by market data.

## 6. Joining OHLCV and News

Create **one row per ticker per decision timestamp**. Example:

| Ticker | Cutoff | RSI | 5d momentum | 1d sentiment | News count | Target future return |
|---|---|---:|---:|---:|---:|---:|
| AAPL | Mon 16:00 ET | 62 | +1.5% | +0.72 | 14 | +2.1% |
| MSFT | Mon 16:00 ET | 44 | -0.8% | -0.61 | 11 | -1.3% |

*Illustrative values only. Target future return must never be an input feature.*

**Leakage prevention:** A headline published at 4:30 PM ET cannot influence a decision made at 4:00 PM ET. Respect publication times, availability delays, market holidays, corporate actions, and the exact trade entry time. If publication time is unknown, adopt a conservative lag rather than assuming it was available intraday.

## 7. Fusion and Decision Models

### Stage A: Single fusion model (start here)

Train LightGBM on OHLCV + aggregated sentiment/embedding features to predict forward returns or the probability of profitable trades after costs.

```python
price_features = [
    "return_1d", "return_5d", "rsi", "macd",
    "volatility_20d", "volume_zscore",
]
news_features = [
    "sentiment_mean_1d", "sentiment_mean_7d",
    "sentiment_std_7d", "positive_news_ratio_1d",
    "news_count_1d", "sentiment_change_3d",
]
feature_columns = price_features + news_features
# Add news_embedding_0 ... news_embedding_31 for embedding experiments.
```

### Stage B: Two-stage model (only if justified)

1. XGBoost forecasts returns using price-based features.
2. LightGBM consumes **out-of-fold chronological XGBoost predictions** plus news features and market/portfolio context.
3. A policy maps expected net returns and risks to BUY/HOLD/SELL.

Do not train the second-stage model on in-sample predictions from the first stage: this creates an unrealistically strong input and contaminates validation.

### Example decision policy

```python
def decide(expected_return, buy_threshold, sell_threshold, has_position):
    if expected_return > buy_threshold:
        return "BUY"
    if expected_return < sell_threshold and has_position:
        return "SELL"
    return "HOLD"
```

This is only a baseline: real thresholds should account for execution costs, risk, uncertainty, position sizes, and expected holding period. A negative forecast without an existing position may mean HOLD/AVOID rather than SELL/SHORT.

### System 1 decision model and fine-tuning

A pretrained decision-oriented model can be prompted with structured price, news, and portfolio signals. **Fine-tuning is not necessary for the initial prototype.** Compare it against deterministic rules and a tabular classifier first. Fine-tuning or reinforcement learning should be considered only if controlled evaluation demonstrates a clear benefit. “System 1” is not one standardized model/API; the exact model's supported input and training capabilities must be verified.

## 8. Agent Responsibilities

The LLM analysis agent should:

- Retrieve relevant articles and cite their publication dates/URLs.
- Explain the quantitative prediction and news features using model-grounded explanations such as SHAP.
- Describe uncertainty and competing evidence.
- Report the proposed action and the risk policy's final action.
- Avoid inventing market data, article claims, SHAP contributions, or calibrated probabilities.

Example **illustrative** output:

```json
{
  "ticker": "AAPL",
  "forecast_horizon": "5d",
  "expected_return": 0.023,
  "sentiment_7d": 0.64,
  "proposed_action": "BUY",
  "risk_approved_action": "BUY",
  "explanation": "Positive news sentiment and momentum support the forecast, but volatility raises downside risk.",
  "sources": ["<actual article URL>"]
}
```

The LLM explains a recommendation; it should not bypass deterministic portfolio constraints or execute trades solely from a natural-language judgment.

## 9. Training, Backtesting, and Evaluation

Use chronological training/validation/test periods and walk-forward retraining. For example:

```text
Train 2015–2020 → validate/test 2021
Train 2015–2021 → validate/test 2022
Train 2015–2022 → validate/test 2023
```

Add appropriate purge/embargo gaps where forward-return labels overlap. Refit encoders, PCA, scalers, and learned models within each training fold as applicable. Beware of survivorship bias, stock splits/dividends, changing ticker universes, duplicate articles, and missing-news bias.

Compare:

| Experiment | Inputs | Question |
|---|---|---|
| A | OHLCV only (XGBoost) | Quantitative baseline |
| B | OHLCV + FinBERT | Does sentiment help? |
| C | OHLCV + 32-d news embeddings | Does semantic information help? |
| D | OHLCV + small LLM event labels | Do structured events help? |
| E | OHLCV + sentiment + embeddings | Are the signals complementary? |
| F | Two-stage XGBoost → LightGBM | Does stacking add value? |

Evaluate both prediction and actual portfolio policy:

- Predictive: MAE/RMSE for returns, directional accuracy, calibration, ranking metrics.
- Trading: net returns after fees/slippage, Sharpe ratio, maximum drawdown, turnover, exposure, and performance by market regime.
- Baselines: buy-and-hold, a simple threshold policy, and the OHLCV-only model.

A model with better prediction metrics may still have worse trading performance after costs. Avoid optimizing thresholds on the final test set.

## 10. Practical Build Order

1. **Dataset pipeline:** Download/validate FNSPID, align prices and news by ticker/time, establish a leakage-safe cutoff.
2. **OHLCV baseline:** Engineer technical features and train XGBoost for forward returns.
3. **FinBERT baseline:** Score headlines, aggregate by time window, train a single fusion LightGBM model.
4. **Embeddings experiment:** Encode headlines with BGE-small, aggregate, reduce to ~32 dimensions, compare against sentiment.
5. **Optional event extraction:** Try a small instruct LLM and measure incremental value.
6. **Decision policy:** Convert net expected return/risk into BUY/HOLD/SELL with portfolio-aware rules.
7. **Agent:** Add sourced news summaries and model-grounded explanations.
8. **Validation:** Walk-forward backtests, realistic costs, then paper trading before any consideration of live execution.

## 11. References

- FNSPID dataset: https://huggingface.co/datasets/Zihan1004/FNSPID
- FinBERT: https://huggingface.co/ProsusAI/finbert
- BGE-small: https://huggingface.co/BAAI/bge-small-en-v1.5
- MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- GTE-base: https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5
- XGBoost: https://xgboost.readthedocs.io/
- LightGBM: https://lightgbm.readthedocs.io/

---

**Bottom line:** Prove that news improves out-of-sample trading results before adding a complex decision agent. A good first implementation is **OHLCV + BGE-small embeddings → LightGBM → rule-based risk policy → LLM explanation**, with XGBoost and FinBERT retained as comparison baselines.
