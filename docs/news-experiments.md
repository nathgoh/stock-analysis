# News experiments: results

Does FNSPID financial news improve the 7-day UP / HOLD / DOWN model? **No, not for this label.** News is priced in by the next open, where the label's trade enters, and what's left over the following week is too small to help.

## Setup

- **Data:** FNSPID prices and news. Training and evaluation start at `START_DATE` (2010-01-01).
- **Splits (chronological, with 14-day gaps):**
  - train: 2010-01 → 2020-01 (11.9M rows)
  - val: 2020-01 → 2021-10 (2.1M rows)
  - test: 2021-11 → 2023-12 (2.3M rows, 4,455 stocks)
- **Label:** decided after the close of day t. The trade enters at the next open and exits at the open 7 trading days later. UP / DOWN beyond ±0.5 × volatility × √7, otherwise HOLD.
- **News:** 5.2M cleaned articles, each headline (and article summary, where there is one) scored with FinBERT. They're aggregated per stock and trading day into 10 headline features and 4 summary features (`NewsFeatureName`, `SummaryNewsFeatureName`).
- **Model:** XGBoost with the tuned parameters of the `xgboost-2010-open-to-open` study, the same for every run, with 3 seeds each.
- **Metrics:** per-date rank correlation ("IC") of the score `P(UP) − P(DOWN)` with the forward return. This is reported overall, within volatility terciles ("vol-neutral") and by liquidity, alongside macro F1 and log loss.
- **Reproduce:**
  - `uv run --group forecasting python -m forecasting.training.experiments` (resumable, ~11 min on GPU)
  - results go to `data/artifacts/experiments/`

## 1. The price-only baseline (experiment A)

The model beats the naive baselines. Macro F1 is 0.378, against 0.346 for persistence and 0.347 for trailing-return momentum. Log loss is 1.0869, against 1.0931 for the class prior.

Its ranking edge, though, is mostly a volatility effect:
- **The score is mostly a volatility ranking:** its per-date rank correlation with `volatility_21d` is −0.81. Volatile stocks have lower 7-day log returns by construction, and in this test period (which includes 2022) they also fell more.
- **IC overall is 0.068. Within volatility terciles it's only 0.014 (low), 0.017 (mid) and 0.061 (high).**
- **It works best where trading is hardest:** IC is 0.11 in the least liquid fifth of stocks and 0.028 in the most liquid.
- Probabilities are not calibrated, because class weights push them toward 1/3 (mean ECE about 0.044).

So new features should be judged on vol-neutral and liquid-stock IC, not on macro F1.

## 2. Adding news (experiments B and C)

| | A prices | B + headlines | C + summaries | B placebo | B same-day |
|---|---:|---:|---:|---:|---:|
| macro F1 (±std over seeds) | 0.3779 ±0.0010 | 0.3782 ±0.0009 | 0.3779 ±0.0010 | 0.3789 ±0.0003 | 0.3783 ±0.0010 |
| log loss | 1.0872 | 1.0871 | 1.0870 | 1.0868 | 1.0870 |
| IC | 0.0680 | 0.0684 | 0.0686 | 0.0683 | 0.0681 |
| vol-neutral IC | 0.0305 | 0.0315 | 0.0319 | 0.0313 | 0.0313 |
| IC, most liquid fifth | 0.0281 | 0.0280 | 0.0276 | 0.0289 | 0.0276 |
| IC, rows with news in the last 7 days | 0.0520 | 0.0523 | 0.0524 | 0.0523 | 0.0519 |

The IC columns use the predictions averaged over the 3 seeds.

- **Change in vol-neutral IC against A:** B +0.0010 (t 0.8), C +0.0014 (t 1.4). Neither is significant, and the placebo gets +0.0008. The placebo shuffles headline news between stocks on the same date, so even that small gain isn't stock-specific.
- **Liquid stocks:** no change.
- **Feature usage:** the model barely uses the news. In B, news features account for 1.5% of the total gain.
- **News features on their own:** small rank correlations with the 7-day return. The best is recency-weighted sentiment, at 0.005 in train (t 3.0) and 0.013 in test (t 2.3).

## 3. When is news priced in?

Rank correlation of a stock-day's mean headline sentiment with the stock's market-adjusted return over windows around the article date. A is the last trading day on or before the article date.

| Window | Train IC (t) | Test IC (t) |
|---|---:|---:|
| day of the news (close A−1 → close A) | 0.102 (74) | 0.078 (29) |
| overnight (close A → open A+1) | 0.024 (20) | 0.017 (6.7) |
| next day (open A+1 → close A+1) | 0.005 (3.9) | 0.017 (6.5) |
| 7-day label of row A (open A+1 → open A+8) | 0.005 (0.7) | 0.011 (1.3) |
| 7-day label of row A+1 (open A+2 → open A+9) | 0.004 (−0.3) | 0.006 (1.0) |
| after the first day (close A+1 → open A+8) | 0.003 (−0.6) | 0.007 (0.8) |

The results within volatility terciles are nearly identical.

**Event tags.** Keyword rules on headlines give the same picture. Guidance cuts move a stock −177 bps on the day and −60 bps overnight; offerings −131 / −88; earnings misses −86 / −33; guidance raises +112 / +44. After the next open, no tag shows a reliable move: in the 7-day windows, t-stats are mostly under 2, and signs flip between train and test.

The tags are also unstable over time. Analyst headlines nearly vanish after 2020: price-target raises occur on 7,620 stock-days in train against 13 in test. The scratch analysis behind this section (event regexes, windows) was not kept in the repo.

## Conclusions

- **For a 7-day label entering at the next open, neither headline sentiment, summary sentiment nor event tags add anything over prices.** The reaction to news happens on the day and overnight, before the entry.
- **Same-day features are not lookahead for this label.** An article's UTC date ends at 20:00 New York time, before the next open. Phase 5 used the more conservative next-day rule by default; `B_same_day` shows the choice didn't matter here.
- **Where news could still help:** the first trading day after the news (next open → next close). There, the test IC is 0.017 (t 6.5), and buybacks and dividend raises add about 20 bps (t ~6.5 in train). A 1-day label pays far more in trading costs, though.

## Caveats

- **One test period:** Nov 2021 – Dec 2023, which includes the 2022 bear market.
- **Data quality:**
  - News coverage is uneven. About 39% of rows have a headline in the last 7 days. Summaries cover 15% of train rows and 39% of test rows.
  - About 5% of articles are roundups tagged to 50+ stocks.
  - A Dec 2023 flood of generic articles under ~50 tickers was removed (`MAX_ARTICLES_PER_SYMBOL_DAY`).
- **Model and features:**
  - The hyperparameters were tuned on prices only, and are conservative (≈50 trees, `min_child_weight` ≈ 4,500, `colsample_bytree` 0.44).
  - Re-tuning wasn't tried, since the news features show little standalone signal.
  - FinBERT scores headlines and summaries, not whether the article is about the tagged stock.
