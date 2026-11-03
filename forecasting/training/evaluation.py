import numpy as np
import pandas as pd

from forecasting.data import LABEL_HORIZON
from forecasting.models import IndicatorName, NewsFeatureName, StockRating

# Dates (or date segments) with fewer rows than this are skipped by the per-date metrics.
MIN_ROWS_PER_DATE = 50


def independent_dates(dates) -> np.ndarray:
    """
    Every (LABEL_HORIZON + 1)-th date. The label windows of nearby dates overlap, so
    t-stats only use these dates.
    """

    return np.sort(pd.unique(np.asarray(dates)))[:: LABEL_HORIZON + 1]


def t_stat(daily: pd.Series) -> float:
    """
    t-stat of the mean of a per-date series, on non-overlapping dates.
    """

    values = daily[daily.index.isin(independent_dates(daily.index))].dropna()
    return float(values.mean() / (values.std() / np.sqrt(len(values))))


def _rank_ic(
    frame: pd.DataFrame, score: str, target: str, keys: list[str], min_rows: int
) -> pd.Series:
    """
    Spearman correlation between `score` and `target` per `keys` group, leaving out
    groups with fewer than `min_rows` rows.
    """

    groups = frame.groupby(keys, observed=True, sort=False)
    # Spearman correlation is the Pearson correlation of the (average) ranks.
    ranks = frame[keys].assign(x=groups[score].rank(), y=groups[target].rank())
    ranks = ranks.assign(
        xx=ranks["x"] ** 2, yy=ranks["y"] ** 2, xy=ranks["x"] * ranks["y"]
    )
    sums = ranks.groupby(keys, observed=True).agg(
        n=("x", "size"),
        x=("x", "sum"),
        y=("y", "sum"),
        xx=("xx", "sum"),
        yy=("yy", "sum"),
        xy=("xy", "sum"),
    )
    sums = sums[sums["n"] >= min_rows]

    n = sums["n"]
    cov = sums["xy"] - sums["x"] * sums["y"] / n
    var_x = sums["xx"] - sums["x"] ** 2 / n
    var_y = sums["yy"] - sums["y"] ** 2 / n
    return (cov / np.sqrt(var_x * var_y)).sort_index()


def daily_rank_ic(
    frame: pd.DataFrame, score: str, target: str, min_rows: int = MIN_ROWS_PER_DATE
) -> pd.Series:
    """
    Spearman correlation between `score` and `target` on each date.
    """

    return _rank_ic(frame, score, target, ["date"], min_rows)


def daily_rank_ic_by(
    frame: pd.DataFrame,
    score: str,
    target: str,
    by: str,
    min_rows: int = MIN_ROWS_PER_DATE,
) -> pd.DataFrame:
    """
    Spearman correlation between `score` and `target` on each date, within each `by`
    segment (one column per segment).
    """

    return _rank_ic(frame, score, target, ["date", by], min_rows).unstack(by)


def _bucket(values: pd.Series, dates: pd.Series, n_buckets: int) -> pd.Series:
    """
    Per-date quantile bucket of `values`, 1 (lowest) to `n_buckets`.
    """

    pct = values.groupby(dates).rank(pct=True)
    return np.ceil(pct * n_buckets).clip(1, n_buckets).astype("Int8")


def ranking_frame(test_df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """
    What the ranking metrics need per test row: the model's score
    (P(UP) - P(DOWN)), the forward return, and the segments the metrics are split by.
    """

    index = StockRating.to_index_map()
    dates = test_df["date"]
    frame = pd.DataFrame(
        {
            "date": dates,
            "score": proba[:, index[StockRating.UP]]
            - proba[:, index[StockRating.DOWN]],
            "forward_return": test_df["forward_return"],
            # Decile returns are averaged as simple returns, log returns are lower for
            # volatile stocks by construction (by about half the variance).
            "simple_return": np.expm1(test_df["forward_return"]),
            "volatility_tercile": _bucket(
                test_df[IndicatorName.VOLATILITY_21D], dates, 3
            ),
            "liquidity_quintile": _bucket(
                test_df["close"] * test_df["volume"], dates, 5
            ),
        }
    )
    if NewsFeatureName.NEWS_COUNT_7D in test_df:
        frame["has_news"] = test_df[NewsFeatureName.NEWS_COUNT_7D] > 0

    return frame


def ranking_metrics(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    How well the score ranks stocks against each other on each date, from a
    `ranking_frame`. Returns the summary metrics, and the per-date series used to
    compare models.

    - IC: per-date rank correlation of score and forward return, overall, within each
      volatility tercile and liquidity quintile, and on rows with / without news.
    - Volatility-neutral IC: the mean of the within-tercile ICs, so ranking stocks by
      how volatile they are doesn't count.
    - Deciles: mean 7-day simple return of the top / bottom score decile, in bps.
    """

    ic = daily_rank_ic(frame, "score", "forward_return")
    by_volatility = daily_rank_ic_by(
        frame, "score", "forward_return", "volatility_tercile"
    )
    by_liquidity = daily_rank_ic_by(
        frame, "score", "forward_return", "liquidity_quintile"
    )
    volatility_neutral = by_volatility.mean(axis=1)

    pct = frame.groupby("date")["score"].rank(pct=True)
    returns = frame.groupby("date")["simple_return"]
    top = frame.loc[pct > 0.9].groupby("date")["simple_return"].mean()
    bottom = frame.loc[pct <= 0.1].groupby("date")["simple_return"].mean()
    spread = top - bottom

    daily = pd.DataFrame(
        {
            "ic": ic,
            "volatility_neutral_ic": volatility_neutral,
            "most_liquid_ic": by_liquidity[5],
            "spread_bps": spread * 1e4,
        }
    )
    metrics = {
        "ic": ic.mean(),
        "ic_t": t_stat(ic),
        "ic_positive_days": (ic > 0).mean(),
        "volatility_neutral_ic": volatility_neutral.mean(),
        "volatility_neutral_ic_t": t_stat(volatility_neutral),
        **{f"ic_volatility_{n}": by_volatility[n].mean() for n in by_volatility},
        **{f"ic_liquidity_{n}": by_liquidity[n].mean() for n in by_liquidity},
        "top_decile_bps": top.mean() * 1e4,
        "bottom_decile_bps": bottom.mean() * 1e4,
        "spread_bps": spread.mean() * 1e4,
        "spread_t": t_stat(spread),
        "top_minus_mean_bps": (top - returns.mean()).mean() * 1e4,
    }
    if "has_news" in frame:
        by_news = daily_rank_ic_by(frame, "score", "forward_return", "has_news")
        daily["news_ic"] = by_news[True]
        metrics["ic_news"] = by_news[True].mean()
        metrics["ic_no_news"] = by_news[False].mean()

    return {k: float(v) for k, v in metrics.items()}, daily


def calibration_errors(
    proba: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """
    Expected calibration error per class: the mean gap between predicted probability
    and observed frequency, over `n_bins` quantile bins of the predicted probability.
    """

    errors = {}
    for rating, i in StockRating.to_index_map().items():
        p = pd.Series(proba[:, i])
        bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
        observed = pd.Series(y == i).groupby(bins).mean()
        predicted = p.groupby(bins).mean()
        weight = p.groupby(bins).size() / len(p)
        errors[rating] = float(((predicted - observed).abs() * weight).sum())

    return errors
