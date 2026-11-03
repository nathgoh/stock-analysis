from enum import StrEnum


class StockRating(StrEnum):
    DOWN = "DOWN"
    HOLD = "HOLD"
    UP = "UP"

    @classmethod
    def to_index_map(cls) -> dict[str, int]:
        return {r.value: i for i, r in enumerate(cls)}


class IndicatorName(StrEnum):
    RETURN_1D = "return_1d"
    RETURN_5D = "return_5d"
    RETURN_14D = "return_14d"
    SMA_20 = "sma_20"  # Simple Moving Average
    SMA_50 = "sma_50"
    SMA_200 = "sma_200"
    SMA_50_200_SPREAD = "sma_50_200_spread"
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    EMA_10 = "ema_10"  # Exponential Moving Average
    EMA_20 = "ema_20"
    EMA_50 = "ema_50"
    VOLATILITY_14D = "volatility_14d"
    VOLATILITY_21D = "volatility_21d"
    VOLATILITY_60D = "volatility_60d"
    MOMENTUM_7D = "momentum_7d"
    MOMENTUM_14D = "momentum_14d"
    RSI_14D = "rsi_14d"  # Relative Strength Index
    MACD = "macd"  # Moving Average Convergence Divergence
    MACD_SIGNAL = "macd_signal"
    MACD_HISTOGRAM = "macd_histogram"
    DMI = "dmi"  # Directional Movement Index
    ADX = "adx"  # Average Directional Index
    INTRADAY_RANGE = "intraday_range"
    GAP = "gap"
    RELATIVE_VOLUME_21D = "relative_volume_21d"


class MarketIndicatorName(StrEnum):
    """
    Cross-sectional indicators, comparing a stock against every symbol on the same date.
    """

    MARKET_RETURN_1D = "market_return_1d"
    MARKET_RETURN_5D = "market_return_5d"
    EXCESS_RETURN_1D = "excess_return_1d"
    EXCESS_RETURN_5D = "excess_return_5d"
    MOMENTUM_14D_RANK = "momentum_14d_rank"
    RELATIVE_VOLUME_21D_RANK = "relative_volume_21d_rank"


class NewsFeatureName(StrEnum):
    """
    Per stock and trading day, from the FinBERT scores of the headlines. Windows count
    trading days, an article dated D counts from the next trading day after D.
    """

    NEWS_COUNT_1D = "news_count_1d"
    NEWS_COUNT_7D = "news_count_7d"
    SENTIMENT_MEAN_1D = "sentiment_mean_1d"
    SENTIMENT_MEAN_7D = "sentiment_mean_7d"
    SENTIMENT_STD_7D = "sentiment_std_7d"
    POSITIVE_RATIO_1D = "positive_ratio_1d"
    NEGATIVE_COUNT_1D = "negative_count_1d"
    SENTIMENT_CHANGE_3D = "sentiment_change_3d"
    SENTIMENT_EWM = "sentiment_ewm"
    DAYS_SINCE_LAST_NEWS = "days_since_last_news"


class SummaryNewsFeatureName(StrEnum):
    """
    Like `NewsFeatureName`, from the FinBERT scores of the article summaries, which only
    exist for some articles before 2021.
    """

    SUMMARY_COUNT_7D = "summary_count_7d"
    SUMMARY_SENTIMENT_MEAN_1D = "summary_sentiment_mean_1d"
    SUMMARY_SENTIMENT_MEAN_7D = "summary_sentiment_mean_7d"
    SUMMARY_SENTIMENT_EWM = "summary_sentiment_ewm"
