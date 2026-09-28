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
