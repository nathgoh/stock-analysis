from dataclasses import dataclass
from enum import StrEnum
from typing import Callable

import pandas as pd


class IndicatorName(StrEnum):
    RETURN_1D = "return_1d"
    RETURN_5D = "return_5d"
    RETURN_14D = "return_14d"
    SMA_20 = "sma_20"  # Simple Moving Average
    SMA_50 = "sma_50"
    SMA_200 = "sma_200"
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


@dataclass(frozen=True)
class IndicatorSpec:
    name: IndicatorName
    description: str
    func: Callable[[pd.DataFrame], pd.Series]


def _sma(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the simple moving average (SMA) of closing prices over a specified window.
    """

    return df["close"].rolling(window=window).mean()


def _ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the exponential moving average (EMA) of closing prices over a specified window.
    """

    return df["close"].ewm(span=window).mean()


def _rsi(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) over a specified rolling window.
    """

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss

    return 100 - (100 / (1 + rs))


def _macd(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the Moving Average Convergence Divergence (MACD) line.
    """

    ema_12 = _ema(df, window=12)
    ema_26 = _ema(df, window=26)

    return ema_12 - ema_26


def _macd_signal(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the MACD signal line using a 9-day EMA of the MACD.
    """

    ema_12 = _ema(df, window=12)
    ema_26 = _ema(df, window=26)

    return _ema(ema_12 - ema_26, window=9)


def _macd_histogram(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the MACD histogram as the difference between the MACD and its signal line.
    """

    macd = _macd(df)
    macd_signal = _macd_signal(df)

    return macd - macd_signal


def _di(df: pd.DataFrame, wilder: int = 14) -> tuple[pd.Series, pd.Series]:
    """
    Calculate positive (+DI) and negative (-DI) directional indicators using Wilder's smoothing.
    """

    high = df["high"]
    low = df["low"]
    close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    true_range = pd.concat(
        [high - low, high - close.shift(1), low - close.shift(1)], axis=1
    ).max(axis=1)

    # Wilder smoothing, smoothing factor of 1 / wilder
    wilder = 14
    avg_true_range = true_range.ewm(alpha=wilder, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / wilder, adjust=False).mean() / avg_true_range
    minus_di = (
        100 * minus_dm.ewm(alpha=1 / wilder, adjust=False).mean() / avg_true_range
    )

    return plus_di, minus_di


def _dmi(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the Directional Movement Index (DMI) based on positive and negative directional indicators.
    """

    plus_di, minus_di = _di(df)

    return 100 * abs(plus_di - minus_di) / abs(plus_di + minus_di)


def _adx(df: pd.DataFrame, wilder: int = 14) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX) by smoothing the DMI.
    """

    dx = _dmi(df)

    return dx.ewm(alpha=1 / wilder, adjust=False).mean()


INDICATORS: dict[IndicatorName, IndicatorSpec] = {
    IndicatorName.RETURN_1D: IndicatorSpec(
        name=IndicatorName.RETURN_1D,
        description="1-day return",
        func=lambda df: df["close"].pct_change(),
    ),
    IndicatorName.RETURN_5D: IndicatorSpec(
        name=IndicatorName.RETURN_5D,
        description="5-day return",
        func=lambda df: df["close"].pct_change(periods=5),
    ),
    IndicatorName.RETURN_14D: IndicatorSpec(
        name=IndicatorName.RETURN_14D,
        description="14-day return",
        func=lambda df: df["close"].pct_change(periods=14),
    ),
    IndicatorName.SMA_20: IndicatorSpec(
        name=IndicatorName.SMA_20,
        description="20-day simple moving average",
        func=lambda df: _sma(df, window=20),
    ),
    IndicatorName.SMA_50: IndicatorSpec(
        name=IndicatorName.SMA_50,
        description="50-day simple moving average",
        func=lambda df: _sma(df, window=50),
    ),
    IndicatorName.SMA_200: IndicatorSpec(
        name=IndicatorName.SMA_200,
        description="200-day simple moving average",
        func=lambda df: _sma(df, window=200),
    ),
    IndicatorName.GOLDEN_CROSS: IndicatorSpec(
        name=IndicatorName.GOLDEN_CROSS,
        description="Golden cross (50-day SMA crosses above 200-day SMA), bullish signal",
        func=lambda df: (
            (_sma(df, 50) > _sma(df, 200))
            & (_sma(df, 50).shift(1) <= _sma(df, 200).shift(1))
        ),
    ),
    IndicatorName.DEATH_CROSS: IndicatorSpec(
        name=IndicatorName.DEATH_CROSS,
        description="Death cross (50-day SMA crosses below 200-day SMA), bearish signal",
        func=lambda df: (
            (_sma(df, 50) < _sma(df, 200))
            & (_sma(df, 50).shift(1) >= _sma(df, 200).shift(1))
        ),
    ),
    IndicatorName.EMA_10: IndicatorSpec(
        name=IndicatorName.EMA_10,
        description="10-day exponential moving average",
        func=lambda df: _ema(df, window=10),
    ),
    IndicatorName.EMA_20: IndicatorSpec(
        name=IndicatorName.EMA_20,
        description="20-day exponential moving average",
        func=lambda df: _ema(df, window=20),
    ),
    IndicatorName.EMA_50: IndicatorSpec(
        name=IndicatorName.EMA_50,
        description="50-day exponential moving average",
        func=lambda df: _ema(df, window=50),
    ),
    IndicatorName.VOLATILITY_14D: IndicatorSpec(
        name=IndicatorName.VOLATILITY_14D,
        description="14-day rolling volatility, std of returns",
        func=lambda df: df["close"].pct_change().rolling(window=14).std(),
    ),
    IndicatorName.VOLATILITY_21D: IndicatorSpec(
        name=IndicatorName.VOLATILITY_21D,
        description="21-day rolling volatility, std of returns",
        func=lambda df: df["close"].pct_change().rolling(window=21).std(),
    ),
    IndicatorName.VOLATILITY_60D: IndicatorSpec(
        name=IndicatorName.VOLATILITY_60D,
        description="60-day rolling volatility, std of returns",
        func=lambda df: df["close"].pct_change().rolling(window=60).std(),
    ),
    IndicatorName.MOMENTUM_7D: IndicatorSpec(
        name=IndicatorName.MOMENTUM_7D,
        description="7-day momentum, price change over 7 days",
        func=lambda df: df["close"] / df["close"].shift(7) - 1,
    ),
    IndicatorName.MOMENTUM_14D: IndicatorSpec(
        name=IndicatorName.MOMENTUM_14D,
        description="14-day momentum, price change over 14 days",
        func=lambda df: df["close"] / df["close"].shift(14) - 1,
    ),
    IndicatorName.RSI_14D: IndicatorSpec(
        name=IndicatorName.RSI_14D,
        description="14-day Relative Strength Index",
        func=lambda df: _rsi(df, window=14),
    ),
    IndicatorName.MACD: IndicatorSpec(
        name=IndicatorName.MACD,
        description="MACD indicator",
        func=lambda df: _macd(df),
    ),
    IndicatorName.MACD_SIGNAL: IndicatorSpec(
        name=IndicatorName.MACD_SIGNAL,
        description="MACD signal line",
        func=lambda df: _macd_signal(df),
    ),
    IndicatorName.MACD_HISTOGRAM: IndicatorSpec(
        name=IndicatorName.MACD_HISTOGRAM,
        description="MACD histogram",
        func=lambda df: _macd_histogram(df),
    ),
    IndicatorName.DMI: IndicatorSpec(
        name=IndicatorName.DMI,
        description="Directional Movement Index",
        func=lambda df: _dmi(df),
    ),
    IndicatorName.ADX: IndicatorSpec(
        name=IndicatorName.ADX,
        description="Average Directional Index",
        func=lambda df: _adx(df),
    ),
}


def add_stock_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all the stock indicators listed in the IndicatorName enum.
    """

    df = df.copy()

    for name, spec in INDICATORS.items():
        df[name] = spec.func(df)

    return df
