from forecasting.data import load_or_build_metrics_dataset

def compute_model_features():
    """
    Compute ratios for any metric / value is more of a raw value (i.e stock high, low, close prices, EMA, SMA, etc.)
    """

    stock_metrics_dataset = load_or_build_metrics_dataset()