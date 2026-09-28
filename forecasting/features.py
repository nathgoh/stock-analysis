import pandas as pd

from forecasting.data import load_or_build_metrics_dataset

# Raw price levels differ by orders of magnitude between stocks,
# so they are inputs to the indicators rather than features themselves.
NON_FEATURE_COLUMNS = [
    "symbol",
    "date",
    "volume",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "adj_open",
    "adj_high",
    "adj_low",
    # Used to compute the labels, so excluded to prevent leakage.
    # forward_return looks ahead, so it can't be a feature.
    "forward_return",
    "threshold",
    "label",
    # label_end_date and trailing_return only feed the evaluation baselines.
    "label_end_date",
    "trailing_return",
]


def compute_model_features(
    dataset: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the metrics dataset into the model's features and its labels.
    """

    if dataset is None:
        dataset = load_or_build_metrics_dataset()

    features = dataset.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    labels = dataset["label"]

    return features, labels
