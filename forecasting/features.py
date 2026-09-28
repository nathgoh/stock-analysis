import pandas as pd

from forecasting.data import NON_FEATURE_COLUMNS, load_or_build_metrics_dataset


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
