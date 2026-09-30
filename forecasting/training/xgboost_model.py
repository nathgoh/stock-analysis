from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import f1_score, log_loss

from forecasting.data import (
    build_split_datasets,
    class_weights,
    load_or_build_metrics_dataset,
)
from forecasting.features import compute_model_features
from forecasting.models import StockRating

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "data" / "artifacts" / "xgboost"

BASE_PARAMS = {
    "objective": "multi:softprob",
    "num_class": len(StockRating),
    "eval_metric": "mlogloss",
    "n_estimators": 2000,
    "device": "cuda",
    "random_state": 0,
}

def prepare_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features, labels = compute_model_features(df)

    return features.astype("float32"), labels.map(StockRating.to_index_map())


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    verbose: int = 0,
) -> xgb.XGBClassifier:
    weights = class_weights(y_train)

    model = xgb.XGBClassifier(**{**BASE_PARAMS, **params})
    model.fit(
        X_train,
        y_train,
        sample_weight=y_train.map(weights).to_numpy(),
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[y_val.map(weights).to_numpy()],
        verbose=verbose,
    )

    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    test_df: pd.DataFrame,
    y_train: pd.Series,
    class_scale: np.ndarray | None = None,
):
    """
    Score the model on the test set and compare against the naive baselines.
    """

    stock_ratings = StockRating.to_index_map()
    labels = list(stock_ratings.values())

    X_test, y_test = prepare_model_features(test_df)
    test_proba = predict_proba(model, X_test)

    # Benchmark against persistence baseline, which only scores dates where a prior
    # horizon window has been resolved. Filter all models to this subset for comparison.
    lagged_label = resolved_label_lag(test_df)
    has_lag = lagged_label.notna().to_numpy()
    y_eval = y_test.to_numpy()[has_lag]
    proba = test_proba[has_lag]

    # For log loss comparison, baseline class distribution
    prior = y_train.value_counts(normalize=True).sort_index().to_numpy()

    predictions = {
        "XGBoost": (np.argmax(proba, axis=1), proba),
        # Always predict the most frequent training class
        "Majority class": (np.full_like(y_eval, y_train.mode()[0]), None),
        # Predicts the symbol's most recently completed historical label
        "Persistence": (lagged_label[has_lag].map(stock_ratings).to_numpy(), None),
        # Predicts that the past horizon day price trend continues
        "Trailing return": (
            trailing_return_label(test_df)[has_lag].map(stock_ratings).to_numpy(),
            None,
        ),
        "Class prior": (None, np.tile(prior, (len(y_eval), 1))),
    }
    if class_scale is not None:
        predictions["XGBoost (Scaled)"] = (apply_class_scale(proba, class_scale), None)

    header = "".join(f"{r.value + ' F1':>10}" for r in StockRating)
    print(f"{'':<18}{'macro F1':>10}{header}{'log loss':>10}")

    results = {}
    for name, (y_pred, y_proba) in predictions.items():
        row = dict.fromkeys(
            ["macro F1", *(f"{r.value} F1" for r in StockRating), "log loss"]
        )
        if y_pred is not None:
            row["macro_f1"] = f1_score(y_eval, y_pred, average="macro")
            for rating, score in zip(
                StockRating, f1_score(y_eval, y_pred, average=None, labels=labels)
            ):
                row[f"{rating.value} F1"] = score
        if y_proba is not None:
            row["log loss"] = log_loss(y_eval, y_proba, labels=labels)
        results[name] = row
    print(
        pd.DataFrame.from_dict(results, orient="index").to_string(
            na_rep="-", float_format="{:.4f}".format
        )
    )


def predict_proba(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """
    Get the class probabilities.
    """

    return model.get_booster().predict(xgb.DMatrix(X))


def apply_class_scale(proba: np.ndarray, class_scale: np.ndarray) -> np.ndarray:
    """
    Decision rule, argmax of class probabilities multiplied by a per-class scale.
    """

    return np.argmax(proba * class_scale, axis=1)


def resolved_label_lag(df: pd.DataFrame) -> pd.Series:
    """
    Per row, the label of the latest window per symbol that already closed on or before
    that row's date. Matching on label_end_date rather than shifting rows.
    """

    left = df[["symbol", "date"]].reset_index().sort_values("date")
    right = df[["symbol", "label_end_date", "label"]].sort_values("label_end_date")
    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="label_end_date",
        by="symbol",
    )

    return merged.set_index("index")["label"].reindex(df.index)


def trailing_return_label(df: pd.DataFrame) -> pd.Series:
    """
    Naive prediction based on trend/momentum to label with the StockRating.
    """

    return pd.Series(
        np.select(
            condlist=[
                df["trailing_return"] > df["threshold"],
                df["trailing_return"] < -df["threshold"],
            ],
            choicelist=[StockRating.UP, StockRating.DOWN],
            default=StockRating.HOLD,
        ),
        index=df.index,
    )


if __name__ == "__main__":
    dataset = load_or_build_metrics_dataset()
    train_df, val_df, test_df = build_split_datasets(dataset)

    X_train, y_train = prepare_model_features(train_df)
    X_val, y_val = prepare_model_features(val_df)

    params = {
        "early_stopping_rounds": 50,
        "learning_rate": 0.05,
        "min_child_weight": 50,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    }

    model = fit_xgboost(X_train, y_train, X_val, y_val, params, verbose=25)
    evaluate_model(model, test_df, y_train)
