import numpy as np
import cupy as cp
import pandas as pd
import xgboost as xgb
import optuna

from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss

from forecasting.data import (
    build_split_datasets,
    class_weights,
    load_or_build_metrics_dataset,
)
from forecasting.features import compute_model_features
from forecasting.models import StockRating


def train_xgboost(trial):
    dataset = load_or_build_metrics_dataset()
    train_df, val_df, test_df = build_split_datasets(dataset)

    X_train, y_train = compute_model_features(train_df)
    X_val, y_val = compute_model_features(val_df)
    X_test, y_test = compute_model_features(test_df)
    X_test_gpu = cp.asarray(X_test.to_numpy(dtype="float32"))

    rating_index = StockRating.to_index_map()
    y_train_int = y_train.map(rating_index)
    y_val_int = y_val.map(rating_index)
    y_test_int = y_test.map(rating_index)

    # Computer class weight for training labelts only
    weights = class_weights(y_train)
    weight_train = y_train.map(weights).to_numpy()
    weight_val = y_val.map(weights).to_numpy()

    params = {
        "objective": "multi:softprob",
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.2, 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.7, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, 0.15),
        "early_stopping_rounds": trial.suggest_float("early_stopping_rounds", 25, 50, 100)
    }

    if params["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        params["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        params["min_child_weight"] = trial.suggest_int("min_child_weight", 10, 20, 50)
        params["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if params["booster"] == "dart":
        params["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        params["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        params["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        params["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)


    model = xgb.XGBClassifier(params)
    model.fit(
        X_train,
        y_train_int,
        sample_weight=weight_train,
        eval_set=[(X_val, y_val_int)],
        sample_weight_eval_set=[weight_val],
        verbose=25,
    )

    prediction = model.predict(X_test_gpu)
#     print(
#         classification_report(
#             y_test_int,
#             prediction,
#             target_names=[StockRating.DOWN, StockRating.HOLD, StockRating.UP],
#         )
#     )
#     print(confusion_matrix(y_test_int, prediction))
#
#     # Benchmark against persistence baseline, which only scores dates whwere a prior horizon window has been resolved.
#     # Filter all models to this subset for comparison.
#     lagged_label = resolved_label_lag(test_df)
#     has_lag = lagged_label.notna().to_numpy()
#     y_eval = y_test_int.to_numpy()[has_lag]
#
#     class_indices = list(rating_index.values())
#     prior = y_train_int.value_counts(normalize=True).sort_index().to_numpy()
#     probabilities = {
#         "XGBoost": model.predict_proba(X_test)[has_lag],
#
#         # For log loss comparison, baseline class distribution
#         "Class prior": np.tile(prior, (len(y_eval), 1)),
#     }
#     predictions = {
#         "XGBoost": prediction[has_lag],
#         # Always predict the most frequent training class
#         "Majority class": np.full_like(y_eval, y_train_int.mode()[0]),
#         # Predicts the symbol's most recently completed historical lable
#         "Persistence": lagged_label[has_lag].map(rating_index).to_numpy(),
#         # Predicts that the past horizon day price trend continues
#         "Trailing return": trailing_return_label(test_df)[has_lag]
#         .map(rating_index)
#         .to_numpy(),
#     }
#
#     header = "".join(f"{r.value + ' F1':>10}" for r in StockRating)
#     print(f"{'':<18}{'macro F1':>10}{header}{'log loss':>10}")
#     for name, y_pred in predictions.items():
#         macro_f1 = f1_score(y_eval, y_pred, average="macro")
#         per_class_f1 = f1_score(y_eval, y_pred, average=None, labels=class_indices)
#
#         # Hard-label baselines have no probabilities, so no log loss
#         proba = probabilities.get(name)
#         loss = (
#             f"{log_loss(y_eval, proba, labels=class_indices):>10.4f}"
#             if proba is not None
#             else f"{'-':>10}"
#         )
#         per_class = "".join(f"{f:>10.4f}" for f in per_class_f1)
#         print(f"{name:<18}{macro_f1:>10.4f}{per_class}{loss}")
#     prior_loss = log_loss(y_eval, probabilities["Class prior"], labels=class_indices)
#     print(f"{'Class prior':<18}{'-':>10}{'-':>10}{'-':>10}{'-':>10}{prior_loss:>10.4f}")


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
    study = optuna.create_study(direction="maximize")
    study.optimize(train_xgboost, n_trials=100)
