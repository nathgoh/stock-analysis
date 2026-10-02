from optuna.trial import FrozenTrial
import numpy as np
from sklearn.metrics import f1_score
from pathlib import Path
import optuna

from forecasting.data import (
    build_split_datasets,
    load_or_build_metrics_dataset,
    walk_forward_splits,
)
from forecasting.training.xgboost_model import (
    ARTIFACTS_DIR,
    BASE_PARAMS,
    evaluate_model,
    fit_xgboost,
    prepare_model_features,
    save_model
)


def build_params(trial: optuna.Trial) -> dict:
    """
    Search space to tune hyperparameters on.
    """

    return {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 50, 5000, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "early_stopping_rounds": trial.suggest_int(
            "early_stopping_rounds", 50, 200, step=50
        ),
    }


def run_trials(artifacts_dir: Path, n_trials: int = 50):
    """
    Run the trials for finetuning the hyperparameters.
    """

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_or_build_metrics_dataset()
    train_df, val_df, test_df = build_split_datasets(dataset)

    fold_features = []
    for fold_train, fold_val in walk_forward_splits(dataset, n_folds=5):
        X_train_fold, y_train_fold = prepare_model_features(fold_train)
        X_val_fold, y_val_fold = prepare_model_features(fold_val)

        fold_features.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))

    def objective(trial: optuna.Trial):
        params = build_params(trial)
        losses = []
        f1_scores = []

        for X_train_fold, y_train_fold, X_val_fold, y_val_fold in fold_features:
            model = fit_xgboost(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                params,
            )

            y_pred = model.predict(X_val_fold)

            losses.append(float(model.best_score))
            f1_scores.append(
                f1_score(
                    y_val_fold,
                    y_pred,
                    average="macro",
                    labels=model.classes_,
                    zero_division=0,
                )
            )

        # Save individual fold results for inspection.
        trial.set_user_attr("fold_log_losses", losses)
        trial.set_user_attr("fold_macro_f1", f1_scores)

        return float(np.mean(losses)), float(np.mean(f1_scores))

    study = optuna.create_study(
        study_name="xgboost-tuning",
        storage=f"sqlite:///{artifacts_dir / 'optuna_study.db'}",
        load_if_exists=True,
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(objective, n_trials=n_trials)

    def selection_key(trial: FrozenTrial):
        values = trial.values
        assert values is not None

        return values[1], -values[0]

    trials = [
        trial for trial in study.best_trials
        if trial.values is not None
    ]
    best_trial = max(
        trials,
        key=selection_key,
    )
    values = best_trial.values
    assert values is not None

    X_train, y_train = prepare_model_features(train_df)
    X_val, y_val = prepare_model_features(val_df)

    params = {**BASE_PARAMS, **best_trial.params}
    model = fit_xgboost(X_train, y_train, X_val, y_val, params, verbose=25)

    print(f"Mean validation log loss: {values[0]:.4f}")
    print(f"Mean validation macro F1: {values[1]:.4f}")
    print(f"Best parameters: {params}")

    evaluate_model(model, test_df, y_train)
    save_model(model)


if __name__ == "__main__":
    run_trials(ARTIFACTS_DIR)
