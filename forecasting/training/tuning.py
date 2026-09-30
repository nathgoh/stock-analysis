from pathlib import Path
import optuna

from forecasting.data import (
    build_split_datasets,
    load_or_build_metrics_dataset,
)
from forecasting.training.xgboost_model import (
    ARTIFACTS_DIR,
    BASE_PARAMS,
    evaluate_model,
    fit_xgboost,
    prepare_model_features
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
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15, step=0.05),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 50, 200, step=50)
    }

def run_trials(artifacts_dir: Path, n_trials: int = 50):
    """
    Run the trials for finetuning the hyperparameters.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_or_build_metrics_dataset()
    train_df, val_df, test_df = build_split_datasets(dataset)

    X_train, y_train = prepare_model_features(train_df)
    X_val, y_val = prepare_model_features(val_df)

    def objective(trial: optuna.Trial):
        params = build_params(trial)
        model = fit_xgboost(X_train, y_train, X_val, y_val, params)

        return float(model.best_score)

    study = optuna.create_study(
        study_name="xgboost-tuning",
        storage=f"sqlite:///{artifacts_dir / 'optuna_study.db'}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=n_trials)

    params = {**BASE_PARAMS, **study.best_params}
    model = fit_xgboost(
        X_train, y_train, X_val, y_val, params, verbose=25
    )

    print(f"Best validation log loss: {study.best_value:.4f}")
    print(f"Best parameters: {params}")

    evaluate_model(model, test_df, y_train)

if __name__ == "__main__":
    run_trials(ARTIFACTS_DIR)
