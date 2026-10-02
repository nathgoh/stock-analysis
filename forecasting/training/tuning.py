from pathlib import Path

import numpy as np
import optuna
from optuna.trial import FrozenTrial
from sklearn.metrics import f1_score

from forecasting.data import (
    build_split_datasets,
    load_or_build_metrics_dataset,
    sample_stock_symbols,
    walk_forward_splits,
)
from forecasting.training.xgboost_model import (
    ARTIFACTS_DIR,
    BASE_PARAMS,
    evaluate_model,
    fit_xgboost,
    predict_proba,
    prepare_model_features,
    save_model,
)

# The search runs on a subset of symbols to keep each trial cheap, the final model
# refits on all symbols.
SEARCH_SYMBOL_FRACTION = 0.3
SEARCH_SEED = 0
N_FOLDS = 5
# Pareto trials re-checked on the full data before one is picked.
FINAL_CANDIDATES = 3


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
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
    }


def run_trials(artifacts_dir: Path, n_trials: int = 50):
    """
    Run the trials for finetuning the hyperparameters.
    """

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_or_build_metrics_dataset()
    train_df, val_df, test_df = build_split_datasets(dataset)
    search_dataset = sample_stock_symbols(
        dataset, SEARCH_SYMBOL_FRACTION, seed=SEARCH_SEED
    )
    folds = walk_forward_splits(
        search_dataset, n_folds=N_FOLDS, boundary_dates=dataset["date"].unique()
    )
    assert folds[-1][1]["date"].max() < test_df["date"].min()

    fold_features = []
    for fold_train, fold_val in folds:
        assert fold_train["label_end_date"].max() < fold_val["date"].min()
        X_train_fold, y_train_fold = prepare_model_features(fold_train)
        X_val_fold, y_val_fold = prepare_model_features(fold_val)

        fold_features.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
    del folds, search_dataset

    def objective(trial: optuna.Trial):
        params = build_params(trial)
        fit_params = {
            **params,
            "min_child_weight": params["min_child_weight"] * SEARCH_SYMBOL_FRACTION
        }
        losses = []
        f1_scores = []
        iterations = []

        for X_train_fold, y_train_fold, X_val_fold, y_val_fold in fold_features:
            model = fit_xgboost(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                y_val_fold,
                fit_params,
            )

            y_pred = np.argmax(predict_proba(model, X_val_fold), axis=1)

            losses.append(float(model.best_score))
            iterations.append(int(model.best_iteration) + 1)
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
        trial.set_user_attr("best_iterations", iterations)

        return float(np.mean(losses)), float(np.mean(f1_scores))

    # Trials from a different fraction, seed or fold count are not comparable, so a
    # resumed study must have been created with the same search settings.
    study = optuna.create_study(
        study_name="xgboost-tuning",
        storage=f"sqlite:///{artifacts_dir / 'optuna_study.db'}",
        load_if_exists=True,
        directions=["minimize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    search_config = {
        "fraction": SEARCH_SYMBOL_FRACTION,
        "seed": SEARCH_SEED,
        "n_folds": N_FOLDS,
    }
    stored_config = study.user_attrs.get("search_config")
    if stored_config is None:
        study.set_user_attr("search_config", search_config)
    elif stored_config != search_config:
        raise ValueError(
            f"Study was created with {stored_config}, now {search_config}. "
            "Use a new study name or storage."
        )
    study.optimize(objective, n_trials=n_trials)

    def selection_key(trial: FrozenTrial):
        values = trial.values
        assert values is not None

        return values[1], -values[0]

    trials = sorted(
        (trial for trial in study.best_trials if trial.values is not None),
        key=selection_key,
        reverse=True,
    )

    # Params were tuned on a symbol subset, so refit the top candidates on all symbols
    # and pick by the validation score there. Trial params are already on the full-data
    # scale (min_child_weight is only scaled inside the objective).
    X_train, y_train = prepare_model_features(train_df)
    X_val, y_val = prepare_model_features(val_df)

    best_model, best_params, best_key = None, None, None
    for trial in trials[:FINAL_CANDIDATES]:
        assert trial.values is not None

        params = {**BASE_PARAMS, **trial.params}
        model = fit_xgboost(X_train, y_train, X_val, y_val, params, verbose=25)
        val_f1 = f1_score(
            y_val,
            np.argmax(predict_proba(model, X_val), axis=1),
            average="macro",
            labels=model.classes_,
            zero_division=0,
        )
        key = (val_f1, -float(model.best_score))
        print(
            f"Trial {trial.number}: search log loss {trial.values[0]:.4f}, "
            f"search macro F1 {trial.values[1]:.4f} | "
            f"full-data val log loss {model.best_score:.4f}, macro F1 {val_f1:.4f}"
        )
        if best_key is None or key > best_key:
            best_model, best_params, best_key = model, params, key

    assert best_model is not None
    print(f"Best parameters: {best_params}")

    evaluate_model(best_model, test_df, y_train)
    save_model(best_model)


if __name__ == "__main__":
    run_trials(ARTIFACTS_DIR)
