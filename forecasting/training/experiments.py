import json
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

from forecasting.data import START_DATE, build_split_datasets, split_date_ranges
from forecasting.features import compute_model_features
from forecasting.models import (
    IndicatorName,
    MarketIndicatorName,
    NewsFeatureName,
    StockRating,
    SummaryNewsFeatureName,
)
from forecasting.news_features import (
    NEWS_COUNT_FEATURES,
    load_fusion_dataset,
    load_or_build_news_features,
)
from forecasting.training.evaluation import (
    calibration_errors,
    ranking_frame,
    ranking_metrics,
    t_stat,
)
from forecasting.training.tuning import STUDY_NAME, build_params
from forecasting.training.xgboost_model import (
    ARTIFACTS_DIR,
    BASE_PARAMS,
    evaluate_model,
    fit_xgboost,
    predict_proba,
)

EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
SEEDS = [0, 1, 2]

PRICE_FEATURES = [str(c) for c in [*IndicatorName, *MarketIndicatorName]]
HEADLINE_FEATURES = [str(c) for c in NewsFeatureName]
SUMMARY_FEATURES = [str(c) for c in SummaryNewsFeatureName]
# Columns the evaluation needs besides the features.
EVAL_COLUMNS = [
    "symbol",
    "date",
    "label",
    "label_end_date",
    "trailing_return",
    "threshold",
    "forward_return",
    "close",
    "volume",
]
# The same-day headline features are loaded next to the normal ones with this suffix.
SAME_DAY_SUFFIX = "_same_day"


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    features: list[str]
    # Headline features from `news_features_same_day.parquet` (articles count on their
    # own date, still before the label's entry at the next open).
    same_day: bool = False
    # Headline features shuffled between the stocks of each date.
    shuffle_news: bool = False


EXPERIMENTS = [
    Experiment("A", "prices only", PRICE_FEATURES),
    Experiment("B", "prices + headline news", PRICE_FEATURES + HEADLINE_FEATURES),
    Experiment(
        "C",
        "prices + headline + summary news",
        PRICE_FEATURES + HEADLINE_FEATURES + SUMMARY_FEATURES,
    ),
    Experiment(
        "B_placebo",
        "B, headline news shuffled between stocks on each date",
        PRICE_FEATURES + HEADLINE_FEATURES,
        shuffle_news=True,
    ),
    Experiment(
        "B_same_day",
        "B with same-day news (one day fresher)",
        PRICE_FEATURES + HEADLINE_FEATURES,
        same_day=True,
    ),
]


def load_tuned_params() -> dict:
    """
    Hyperparameters of the saved price-only model (the best of STUDY_NAME), used for
    every experiment so they only differ in their features.
    """

    info = json.loads((ARTIFACTS_DIR / "xgboost_model.json").read_text())
    if info.get("study_name") != STUDY_NAME or info.get("start_date") != START_DATE:
        raise ValueError(
            f"The saved model is from study {info.get('study_name')} starting "
            f"{info.get('start_date')}, expected {STUDY_NAME} from {START_DATE}. "
            "Run `forecasting.training.tuning` first."
        )
    # Reads the search space's parameters back out of the saved ones.
    tuned = build_params(optuna.trial.FixedTrial(info["params"]))
    return {**BASE_PARAMS, **tuned}


def load_experiment_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train / val / test splits of the fused dataset, with only the columns the
    experiments use, plus the same-day headline features (suffixed SAME_DAY_SUFFIX).
    """

    fused = load_fusion_dataset(start_date=START_DATE)
    all_features = PRICE_FEATURES + HEADLINE_FEATURES + SUMMARY_FEATURES
    # The experiments list their features explicitly, check they are exactly the
    # columns the model would otherwise use.
    model_features = set(compute_model_features(fused.head())[0].columns)
    assert model_features == set(all_features), model_features ^ set(all_features)
    fused = fused[EVAL_COLUMNS + all_features]

    same_day = load_or_build_news_features(same_day=True, start_date=START_DATE)
    same_day = same_day[["symbol", "date", *HEADLINE_FEATURES]].rename(
        columns={c: f"{c}{SAME_DAY_SUFFIX}" for c in HEADLINE_FEATURES}
    )
    fused = fused.merge(same_day, on=["symbol", "date"], how="left", validate="1:1")
    del same_day
    counts = [f"{c}{SAME_DAY_SUFFIX}" for c in NEWS_COUNT_FEATURES]
    counts = [c for c in counts if c in fused]
    fused[counts] = fused[counts].fillna(0)

    return build_split_datasets(fused)


def shuffle_within_date(values: np.ndarray, dates: pd.Series, seed: int) -> np.ndarray:
    """
    Rows of `values` shuffled between the rows of the same date.
    """

    codes = pd.factorize(dates)[0]
    rng = np.random.default_rng(seed)
    by_date = np.lexsort((np.arange(len(codes)), codes))
    shuffled = np.lexsort((rng.random(len(codes)), codes))

    out = np.empty_like(values)
    out[by_date] = values[shuffled]
    return out


def experiment_features(
    df: pd.DataFrame, experiment: Experiment, seed: int
) -> pd.DataFrame:
    """
    The feature matrix of `experiment` for a split.
    """

    columns = [
        f"{c}{SAME_DAY_SUFFIX}" if experiment.same_day and c in HEADLINE_FEATURES else c
        for c in experiment.features
    ]
    X = df[columns].astype("float32")
    X.columns = experiment.features

    if experiment.shuffle_news:
        X[HEADLINE_FEATURES] = shuffle_within_date(
            X[HEADLINE_FEATURES].to_numpy(), df["date"], seed
        )
    return X


def run_experiment(
    experiment: Experiment,
    seed: int,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    params: dict,
) -> None:
    """
    Train one experiment with one seed, saving its test probabilities (`.npy`) and
    scores (`.json`). Skipped if a run with the same config is already saved.
    """

    train_df, val_df, test_df = splits
    config = {
        "description": experiment.description,
        "features": experiment.features,
        "same_day": experiment.same_day,
        "shuffle_news": experiment.shuffle_news,
        "seed": seed,
        "params": {**params, "random_state": seed},
        "splits": split_date_ranges(train=train_df, val=val_df, test=test_df),
    }
    directory = EXPERIMENTS_DIR / experiment.name
    result_path = directory / f"seed{seed}.json"
    if result_path.exists():
        if json.loads(result_path.read_text())["config"] == config:
            print(f"{experiment.name} seed {seed}: already done", flush=True)
            return

    index = StockRating.to_index_map()
    y_train, y_val = train_df["label"].map(index), val_df["label"].map(index)
    # A different shuffle per split, so the placebo can't line up across them.
    X_train = experiment_features(train_df, experiment, seed)
    X_val = experiment_features(val_df, experiment, seed + 1000)
    model = fit_xgboost(X_train, y_train, X_val, y_val, config["params"])
    del X_train, X_val

    X_test = experiment_features(test_df, experiment, seed + 2000)
    proba = predict_proba(model, X_test)
    scores = evaluate_model(model, test_df, y_train, X_test=X_test, show=False)

    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"seed{seed}.npy", proba.astype(np.float32))
    result = {
        "config": config,
        "best_iteration": int(model.best_iteration),
        "scores": scores,
        "calibration_errors": calibration_errors(
            proba, test_df["label"].map(index).to_numpy()
        ),
    }
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(
        f"{experiment.name} seed {seed}: macro F1 "
        f"{scores['XGBoost']['macro F1']:.4f}, log loss "
        f"{scores['XGBoost']['log loss']:.4f}, best iteration {model.best_iteration}",
        flush=True,
    )


def summarize(test_df: pd.DataFrame, experiments: list[Experiment]) -> dict:
    """
    Compare the saved experiment runs on the test set, print the tables and save them
    to `summary.json`. Ranking metrics are computed on the predictions averaged over
    the seeds; macro F1, log loss and IC are also given per seed (mean and std).
    """

    rows, per_seed, daily = {}, {}, {}
    for experiment in experiments:
        directory = EXPERIMENTS_DIR / experiment.name
        seeds = [s for s in SEEDS if (directory / f"seed{s}.npy").exists()]
        probas = [np.load(directory / f"seed{s}.npy") for s in seeds]
        results = [json.loads((directory / f"seed{s}.json").read_text()) for s in seeds]

        seed_ics = [
            ranking_metrics(ranking_frame(test_df, p))[0]["volatility_neutral_ic"]
            for p in probas
        ]
        per_seed[experiment.name] = {
            "seeds": len(seeds),
            "macro_f1": [r["scores"]["XGBoost"]["macro F1"] for r in results],
            "log_loss": [r["scores"]["XGBoost"]["log loss"] for r in results],
            "volatility_neutral_ic": seed_ics,
        }
        metrics, daily[experiment.name] = ranking_metrics(
            ranking_frame(test_df, np.mean(probas, axis=0))
        )
        ece = [np.mean(list(r["calibration_errors"].values())) for r in results]
        rows[experiment.name] = {**metrics, "mean_ece": float(np.mean(ece))}

    def mean_std(values: list[float]) -> str:
        return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

    print("\n== Per seed (mean ± std)")
    print(
        pd.DataFrame(
            {
                name: {
                    "seeds": s["seeds"],
                    "macro F1": mean_std(s["macro_f1"]),
                    "log loss": mean_std(s["log_loss"]),
                    "vol-neutral IC": mean_std(s["volatility_neutral_ic"]),
                }
                for name, s in per_seed.items()
            }
        ).T.to_string()
    )

    print("\n== Seed-averaged predictions")
    print(pd.DataFrame(rows).round(4).to_string())

    comparisons = {}
    if "A" in daily:
        for name, series in daily.items():
            if name == "A":
                continue
            diff = series - daily["A"]
            comparisons[name] = {
                f"delta_{column}": float(diff[column].mean()) for column in diff.columns
            } | {f"delta_{column}_t": t_stat(diff[column]) for column in diff.columns}
        print("\n== Difference vs A (per-date mean, t-stat on non-overlapping dates)")
        print(pd.DataFrame(comparisons).round(4).to_string())

    summary = {"per_seed": per_seed, "seed_averaged": rows, "vs_A": comparisons}
    (EXPERIMENTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_experiments(
    experiments: list[Experiment] = EXPERIMENTS, seeds: list[int] = SEEDS
) -> dict:
    params = load_tuned_params()
    splits = load_experiment_splits()
    for experiment in experiments:
        for seed in seeds:
            run_experiment(experiment, seed, splits, params)

    return summarize(splits[2], experiments)


if __name__ == "__main__":
    run_experiments()
