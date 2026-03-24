"""Optuna search space and study runner for LightGBM pipeline hyperparameters."""

from __future__ import annotations

import logging
from typing import Any, Callable

import optuna
from sklearn.pipeline import Pipeline

from models.pipeline import build_lgbm_churn_pipeline
from models.training import cross_validate_stratified_roc_auc

logger = logging.getLogger(__name__)


def suggest_lgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Hyperparameters matched to :func:`build_lgbm_churn_pipeline` knobs."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 700),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 24, 96),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
    }


def make_cv_objective(
    X: Any,
    y: Any,
    *,
    n_splits: int,
    random_state: int,
    lgbm_n_jobs: int = 1,
) -> Callable[[optuna.Trial], float]:
    """
    Optuna objective: maximize mean out-of-fold ROC-AUC.

    Uses ``lgbm_n_jobs=1`` inside trials to avoid oversubscription when running
    parallel Optuna workers (set workers to 1 for heavy ``n_jobs`` on the booster).
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params(trial)
        pipe: Pipeline = build_lgbm_churn_pipeline(
            random_state=random_state,
            lgbm_n_jobs=lgbm_n_jobs,
            **params,
        )
        cv = cross_validate_stratified_roc_auc(
            pipe,
            X,
            y,
            n_splits=n_splits,
            random_state=random_state,
            verbose=False,
        )
        trial.set_user_attr("std_roc_auc", cv["std_roc_auc"])
        return cv["mean_roc_auc"]

    return objective


def run_hyperparameter_search(
    X: Any,
    y: Any,
    *,
    n_trials: int,
    n_splits: int,
    random_state: int,
    study_name: str | None = None,
    storage: str | None = None,
    callbacks: list[Any] | None = None,
    lgbm_n_jobs: int = 1,
    show_progress_bar: bool = False,
) -> optuna.Study:
    """Run an Optuna study maximizing CV ROC-AUC; returns the fitted :class:`optuna.Study`."""
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
        sampler=sampler,
    )
    objective = make_cv_objective(
        X,
        y,
        n_splits=n_splits,
        random_state=random_state,
        lgbm_n_jobs=lgbm_n_jobs,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks or [],
        show_progress_bar=show_progress_bar,
    )
    logger.info(
        "Optuna finished: best_value=%.6f params=%s",
        study.best_value,
        study.best_params,
    )
    return study
