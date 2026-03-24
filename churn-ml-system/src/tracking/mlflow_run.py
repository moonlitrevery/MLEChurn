"""MLflow helpers for training runs (metrics, params, sklearn pipeline artifact)."""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from typing import Any, Mapping

import mlflow
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def start_mlflow_run(
    experiment_name: str,
    *,
    run_name: str | None = None,
    tracking_uri: str | None = None,
) -> AbstractContextManager[Any]:
    """
    Set experiment and return ``mlflow.start_run`` context manager.

    ``tracking_uri`` defaults to ``MLFLOW_TRACKING_URI`` or local ``./mlruns``.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_training_metrics(cv_result: Mapping[str, Any]) -> None:
    mlflow.log_metric("cv_mean_roc_auc", float(cv_result["mean_roc_auc"]))
    mlflow.log_metric("cv_std_roc_auc", float(cv_result["std_roc_auc"]))
    for i, score in enumerate(cv_result["fold_scores"], start=1):
        mlflow.log_metric(f"cv_fold_{i}_roc_auc", float(score))


def log_sklearn_pipeline(
    pipeline: Pipeline,
    *,
    artifact_path: str = "churn_pipeline",
    registered_model_name: str | None = None,
) -> None:
    """Persist the fitted sklearn pipeline as an MLflow model artifact."""
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )
    logger.info("Logged sklearn pipeline to MLflow artifact_path=%s", artifact_path)
