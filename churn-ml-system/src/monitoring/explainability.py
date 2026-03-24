"""SHAP values for the LightGBM step (after the rest of the pipeline transforms raw rows)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def design_matrix_as_array(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply all steps before ``model``; return a dense float matrix for the booster."""
    if not isinstance(pipeline, Pipeline) or "model" not in pipeline.named_steps:
        raise ValueError("Expected a sklearn Pipeline with a 'model' step.")
    Xt = pipeline[:-1].transform(X)
    if hasattr(Xt, "to_numpy"):
        return np.asarray(Xt.to_numpy(dtype=np.float64), dtype=np.float64)
    return np.asarray(Xt, dtype=np.float64)


def design_matrix_column_names(pipeline: Pipeline, n_columns: int) -> list[str]:
    try:
        names = pipeline[:-1].get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        return [f"feature_{i}" for i in range(n_columns)]


def pipeline_lightgbm_classifier(pipeline: Pipeline) -> Any:
    return pipeline.named_steps["model"]


def expected_value_class_one(explainer: Any) -> float | None:
    ev = explainer.expected_value
    if ev is None:
        return None
    arr = np.asarray(ev, dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    return float(arr[-1]) if arr.size > 1 else float(arr[0])


def compute_batch_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, list[str], Any, float | None]:
    """
    One TreeExplainer fit; SHAP values for every row (positive class).

    Returns ``(values_2d, feature_names, explainer, expected_value_class_one)``.
    """
    clf = pipeline_lightgbm_classifier(pipeline)
    Xm = design_matrix_as_array(pipeline, X)
    names = design_matrix_column_names(pipeline, Xm.shape[1])
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xm)
    if isinstance(sv, list):
        matrix = np.asarray(sv[1], dtype=np.float64)
    else:
        matrix = np.asarray(sv, dtype=np.float64)
    ev1 = expected_value_class_one(explainer)
    logger.info("batch_shap: rows=%s features=%s", matrix.shape[0], matrix.shape[1])
    return matrix, names, explainer, ev1


def explain_global(pipeline: Pipeline, X: pd.DataFrame) -> dict[str, Any]:
    """Mean |SHAP| per transformed column (global importance for this sample)."""
    matrix, names, explainer, ev1 = compute_batch_shap_values(pipeline, X)
    mean_abs = np.mean(np.abs(matrix), axis=0)
    return {
        "feature_names": names,
        "mean_abs_shap": mean_abs.tolist(),
        "expected_value": ev1,
    }


def explain_instance(pipeline: Pipeline, X_row: pd.DataFrame) -> dict[str, Any]:
    """SHAP vector for a single row (same order as ``feature_names``)."""
    if len(X_row) != 1:
        raise ValueError("Pass exactly one row.")
    matrix, names, _, ev1 = compute_batch_shap_values(pipeline, X_row)
    row = matrix.reshape(-1)
    proba = float(pipeline.predict_proba(X_row)[0, 1])
    return {
        "feature_names": names,
        "shap_values": row.tolist(),
        "expected_value": float(ev1) if ev1 is not None else float("nan"),
        "churn_probability": proba,
    }
