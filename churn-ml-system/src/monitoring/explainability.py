"""SHAP explanations for the trained LightGBM step inside the sklearn pipeline."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _transform_features(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Expected sklearn Pipeline.")
    if "model" not in pipeline.named_steps:
        raise ValueError("Pipeline must have a 'model' step (LightGBM classifier).")
    pre = pipeline[:-1]
    Xt = pre.transform(X)
    if hasattr(Xt, "to_numpy"):
        return np.asarray(Xt.to_numpy(dtype=np.float64), dtype=np.float64)
    return np.asarray(Xt, dtype=np.float64)


def _feature_names(pipeline: Pipeline, n_features: int) -> list[str]:
    pre = pipeline[:-1]
    try:
        names = pre.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        return [f"feature_{i}" for i in range(n_features)]


def _lgbm_classifier(pipeline: Pipeline) -> Any:
    clf = pipeline.named_steps["model"]
    return clf


def _expected_value_positive_class(explainer: Any) -> float | None:
    ev = explainer.expected_value
    if ev is None:
        return None
    arr = np.asarray(ev, dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    return float(arr[-1]) if arr.size > 1 else float(arr[0])


def explain_global(pipeline: Pipeline, X: pd.DataFrame) -> dict[str, Any]:
    """
    Global importance: mean absolute SHAP value per transformed feature (churn class).

    ``X`` should match training-style raw columns (same as inference).
    """
    clf = _lgbm_classifier(pipeline)
    Xt = _transform_features(pipeline, X)
    names = _feature_names(pipeline, Xt.shape[1])
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = np.asarray(sv[1], dtype=np.float64)
    else:
        sv = np.asarray(sv, dtype=np.float64)
    mean_abs = np.mean(np.abs(sv), axis=0)
    logger.info("explain_global: n=%s rows, %s features", Xt.shape[0], len(names))
    return {
        "feature_names": names,
        "mean_abs_shap": mean_abs.tolist(),
        "expected_value": _expected_value_positive_class(explainer),
    }


def explain_instance(pipeline: Pipeline, X_row: pd.DataFrame) -> dict[str, Any]:
    """
    Local explanation for a single row (DataFrame with one row).

    Returns SHAP values aligned with ``feature_names`` and predicted churn probability.
    """
    if len(X_row) != 1:
        raise ValueError("X_row must contain exactly one row.")
    clf = _lgbm_classifier(pipeline)
    Xt = _transform_features(pipeline, X_row)
    names = _feature_names(pipeline, Xt.shape[1])
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        row = np.asarray(sv[1], dtype=np.float64).reshape(-1)
    else:
        row = np.asarray(sv, dtype=np.float64).reshape(-1)
    proba = float(pipeline.predict_proba(X_row)[0, 1])
    base_val = _expected_value_positive_class(explainer)
    if base_val is None:
        base_val = float("nan")
    logger.debug("explain_instance: proba=%.6f", proba)
    return {
        "feature_names": names,
        "shap_values": row.tolist(),
        "expected_value": base_val,
        "churn_probability": proba,
    }
