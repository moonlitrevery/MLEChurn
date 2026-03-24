"""Cross-validation and fit helpers for the churn pipeline."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def prepare_churn_target(y: pd.Series | np.ndarray | list[Any]) -> np.ndarray:
    """
    Map ``Churn`` labels to binary {0, 1}.

    Accepts ``Yes``/``No`` strings (case-insensitive) or numeric/boolean targets.
    """
    if isinstance(y, pd.Series):
        s = y
    else:
        s = pd.Series(y)
    if pd.api.types.is_bool_dtype(s):
        return s.astype(np.int8).to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        u = set(pd.unique(s.dropna()))
        if u <= {0, 1}:
            return s.astype(np.int8).to_numpy()
    mapped = s.astype(str).str.strip().str.lower().eq("yes").astype(np.int8)
    return mapped.to_numpy()


def cross_validate_stratified_roc_auc(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Stratified K-fold CV; logs per-fold and mean ROC-AUC when ``verbose``.

    Returns:
        ``fold_scores``, ``mean_roc_auc``, ``std_roc_auc``.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    fold_scores: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        estimator = clone(pipeline)
        estimator.fit(X.iloc[train_idx], y[train_idx])
        proba = estimator.predict_proba(X.iloc[val_idx])[:, 1]
        score = roc_auc_score(y[val_idx], proba)
        fold_scores.append(float(score))
        if verbose:
            logger.info("Fold %s/%s ROC-AUC: %.6f", fold, n_splits, score)

    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    if verbose:
        logger.info("Mean ROC-AUC: %.6f (std %.6f)", mean_auc, std_auc)
    return {
        "fold_scores": fold_scores,
        "mean_roc_auc": mean_auc,
        "std_roc_auc": std_auc,
    }


def fit_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    """Fit a cloned pipeline on all rows (use after CV for final model)."""
    fitted = clone(pipeline)
    fitted.fit(X, y)
    return fitted
