from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from data.loading import resolve_project_root

DEFAULT_MODEL_REL = Path("models") / "churn_pipeline.joblib"


def resolve_model_path(
    explicit: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    env_var: str = "CHURN_MODEL_PATH",
    default_rel: str | Path = DEFAULT_MODEL_REL,
) -> Path:
    root = resolve_project_root(project_root)
    raw = explicit if explicit is not None else os.environ.get(env_var)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        rel = default_rel
    else:
        rel = raw
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def load_churn_pipeline(
    path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    env_var: str = "CHURN_MODEL_PATH",
) -> Pipeline:
    
    resolved = resolve_model_path(path, project_root=project_root, env_var=env_var)
    if not resolved.is_file():
        raise FileNotFoundError(f"Model artifact not found: {resolved}")
    model: Any = joblib.load(resolved)
    if not isinstance(model, Pipeline):
        raise TypeError(
            f"Expected sklearn.pipeline.Pipeline, got {type(model).__name__} from {resolved}."
        )
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded estimator has no predict_proba; cannot return churn probability.")
    return model


def predict_churn_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X)!r}.")
    proba = model.predict_proba(X)
    return np.asarray(proba[:, 1], dtype=np.float64)
