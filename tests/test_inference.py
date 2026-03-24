"""Tests for :mod:`inference.predictor`."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.inference.predictor import (
    DEFAULT_MODEL_REL,
    load_churn_pipeline,
    predict_churn_proba,
    resolve_model_path,
)


def test_resolve_model_path_relative_to_project_root(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    models = root / "models"
    models.mkdir(parents=True)
    artifact = models / "churn_pipeline.joblib"
    artifact.write_bytes(b"x")

    p = resolve_model_path(
        "models/churn_pipeline.joblib",
        project_root=root,
        env_var="__UNUSED__",
    )
    assert p == artifact.resolve()


def test_resolve_model_path_absolute(tmp_path: Path) -> None:
    p_abs = tmp_path / "custom" / "model.joblib"
    p_abs.parent.mkdir(parents=True)
    p_abs.write_bytes(b"x")

    p = resolve_model_path(p_abs, env_var="__UNUSED__")
    assert p == p_abs.resolve()


def test_predict_churn_proba_requires_dataframe() -> None:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
    y = np.array([0, 1])
    pipe.fit(X, y)
    with pytest.raises(TypeError, match="DataFrame"):
        predict_churn_proba(pipe, X.to_numpy())


def test_predict_churn_proba_binary_positive_class(tmp_path: Path) -> None:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    y = (rng.random(40) > 0.5).astype(int)
    pipe.fit(X, y)
    path = tmp_path / "pl.joblib"
    joblib.dump(pipe, path)

    loaded = load_churn_pipeline(path, project_root=tmp_path, env_var="__UNUSED__")
    proba = predict_churn_proba(loaded, X.iloc[:5])
    assert proba.shape == (5,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_load_churn_pipeline_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_churn_pipeline(
            "models/missing.joblib",
            project_root=tmp_path,
            env_var="__UNUSED__",
        )


def test_load_churn_pipeline_rejects_non_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "bad.joblib"
    joblib.dump({"not": "pipeline"}, path)
    with pytest.raises(TypeError, match="Pipeline"):
        load_churn_pipeline(path, project_root=tmp_path, env_var="__UNUSED__")


def test_load_churn_pipeline_default_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHURN_MODEL_PATH", raising=False)
    models = tmp_path / "models"
    models.mkdir()
    artifact = models / DEFAULT_MODEL_REL.name
    pipe = Pipeline([("clf", LogisticRegression(max_iter=50))])
    pipe.fit(pd.DataFrame({"x": [0, 1]}), [0, 1])
    joblib.dump(pipe, artifact)

    loaded = load_churn_pipeline(project_root=tmp_path)
    assert isinstance(loaded, Pipeline)
