"""Tests for :class:`features.engineering.ChurnFeatureEngineer`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import OUTPUT_COLUMNS, ChurnFeatureEngineer


def _minimal_frame(n: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": i,
                "tenure": 1 + i * 10,
                "MonthlyCharges": 50.0 + i,
                "TotalCharges": 100.0 * (i + 1),
                "InternetService": "Fiber optic" if i % 2 == 0 else "DSL",
                "MultipleLines": "Yes" if i % 2 == 0 else "No",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
            }
        )
    return pd.DataFrame(rows)


def test_churn_feature_engineer_columns_and_shape() -> None:
    X = _minimal_frame()
    fe = ChurnFeatureEngineer()
    out = fe.fit_transform(X)
    assert list(out.columns) == list(OUTPUT_COLUMNS)
    assert out.shape == (len(X), len(OUTPUT_COLUMNS))


def test_churn_feature_engineer_avg_monthly_matches_formula() -> None:
    X = pd.DataFrame(
        [
            {
                "tenure": 10,
                "MonthlyCharges": 80.0,
                "TotalCharges": 800.0,
                "InternetService": "DSL",
                "MultipleLines": "No",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
            }
        ]
    )
    fe = ChurnFeatureEngineer()
    out = fe.fit_transform(X)
    assert np.isclose(out.loc[0, "avg_monthly_from_total"], 80.0)
    assert np.isclose(out.loc[0, "monthly_minus_implied_avg"], 0.0)


def test_churn_feature_engineer_tenure_zero_uses_safe_divisor() -> None:
    X = pd.DataFrame(
        [
            {
                "tenure": 0,
                "MonthlyCharges": 70.0,
                "TotalCharges": 70.0,
                "InternetService": "DSL",
                "MultipleLines": "No",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
            }
        ]
    )
    fe = ChurnFeatureEngineer()
    out = fe.fit_transform(X)
    assert np.isclose(out.loc[0, "avg_monthly_from_total"], 70.0)


def test_churn_feature_engineer_missing_columns_raises() -> None:
    X = _minimal_frame().drop(columns=["tenure"])
    fe = ChurnFeatureEngineer()
    with pytest.raises(ValueError, match="Missing required columns"):
        fe.fit(X)


def test_churn_feature_engineer_coerces_invalid_totalcharges_with_median() -> None:
    X = pd.DataFrame(
        [
            {
                "tenure": 1,
                "MonthlyCharges": 50.0,
                "TotalCharges": 100.0,
                "InternetService": "DSL",
                "MultipleLines": "No",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
            },
            {
                "tenure": 2,
                "MonthlyCharges": 60.0,
                "TotalCharges": np.nan,
                "InternetService": "DSL",
                "MultipleLines": "No",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
            },
        ]
    )
    fe = ChurnFeatureEngineer()
    fe.fit(X)
    assert fe.total_charges_median_ == 100.0
    out = fe.transform(X)
    assert not out["avg_monthly_from_total"].isna().any()
