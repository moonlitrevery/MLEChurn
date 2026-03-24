"""Smoke tests for numeric drift report."""

from __future__ import annotations

import pandas as pd

from src.monitoring.data_drift import compute_numeric_drift_report


def test_drift_report_flags_large_mean_shift() -> None:
    ref = pd.DataFrame({"tenure": [1, 2, 3, 4, 5], "MonthlyCharges": [50.0] * 5})
    cur = pd.DataFrame({"tenure": [100, 101, 102, 103, 104], "MonthlyCharges": [50.0] * 5})
    rep = compute_numeric_drift_report(
        ref,
        cur,
        columns=["tenure", "MonthlyCharges"],
        mean_abs_threshold=10.0,
        std_abs_threshold=50.0,
        ks_alpha=0.05,
    )
    assert rep["summary"]["n_features_checked"] == 2
    assert rep["features"]["tenure"]["drift_mean_or_std"] is True
    assert rep["features"]["tenure"]["ks_pvalue"] is not None
