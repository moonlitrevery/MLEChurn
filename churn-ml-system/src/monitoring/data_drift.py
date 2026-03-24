"""Univariate numeric drift between a reference frame (e.g. training) and new data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.schema import NUMERIC_COLUMNS

logger = logging.getLogger(__name__)


def compute_numeric_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    mean_abs_threshold: float = 5.0,
    std_abs_threshold: float = 3.0,
    ks_alpha: float | None = 0.05,
) -> dict[str, Any]:
    """
    Compare numeric columns: means/stds, absolute deltas, optional KS two-sample test.

    Drift flags (per feature) when ``abs(mean_ref - mean_cur) > mean_abs_threshold``
    OR ``abs(std_ref - std_cur) > std_abs_threshold``. Tune thresholds to your scale;
    defaults are loose for charges-like magnitudes—tighten for bounded features (e.g. 0/1).

    If ``ks_alpha`` is not None, runs ``scipy.stats.ks_2samp`` and flags low p-value as drift.

    Returns:
        Structured report with per-feature stats and summary counts.
    """
    if columns is None:
        columns = [c for c in NUMERIC_COLUMNS if c in reference_df.columns and c in current_df.columns]
    else:
        columns = [c for c in columns if c in reference_df.columns and c in current_df.columns]

    if not columns:
        logger.warning("No overlapping numeric columns for drift check.")
        return {
            "features": {},
            "summary": {
                "n_features_checked": 0,
                "n_flagged_mean_std": 0,
                "n_flagged_ks": 0,
            },
        }

    ks_2samp = None
    if ks_alpha is not None:
        try:
            from scipy.stats import ks_2samp as _ks
        except ImportError:
            logger.warning("scipy not installed; KS test skipped.")
            ks_alpha = None
        else:
            ks_2samp = _ks

    features: dict[str, Any] = {}
    flagged_ms = 0
    flagged_ks = 0

    for col in columns:
        ref = pd.to_numeric(reference_df[col], errors="coerce").to_numpy(dtype=float)
        cur = pd.to_numeric(current_df[col], errors="coerce").to_numpy(dtype=float)
        ref_m = float(np.nanmean(ref))
        cur_m = float(np.nanmean(cur))
        ref_s = float(np.nanstd(ref, ddof=0))
        cur_s = float(np.nanstd(cur, ddof=0))
        d_mean = abs(ref_m - cur_m)
        d_std = abs(ref_s - cur_s)
        drift_mean_std = (d_mean > mean_abs_threshold) or (d_std > std_abs_threshold)

        entry: dict[str, Any] = {
            "reference_mean": ref_m,
            "current_mean": cur_m,
            "abs_mean_diff": d_mean,
            "reference_std": ref_s,
            "current_std": cur_s,
            "abs_std_diff": d_std,
            "drift_mean_or_std": drift_mean_std,
        }

        if ks_2samp is not None and ks_alpha is not None:
            r = ref[~np.isnan(ref)]
            c = cur[~np.isnan(cur)]
            if len(r) > 1 and len(c) > 1:
                stat, pvalue = ks_2samp(r, c)
                ks_drift = bool(pvalue < ks_alpha)
                entry["ks_statistic"] = float(stat)
                entry["ks_pvalue"] = float(pvalue)
                entry["drift_ks"] = ks_drift
                if ks_drift:
                    flagged_ks += 1
            else:
                entry["ks_statistic"] = None
                entry["ks_pvalue"] = None
                entry["drift_ks"] = None
        else:
            entry["ks_statistic"] = None
            entry["ks_pvalue"] = None
            entry["drift_ks"] = None

        if drift_mean_std:
            flagged_ms += 1
        features[col] = entry

    summary = {
        "n_features_checked": len(columns),
        "n_flagged_mean_std": flagged_ms,
        "n_flagged_ks": flagged_ks,
        "mean_abs_threshold": mean_abs_threshold,
        "std_abs_threshold": std_abs_threshold,
        "ks_alpha": ks_alpha,
    }
    logger.info(
        "Drift report: %s features, %s flagged (mean/std), %s flagged (KS).",
        len(columns),
        flagged_ms,
        flagged_ks,
    )
    return {"features": features, "summary": summary}
