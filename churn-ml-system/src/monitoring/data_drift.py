"""Numeric drift vs a reference dataset (e.g. training slice); write reports under ``reports/``."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.logging_config import log_event
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
    Per numeric column: reference/current mean & std, absolute deltas, drift flags.

    Flags when mean or std delta exceeds the given thresholds (tune to your scale).
    Optional KS two-sample test when ``ks_alpha`` is set.
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

    ks_fn = None
    if ks_alpha is not None:
        try:
            from scipy.stats import ks_2samp
        except ImportError:
            logger.warning("scipy not installed; KS test skipped.")
            ks_alpha = None
        else:
            ks_fn = ks_2samp

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

        if ks_fn is not None and ks_alpha is not None:
            r = ref[~np.isnan(ref)]
            c = cur[~np.isnan(cur)]
            if len(r) > 1 and len(c) > 1:
                stat, pvalue = ks_fn(r, c)
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
    log_event(
        logger,
        logging.INFO,
        "drift_report_computed",
        features_checked=len(columns),
        flagged_mean_std=flagged_ms,
        flagged_ks=flagged_ks,
    )
    return {"features": features, "summary": summary}


def save_drift_report(
    report: dict[str, Any],
    reports_dir: str | Path,
    *,
    timestamp_utc: datetime | None = None,
) -> tuple[Path, Path]:
    """
    Write timestamped JSON (full report) and CSV (one row per feature) under ``reports_dir``.

    Returns ``(json_path, csv_path)``.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = (timestamp_utc or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    stem = f"numeric_drift_{ts}"
    generated_at = (timestamp_utc or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "generated_at_utc": generated_at,
        "features": report.get("features", {}),
        "summary": report.get("summary", {}),
    }
    json_path = reports_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for feat, stats in report.get("features", {}).items():
        row = {"feature": feat, **stats}
        rows.append(row)
    csv_path = reports_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    log_event(
        logger,
        logging.INFO,
        "drift_report_saved",
        json_path=str(json_path),
        csv_path=str(csv_path),
    )
    return json_path, csv_path
