"""
Batch CSV scoring: predictions, optional numeric drift vs a reference file, optional SHAP exports.

Run from repo with ``PYTHONPATH`` pointing at the directory that contains ``src``::

    PYTHONPATH=churn-ml-system python -m src.batch.run_batch -i data.csv -o scored.csv --project-root .
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Parent of ``src`` (e.g. churn-ml-system/) must be on sys.path before ``import src``.
_SRC_PARENT = Path(__file__).resolve().parents[2]
if str(_SRC_PARENT) not in sys.path:
    sys.path.insert(0, str(_SRC_PARENT))

from src.common.logging_config import log_event, setup_logging  # noqa: E402
from src.data.loading import load_csv, resolve_project_root  # noqa: E402
from src.inference.predictor import load_churn_pipeline, predict_churn_proba  # noqa: E402
from src.models.schema import TARGET_COLUMN  # noqa: E402
from src.monitoring.data_drift import (  # noqa: E402
    compute_numeric_drift_report,
    save_drift_report,
)
from src.monitoring.explainability import compute_batch_shap_values, explain_global  # noqa: E402

logger = logging.getLogger(__name__)


def _resolve_output_path(path: Path, project_root: Path | None) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    base = resolve_project_root(project_root) if project_root is not None else Path.cwd()
    return (base / path).resolve()


def _resolve_reports_dir(reports_dir: Path, project_root: Path | None) -> Path:
    reports_dir = reports_dir.expanduser()
    if reports_dir.is_absolute():
        return reports_dir.resolve()
    base = resolve_project_root(project_root) if project_root is not None else Path.cwd()
    return (base / reports_dir).resolve()


def _shap_column_name(feature: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in str(feature))
    return f"shap_{safe}"


def run_batch_inference(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path | None = None,
    project_root: str | Path | None = None,
    reports_dir: str | Path = "reports",
    with_shap: bool = False,
    drift_reference_path: str | Path | None = None,
    **read_csv_kwargs: Any,
) -> Path:
    """
    Score rows, write ``output_path`` with ``churn_probability``.

    If ``drift_reference_path`` is set, compares numeric columns to that CSV and writes
    timestamped JSON+CSV under ``reports_dir`` via :func:`save_drift_report`.

    If ``with_shap`` is True, writes under ``reports_dir``:

    - ``batch_shap_values_<run_id>.csv`` — row index + one SHAP column per transformed feature
    - ``batch_shap_global_<run_id>.json`` — mean |SHAP| for the scored batch
    """
    setup_logging()
    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime("%Y%m%d_%H%M%S")
    pr = Path(project_root).resolve() if project_root is not None else None

    df = load_csv(input_path, project_root=project_root, **read_csv_kwargs)
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")

    model = load_churn_pipeline(model_path, project_root=project_root)
    proba = predict_churn_proba(model, X)

    out = df.copy()
    out["churn_probability"] = proba
    outp = _resolve_output_path(Path(output_path), pr)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)

    rep_root = _resolve_reports_dir(Path(reports_dir), pr)
    rep_root.mkdir(parents=True, exist_ok=True)

    if drift_reference_path is not None:
        ref_df = load_csv(drift_reference_path, project_root=project_root, **read_csv_kwargs)
        drift = compute_numeric_drift_report(ref_df, X)
        save_drift_report(drift, rep_root, timestamp_utc=run_started)

    if with_shap:
        matrix, names, _, _ = compute_batch_shap_values(model, X)
        shap_df = pd.DataFrame(
            matrix,
            columns=[_shap_column_name(n) for n in names],
        )
        shap_df.insert(0, "batch_row_index", np.arange(len(shap_df)))
        if "id" in df.columns:
            shap_df.insert(1, "id", df["id"].values)
        shap_path = rep_root / f"batch_shap_values_{run_id}.csv"
        shap_df.to_csv(shap_path, index=False)

        global_report = explain_global(model, X)
        global_report["generated_at_utc"] = run_started.strftime("%Y-%m-%dT%H:%M:%SZ")
        global_report["n_scored_rows"] = len(X)
        global_path = rep_root / f"batch_shap_global_{run_id}.json"
        global_path.write_text(json.dumps(global_report, indent=2), encoding="utf-8")

        log_event(
            logger,
            logging.INFO,
            "batch_shap_saved",
            shap_values_csv=str(shap_path),
            shap_global_json=str(global_path),
        )

    log_event(
        logger,
        logging.INFO,
        "batch_inference_complete",
        rows=len(out),
        output_csv=str(outp),
        run_id=run_id,
        shap=with_shap,
        drift=bool(drift_reference_path),
    )
    return outp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", type=str, required=True, help="Input CSV path.")
    p.add_argument("--output", "-o", type=str, required=True, help="Output CSV path.")
    p.add_argument("--model", "-m", type=str, default=None, help="Model joblib path.")
    p.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Anchor for relative paths (CHURN_PROJECT_ROOT).",
    )
    p.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for drift + SHAP artifacts (default: ./reports under project root).",
    )
    p.add_argument(
        "--with-shap",
        action="store_true",
        help="Export SHAP values (per row) and global summary JSON under reports-dir.",
    )
    p.add_argument(
        "--drift-reference",
        type=str,
        default=None,
        help="Reference CSV (e.g. training sample) for numeric drift report.",
    )
    args = p.parse_args()
    run_batch_inference(
        args.input,
        args.output,
        model_path=args.model,
        project_root=args.project_root,
        reports_dir=args.reports_dir,
        with_shap=args.with_shap,
        drift_reference_path=args.drift_reference,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
