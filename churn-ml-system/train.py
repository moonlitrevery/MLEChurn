#!/usr/bin/env python3
"""Train churn pipeline: load data, stratified CV, fit full model, joblib dump."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib

# Repo layout: churn-ml-system/train.py, churn-ml-system/src/{data,features,models}
_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC = _PACKAGE_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.loading import load_train_data  # noqa: E402
from models.pipeline import build_lgbm_churn_pipeline  # noqa: E402
from models.schema import TARGET_COLUMN  # noqa: E402
from models.training import (  # noqa: E402
    cross_validate_stratified_roc_auc,
    fit_pipeline,
    prepare_churn_target,
)

DEFAULT_MODEL_REL = Path("models") / "churn_pipeline.joblib"


def _default_project_root() -> Path | None:
    raw = os.environ.get("CHURN_PROJECT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project-root",
        type=Path,
        default=_default_project_root(),
        help="Anchor for relative data paths (default: CHURN_PROJECT_ROOT or cwd).",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory segment or path (default: CHURN_DATA_DIR or 'datasets').",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_PACKAGE_ROOT / DEFAULT_MODEL_REL,
        help=f"joblib path for fitted pipeline (default: {DEFAULT_MODEL_REL}).",
    )
    p.add_argument("--n-splits", type=int, default=5, help="StratifiedKFold folds.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional max rows for faster dev runs (passed to pandas read_csv).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else None
    )

    read_kw = {}
    if args.nrows is not None:
        read_kw["nrows"] = args.nrows

    df = load_train_data(
        args.data_dir,
        project_root=project_root,
        **read_kw,
    )
    if TARGET_COLUMN not in df.columns:
        raise SystemExit(f"Column {TARGET_COLUMN!r} missing from training frame.")

    y = prepare_churn_target(df[TARGET_COLUMN])
    X = df.drop(columns=[TARGET_COLUMN])

    pipeline = build_lgbm_churn_pipeline(
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    cv = cross_validate_stratified_roc_auc(
        pipeline,
        X,
        y,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    fitted = fit_pipeline(pipeline, X, y)

    out_path = args.output.expanduser()
    out_path = (
        (_PACKAGE_ROOT / out_path).resolve()
        if not out_path.is_absolute()
        else out_path.resolve()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, out_path)

    print(f"Saved pipeline to {out_path}")
    print(f"CV mean ROC-AUC: {cv['mean_roc_auc']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
