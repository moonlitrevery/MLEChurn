#!/usr/bin/env python3
"""Train churn pipeline: optional Optuna tuning, MLflow tracking, stratified CV, joblib dump."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib

# Repo layout: train.py at repo root; packages under churn-ml-system/src/
_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "churn-ml-system" / "src"
if not _SRC.is_dir():
    _SRC = _REPO_ROOT / "src"
if not _SRC.is_dir():
    raise SystemExit(
        "Cannot find source tree: expected churn-ml-system/src or src next to train.py."
    )
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.logging_config import get_logger, setup_logging  # noqa: E402
from data.loading import load_train_data  # noqa: E402
from models.pipeline import build_lgbm_churn_pipeline  # noqa: E402
from models.schema import TARGET_COLUMN  # noqa: E402
from models.training import (  # noqa: E402
    cross_validate_stratified_roc_auc,
    fit_pipeline,
    prepare_churn_target,
)
DEFAULT_MODEL_REL = Path("models") / "churn_pipeline.joblib"
logger = get_logger("train")


def _default_project_root() -> Path | None:
    raw = os.environ.get("CHURN_PROJECT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return None


def _optuna_mlflow_callbacks(*, tracking_uri: str | None) -> list:
    try:
        from optuna.integration.mlflow import MLflowCallback
    except ImportError:
        try:
            from optuna.integration import MLflowCallback  # type: ignore[no-redef]
        except ImportError:
            logger.warning(
                "Optuna MLflowCallback not available; trial-level MLflow logging disabled."
            )
            return []

    base: dict = {"metric_name": "mean_roc_auc", "create_experiment": False}
    if tracking_uri:
        base["tracking_uri"] = tracking_uri
    try:
        return [MLflowCallback(**base, nest_trials=True)]
    except TypeError:
        return [MLflowCallback(**base)]


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
        default=_REPO_ROOT / DEFAULT_MODEL_REL,
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
    p.add_argument(
        "--lgbm-n-jobs",
        type=int,
        default=-1,
        help="LightGBM n_jobs for final fit and baseline (non-Optuna) CV.",
    )
    p.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="If > 0, run Optuna with this many trials (CV ROC-AUC objective).",
    )
    p.add_argument(
        "--optuna-storage",
        type=str,
        default=None,
        help="Optional Optuna RDB storage URL (e.g. sqlite:///optuna.db).",
    )
    p.add_argument(
        "--optuna-study-name",
        type=str,
        default=None,
        help="Study name when using --optuna-storage.",
    )
    p.add_argument(
        "--optuna-lgbm-n-jobs",
        type=int,
        default=1,
        help="LightGBM n_jobs inside each Optuna trial (use 1 with parallel workers).",
    )
    p.add_argument(
        "--optuna-log-trials-mlflow",
        action="store_true",
        help="Log each Optuna trial to MLflow (nested runs) when MLflow is enabled.",
    )
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow even if installed.",
    )
    p.add_argument(
        "--mlflow-experiment",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "churn"),
        help="MLflow experiment name (default: env MLFLOW_EXPERIMENT_NAME or 'churn').",
    )
    p.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="Optional MLflow run name.",
    )
    p.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI or local ./mlruns).",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override CHURN_LOG_LEVEL (DEBUG, INFO, WARNING, ...).",
    )
    p.add_argument(
        "--save-best-params",
        type=Path,
        default=None,
        help="Optional path to write JSON of best Optuna params after tuning.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(args.log_level)

    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else None
    )

    read_kw = {}
    if args.nrows is not None:
        read_kw["nrows"] = args.nrows

    logger.info("Loading training data (project_root=%s, nrows=%s)", project_root, args.nrows)
    df = load_train_data(
        args.data_dir,
        project_root=project_root,
        **read_kw,
    )
    if TARGET_COLUMN not in df.columns:
        raise SystemExit(f"Column {TARGET_COLUMN!r} missing from training frame.")

    y = prepare_churn_target(df[TARGET_COLUMN])
    X = df.drop(columns=[TARGET_COLUMN])
    logger.info("Training matrix shape: %s", X.shape)

    use_mlflow = not args.no_mlflow
    mlflow_uri = args.mlflow_tracking_uri
    if use_mlflow and not mlflow_uri:
        mlflow_uri = str((_REPO_ROOT / "mlruns").resolve())

    optuna_callbacks = []
    if use_mlflow and args.optuna_trials > 0 and args.optuna_log_trials_mlflow:
        optuna_callbacks = _optuna_mlflow_callbacks(tracking_uri=mlflow_uri)

    def train_body() -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
        optuna_meta: dict[str, Any] | None = None
        if args.optuna_trials > 0:
            from models.tuning import run_hyperparameter_search

            logger.info("Starting Optuna study (%s trials)...", args.optuna_trials)
            study = run_hyperparameter_search(
                X,
                y,
                n_trials=args.optuna_trials,
                n_splits=args.n_splits,
                random_state=args.random_state,
                study_name=args.optuna_study_name,
                storage=args.optuna_storage,
                callbacks=optuna_callbacks,
                lgbm_n_jobs=args.optuna_lgbm_n_jobs,
                show_progress_bar=False,
            )
            best = dict(study.best_params)
            optuna_meta = {
                "best_value": float(study.best_value),
                "best_params": best,
            }
            if args.save_best_params:
                args.save_best_params.parent.mkdir(parents=True, exist_ok=True)
                args.save_best_params.write_text(json.dumps(best, indent=2), encoding="utf-8")
                logger.info("Wrote best params to %s", args.save_best_params)
            pipeline = build_lgbm_churn_pipeline(
                random_state=args.random_state,
                lgbm_n_jobs=args.lgbm_n_jobs,
                **best,
            )
        else:
            pipeline = build_lgbm_churn_pipeline(
                random_state=args.random_state,
                n_estimators=args.n_estimators,
                lgbm_n_jobs=args.lgbm_n_jobs,
            )

        cv = cross_validate_stratified_roc_auc(
            pipeline,
            X,
            y,
            n_splits=args.n_splits,
            random_state=args.random_state,
            verbose=True,
        )
        fitted = fit_pipeline(pipeline, X, y)

        out_path = args.output.expanduser()
        out_path = (
            (_REPO_ROOT / out_path).resolve()
            if not out_path.is_absolute()
            else out_path.resolve()
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted, out_path)
        logger.info("Saved pipeline to %s", out_path)
        logger.info("CV mean ROC-AUC: %.6f", cv["mean_roc_auc"])
        return fitted, cv, optuna_meta

    if use_mlflow:
        import mlflow

        try:
            from tracking.mlflow_run import (
                log_sklearn_pipeline,
                log_training_metrics,
                start_mlflow_run,
            )
        except ImportError as exc:
            logger.error(
                "MLflow is enabled but could not be imported. Install dependencies "
                "(see requirements.txt): %s",
                exc,
            )
            return 1

        with start_mlflow_run(
            args.mlflow_experiment,
            run_name=args.mlflow_run_name,
            tracking_uri=mlflow_uri,
        ):
            mlflow.log_params(
                {
                    "n_splits": args.n_splits,
                    "random_state": args.random_state,
                    "nrows": args.nrows if args.nrows is not None else "all",
                    "optuna_trials": args.optuna_trials,
                    "lgbm_n_jobs": args.lgbm_n_jobs,
                }
            )
            fitted, cv, optuna_meta = train_body()
            if optuna_meta is not None:
                mlflow.log_metric("optuna_best_cv_mean_roc_auc", optuna_meta["best_value"])
                for key, val in optuna_meta["best_params"].items():
                    mlflow.log_param(f"best__{key}", val)
            log_training_metrics(cv)
            try:
                log_sklearn_pipeline(fitted)
            except Exception as exc:
                logger.warning("MLflow model logging skipped: %s", exc)
    else:
        fitted, cv, _optuna_meta = train_body()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
