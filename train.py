#!/usr/bin/env python3
"""Train churn pipeline: YAML defaults, optional Optuna + MLflow, stratified CV, joblib dump."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib

_REPO_ROOT = Path(__file__).resolve().parent


def _ensure_src_on_path() -> None:
    """Package root is the parent of the ``src`` directory (import ``src.*``)."""
    candidates = [
        _REPO_ROOT / "churn-ml-system",
        _REPO_ROOT,
    ]
    for base in candidates:
        if (base / "src" / "models").is_dir():
            root = str(base.resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise SystemExit(
        "Cannot locate Python package root. Expected churn-ml-system/src or ./src."
    )


_ensure_src_on_path()

from src.common.logging_config import get_logger, setup_logging  # noqa: E402
from src.config.loader import deep_get, load_train_config  # noqa: E402
from src.data.loading import load_train_data  # noqa: E402
from src.models.pipeline import build_lgbm_churn_pipeline  # noqa: E402
from src.models.schema import TARGET_COLUMN  # noqa: E402
from src.models.training import (  # noqa: E402
    cross_validate_stratified_roc_auc,
    fit_pipeline,
    prepare_churn_target,
)

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


def _build_parser(cfg: dict[str, Any]) -> argparse.ArgumentParser:
    g = lambda *keys, default=None: deep_get(cfg, *keys, default=default)

    p = argparse.ArgumentParser(
        description=__doc__,
        epilog="Use --config path/to.yaml (parsed first via parse_known_args; may appear anywhere in argv).",
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=_default_project_root(),
        help="Anchor for relative paths (default: CHURN_PROJECT_ROOT or cwd).",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=g("paths", "data_dir", default="datasets"),
        help="Data directory segment under project root.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="joblib output path (default: paths.model_output in config, under repo root).",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=g("training", "n_splits", default=5),
        help="StratifiedKFold folds.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=g("random_seed", default=42),
        help="Global seed (CV splits, LightGBM, Optuna sampler).",
    )
    p.add_argument(
        "--cv-shuffle",
        action=argparse.BooleanOptionalAction,
        default=bool(g("training", "cv_shuffle", default=True)),
        help="Shuffle folds (recommended; use --no-cv-shuffle for ordered data).",
    )
    p.add_argument("--n-estimators", type=int, default=g("model", "n_estimators", default=400))
    p.add_argument(
        "--learning-rate", type=float, default=g("model", "learning_rate", default=0.05)
    )
    p.add_argument("--num-leaves", type=int, default=g("model", "num_leaves", default=31))
    p.add_argument("--subsample", type=float, default=g("model", "subsample", default=0.9))
    p.add_argument(
        "--colsample-bytree",
        type=float,
        default=g("model", "colsample_bytree", default=0.9),
    )
    p.add_argument("--reg-alpha", type=float, default=g("model", "reg_alpha", default=0.0))
    p.add_argument("--reg-lambda", type=float, default=g("model", "reg_lambda", default=0.0))
    p.add_argument(
        "--min-child-samples",
        type=int,
        default=g("model", "min_child_samples", default=20),
    )
    p.add_argument(
        "--nrows",
        type=int,
        default=g("data", "nrows", default=None),
        help="Optional row cap for dev runs (None = all rows).",
    )
    p.add_argument(
        "--lgbm-n-jobs",
        type=int,
        default=g("training", "lgbm_n_jobs", default=-1),
        help="LightGBM n_jobs for final fit and baseline CV.",
    )
    p.add_argument(
        "--optuna-trials",
        type=int,
        default=g("optuna", "trials", default=0),
        help="Optuna trials (>0 enables tuning).",
    )
    p.add_argument(
        "--optuna-storage",
        type=str,
        default=g("optuna", "storage", default=None),
        help="Optional Optuna RDB URL.",
    )
    p.add_argument(
        "--optuna-study-name",
        type=str,
        default=g("optuna", "study_name", default=None),
    )
    p.add_argument(
        "--optuna-lgbm-n-jobs",
        type=int,
        default=g("optuna", "lgbm_n_jobs", default=1),
    )
    p.add_argument(
        "--optuna-log-trials-mlflow",
        action="store_true",
        help="Nested MLflow runs per Optuna trial (requires MLflow).",
    )
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow (overrides config mlflow.enabled).",
    )
    p.add_argument(
        "--mlflow-experiment",
        type=str,
        default=os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            g("mlflow", "experiment_name", default="churn"),
        ),
    )
    p.add_argument("--mlflow-run-name", type=str, default=None)
    p.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI")
        or g("mlflow", "tracking_uri", default=None),
    )
    p.add_argument("--log-level", type=str, default=g("logging", "level", default=None))
    p.add_argument("--save-best-params", type=Path, default=None)
    return p


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "configs" / "train.yaml",
    )
    pre_args, argv_rest = pre.parse_known_args()
    cfg = load_train_config(pre_args.config)

    parser = _build_parser(cfg)
    args = parser.parse_args(argv_rest)
    args.config = pre_args.config

    setup_logging(args.log_level)

    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else None
    )

    read_kw: dict[str, Any] = {}
    if args.nrows is not None:
        read_kw["nrows"] = args.nrows

    logger.info(
        "Loading training data (config=%s, project_root=%s, nrows=%s)",
        args.config,
        project_root,
        args.nrows,
    )
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

    use_mlflow = bool(deep_get(cfg, "mlflow", "enabled", default=True)) and not args.no_mlflow
    mlflow_uri = args.mlflow_tracking_uri
    if use_mlflow and not mlflow_uri:
        mlflow_uri = str((_REPO_ROOT / "mlruns").resolve())

    optuna_callbacks: list = []
    if use_mlflow and args.optuna_trials > 0 and args.optuna_log_trials_mlflow:
        optuna_callbacks = _optuna_mlflow_callbacks(tracking_uri=mlflow_uri)

    out_default = deep_get(cfg, "paths", "model_output", default="models/churn_pipeline.joblib")
    out_arg = args.output if args.output is not None else Path(out_default)

    def train_body() -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
        optuna_meta: dict[str, Any] | None = None
        if args.optuna_trials > 0:
            from src.models.tuning import run_hyperparameter_search

            logger.info("Starting Optuna study (%s trials)...", args.optuna_trials)
            study = run_hyperparameter_search(
                X,
                y,
                n_trials=args.optuna_trials,
                n_splits=args.n_splits,
                random_state=args.random_state,
                shuffle=args.cv_shuffle,
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
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
                min_child_samples=args.min_child_samples,
                lgbm_n_jobs=args.lgbm_n_jobs,
            )

        cv = cross_validate_stratified_roc_auc(
            pipeline,
            X,
            y,
            n_splits=args.n_splits,
            random_state=args.random_state,
            shuffle=args.cv_shuffle,
            verbose=True,
        )
        fitted = fit_pipeline(pipeline, X, y)

        out_path = out_arg.expanduser()
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
            from src.tracking.mlflow_run import (
                log_sklearn_pipeline,
                log_training_metrics,
                start_mlflow_run,
            )
        except ImportError as exc:
            logger.error(
                "MLflow enabled but import failed (install requirements): %s",
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
                    "config_path": str(args.config),
                    "n_splits": args.n_splits,
                    "random_state": args.random_state,
                    "cv_shuffle": args.cv_shuffle,
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
        _fitted, _cv, _meta = train_body()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
