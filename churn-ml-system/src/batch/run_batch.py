"""Batch CSV scoring: load rows, attach churn_probability, write CSV."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _ensure_src_path() -> None:
    for base in (_REPO_ROOT / "churn-ml-system", _REPO_ROOT):
        if (base / "src" / "batch").is_dir():
            root = str(base.resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            return


_ensure_src_path()

from src.data.loading import load_csv  # noqa: E402
from src.inference.predictor import load_churn_pipeline, predict_churn_proba  # noqa: E402
from src.models.schema import TARGET_COLUMN  # noqa: E402

logger = logging.getLogger(__name__)


def run_batch_inference(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path | None = None,
    project_root: str | Path | None = None,
    **read_csv_kwargs: Any,
) -> Path:
    """
    Load ``input_path`` CSV, score with the churn pipeline, write CSV with original
    columns plus ``churn_probability``.

    Reuses :func:`src.data.loading.load_csv` and :func:`src.inference.predictor.predict_churn_proba`.
    Drops ``Churn`` if present before scoring (labels must not leak into features).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    df = load_csv(input_path, project_root=project_root, **read_csv_kwargs)
    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")

    model = load_churn_pipeline(model_path, project_root=project_root)
    proba = predict_churn_proba(model, X)

    out = df.copy()
    out["churn_probability"] = proba
    outp = Path(output_path).expanduser()
    if not outp.is_absolute() and project_root is not None:
        from src.data.loading import resolve_project_root

        outp = resolve_project_root(project_root) / outp
    elif not outp.is_absolute():
        outp = Path.cwd() / outp
    outp = outp.resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False)
    logger.info("Wrote %s rows to %s", len(out), outp)
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
        help="Path anchor for relative paths (CHURN_PROJECT_ROOT).",
    )
    args = p.parse_args()
    run_batch_inference(
        args.input,
        args.output,
        model_path=args.model,
        project_root=args.project_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
