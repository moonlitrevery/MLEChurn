from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.models.design_matrix import ChurnTableToDesignMatrixTransformer
from src.models.preprocessing import CoerceNumericColumns, DropColumns


def build_lgbm_churn_pipeline(
    *,
    random_state: int = 42,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    min_child_samples: int = 20,
    lgbm_n_jobs: int = -1,
) -> Pipeline:
    """
    Sequential pipeline: table prep → design matrix (engineering + encoding) → LightGBM.

    For reproducible metrics with thread parallelism, set ``lgbm_n_jobs: 1`` in config;
    ``random_state`` is always passed through to the booster.
    """
    return Pipeline(
        steps=[
            ("drop_id", DropColumns()),
            ("coerce_numeric", CoerceNumericColumns()),
            ("design_matrix", ChurnTableToDesignMatrixTransformer()),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    min_child_samples=min_child_samples,
                    random_state=random_state,
                    n_jobs=lgbm_n_jobs,
                    verbose=-1,
                ),
            ),
        ]
    )
