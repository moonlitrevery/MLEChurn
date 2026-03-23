from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from features.engineering import ChurnFeatureEngineer
from models.preprocessing import CoerceNumericColumns, DropColumns
from models.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_lgbm_churn_pipeline(
    *,
    random_state: int = 42,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    lgbm_n_jobs: int = -1,
) -> Pipeline:
    tabular = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                list(NUMERIC_COLUMNS),
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                list(CATEGORICAL_COLUMNS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    tabular.set_output(transform="pandas")

    union = FeatureUnion(
        transformer_list=[
            ("engineered", ChurnFeatureEngineer()),
            ("tabular", tabular),
        ],
    )
    union.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("drop_id", DropColumns()),
            ("coerce_numeric", CoerceNumericColumns()),
            ("features", union),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=0.0,
                    random_state=random_state,
                    n_jobs=lgbm_n_jobs,
                    verbose=-1,
                ),
            ),
        ]
    )
