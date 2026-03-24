"""
Assemble the model design matrix: engineered features + encoded raw tabular columns.

This replaces ``FeatureUnion`` with one explicit step so the high-level pipeline reads
``table prep → design matrix → model`` without parallel transformer branches.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.engineering import ChurnFeatureEngineer
from src.models.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def _tabular_column_transformer() -> ColumnTransformer:
    tab = ColumnTransformer(
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
    tab.set_output(transform="pandas")
    return tab


class ChurnTableToDesignMatrixTransformer(BaseEstimator, TransformerMixin):
    """
    Sequential feature assembly:

    1. Domain engineering (:class:`~src.features.engineering.ChurnFeatureEngineer`)
    2. Numeric imputation + one-hot encoding of raw tabular columns
    3. Column-bind both blocks (same row order as input)
    """

    def __init__(self) -> None:
        self.engineer = ChurnFeatureEngineer()
        self.tabular = _tabular_column_transformer()

    def fit(self, X: Any, y: Any = None) -> ChurnTableToDesignMatrixTransformer:
        self.engineer.fit(X, y)
        self.tabular.fit(X)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        engineered = self.engineer.transform(X)
        tabular = self.tabular.transform(X)
        engineered = engineered.reset_index(drop=True)
        tabular = tabular.reset_index(drop=True)
        return pd.concat([engineered, tabular], axis=1)
