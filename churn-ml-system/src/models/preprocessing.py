"""Lightweight table prep before FeatureUnion (sklearn-compatible)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.models.schema import ID_COLUMN, NUMERIC_COLUMNS


class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns: tuple[str, ...] | None = None) -> None:
        self.columns = (ID_COLUMN,) if columns is None else tuple(columns)

    def fit(self, X: Any, y: Any = None) -> DropColumns:
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__} expects a pandas DataFrame.")
        to_drop = [c for c in self.columns if c in X.columns]
        return X.drop(columns=to_drop, axis=1).copy()


class CoerceNumericColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns: tuple[str, ...] | None = None) -> None:
        self.columns = NUMERIC_COLUMNS if columns is None else tuple(columns)

    def fit(self, X: Any, y: Any = None) -> CoerceNumericColumns:
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__} expects a pandas DataFrame.")
        out = X.copy()
        for col in self.columns:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out
