from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

ADDON_SERVICE_COLS: tuple[str, ...] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

STREAMING_COLS: tuple[str, ...] = ("StreamingTV", "StreamingMovies")

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"tenure", "MonthlyCharges", "TotalCharges", "InternetService", "MultipleLines"}
    | set(ADDON_SERVICE_COLS)
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "avg_monthly_from_total",
    "monthly_minus_implied_avg",
    "tenure_is_new",
    "tenure_is_short",
    "tenure_is_long",
    "addon_yes_count",
    "streaming_yes_count",
    "internet_fiber_optic",
    "phone_multiple_lines_yes",
)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X: Any, y: Any = None) -> ChurnFeatureEngineer:
        X_df = self._as_dataframe(X)
        self._check_columns(X_df)
        total = pd.to_numeric(X_df["TotalCharges"], errors="coerce")
        if total.notna().any():
            self.total_charges_median_ = float(np.nanmedian(total.to_numpy(dtype=float)))
        else:
            self.total_charges_median_ = 0.0
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_out_ = np.asarray(OUTPUT_COLUMNS, dtype=object)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        check_is_fitted(self, "total_charges_median_")
        X_df = self._as_dataframe(X)
        self._check_columns(X_df)

        tenure = pd.to_numeric(X_df["tenure"], errors="coerce").fillna(0).astype(float)
        monthly = pd.to_numeric(X_df["MonthlyCharges"], errors="coerce").fillna(0.0)
        total = pd.to_numeric(X_df["TotalCharges"], errors="coerce").fillna(
            self.total_charges_median_
        )

        tenure_safe = tenure.clip(lower=1.0)
        avg_monthly_from_total = total / tenure_safe

        out = pd.DataFrame(index=X_df.index)
        out["avg_monthly_from_total"] = avg_monthly_from_total
        out["monthly_minus_implied_avg"] = monthly - avg_monthly_from_total
        out["tenure_is_new"] = (tenure <= 3).astype(np.float64)
        out["tenure_is_short"] = (tenure <= 12).astype(np.float64)
        out["tenure_is_long"] = (tenure >= 48).astype(np.float64)
        out["addon_yes_count"] = X_df[list(ADDON_SERVICE_COLS)].eq("Yes").sum(axis=1).astype(
            np.float64
        )
        out["streaming_yes_count"] = X_df[list(STREAMING_COLS)].eq("Yes").sum(axis=1).astype(
            np.float64
        )
        out["internet_fiber_optic"] = (X_df["InternetService"] == "Fiber optic").astype(
            np.float64
        )
        out["phone_multiple_lines_yes"] = (X_df["MultipleLines"] == "Yes").astype(np.float64)

        return out.loc[:, list(OUTPUT_COLUMNS)]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    @staticmethod
    def _as_dataframe(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"{ChurnFeatureEngineer.__name__} requires pandas.DataFrame input, got {type(X)!r}."
            )
        return X

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(X.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
