from __future__ import annotations

ID_COLUMN = "id"
TARGET_COLUMN = "Churn"

NUMERIC_COLUMNS: tuple[str, ...] = (
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)
