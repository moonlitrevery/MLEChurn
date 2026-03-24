"""Operational monitoring: drift detection and model explanations."""

from __future__ import annotations

from typing import Any

__all__ = [
    "compute_numeric_drift_report",
    "explain_global",
    "explain_instance",
]


def __getattr__(name: str) -> Any:
    if name == "compute_numeric_drift_report":
        from src.monitoring.data_drift import compute_numeric_drift_report

        return compute_numeric_drift_report
    if name in ("explain_global", "explain_instance"):
        from src.monitoring.explainability import explain_global, explain_instance

        return {"explain_global": explain_global, "explain_instance": explain_instance}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
