"""Pydantic payloads for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """One or more rows as JSON objects (same feature names as training ``X``)."""

    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Feature rows; must include all numeric and categorical training columns.",
    )

    @field_validator("records")
    @classmethod
    def _non_empty_dicts(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, row in enumerate(v):
            if not isinstance(row, dict):
                raise TypeError(f"records[{i}] must be an object, got {type(row).__name__}.")
            if not row:
                raise ValueError(f"records[{i}] must not be empty.")
        return v


class PredictResponse(BaseModel):
    churn_probability: list[float] = Field(
        ...,
        description="P(churn=1) per record, same order as the request.",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
