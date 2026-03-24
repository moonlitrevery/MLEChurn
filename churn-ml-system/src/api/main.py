"""FastAPI service: load joblib pipeline at startup, expose /predict."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from sklearn.pipeline import Pipeline

from api.schemas import HealthResponse, PredictRequest, PredictResponse
from common.logging_config import setup_logging
from inference.predictor import load_churn_pipeline, predict_churn_proba
from models.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

logger = logging.getLogger(__name__)

REQUIRED_FEATURE_COLUMNS: frozenset[str] = frozenset(NUMERIC_COLUMNS) | frozenset(
    CATEGORICAL_COLUMNS
)


def _required_columns_message(missing: set[str]) -> str:
    return f"Missing required feature columns: {sorted(missing)}."


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    project_root = os.environ.get("CHURN_PROJECT_ROOT")
    path = os.environ.get("CHURN_MODEL_PATH")
    logger.info("Loading churn pipeline (project_root=%s)", project_root)
    app.state.model = load_churn_pipeline(
        path if path else None,
        project_root=project_root,
    )
    logger.info("Pipeline ready.")
    yield


app = FastAPI(
    title="Churn scoring API",
    version="0.1.0",
    lifespan=lifespan,
)


def get_model(request: Request) -> Pipeline:
    model: Any = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return model


@app.get("/health", response_model=HealthResponse)
def health(model: Annotated[Pipeline, Depends(get_model)]) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=True)


@app.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    model: Annotated[Pipeline, Depends(get_model)],
) -> PredictResponse:
    df = pd.DataFrame(body.records)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    missing = REQUIRED_FEATURE_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=_required_columns_message(missing))

    try:
        proba = predict_churn_proba(model, df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(churn_probability=proba.tolist())
