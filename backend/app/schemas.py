"""Pydantic 응답 스키마."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    n_features: Optional[int] = None
    sequence_length: Optional[int] = None
    test_samples: Optional[int] = None


class PredictionPoint(BaseModel):
    index: int
    actual: float
    predicted: float
    residual: float


class RecentPredictionResponse(BaseModel):
    count: int
    metrics: dict = Field(..., description="rmse, mae, r2")
    points: List[PredictionPoint]
