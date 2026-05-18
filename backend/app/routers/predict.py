"""예측 라우터."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..schemas import HealthResponse, PredictionPoint, RecentPredictionResponse

router = APIRouter(tags=["predict"])


def _get_state(request: Request):
    state = getattr(request.app.state, "ml", None)
    if state is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다")
    return state


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = getattr(request.app.state, "ml", None)
    if state is None:
        return HealthResponse(status="loading", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_path=state.model_path,
        n_features=state.n_features,
        sequence_length=state.sequence_length,
        test_samples=int(state.X_test.shape[0]),
    )


@router.get("/predict/recent", response_model=RecentPredictionResponse)
def predict_recent(
    request: Request,
    n: int = Query(20, ge=1, le=500, description="최근 시퀀스 수"),
) -> RecentPredictionResponse:
    """test set 마지막 n개 시퀀스에 대한 예측 vs 실제."""
    state = _get_state(request)

    n_use = min(n, state.X_test.shape[0])
    X = state.X_test[-n_use:]
    y_true_norm = state.y_test_norm[-n_use:]

    y_pred_norm = state.model.predict(X, verbose=0)

    target_scaler = state.engineer.target_scaler
    y_true = target_scaler.inverse_transform(y_true_norm).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_norm).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    points = [
        PredictionPoint(
            index=i,
            actual=float(a),
            predicted=float(p),
            residual=float(a - p),
        )
        for i, (a, p) in enumerate(zip(y_true, y_pred))
    ]

    return RecentPredictionResponse(
        count=n_use,
        metrics={"rmse": rmse, "mae": mae, "r2": r2},
        points=points,
    )
