"""FastAPI entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .deps import load_app_state
from .routers import predict as predict_router
from .routers import runs as runs_router
from .routers import train as train_router
from .training import get_manager

logger = logging.getLogger("solar.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("모델 + 데이터 로드 시작 (lifespan)")
    state = load_app_state()
    app.state.ml = state
    logger.info(
        "로드 완료: model=%s, test_samples=%d, n_features=%d",
        state.model_path,
        state.X_test.shape[0],
        state.n_features,
    )

    def _on_train_complete(job):
        """학습 완료 시 best_model.keras를 메모리에 swap."""
        try:
            new_state = load_app_state()
            app.state.ml = new_state
            logger.info("모델 hot-swap 완료 (run_id=%s)", job.run_id)
        except Exception:
            logger.exception("hot-swap 실패 — 이전 모델 유지")

    get_manager().set_on_complete(_on_train_complete)

    yield
    logger.info("shutdown")


app = FastAPI(title="Solar Power Prediction API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(predict_router.router, prefix="/api")
app.include_router(runs_router.router, prefix="/api")
app.include_router(train_router.router, prefix="/api")
