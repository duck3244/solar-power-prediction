"""학습 트리거 + SSE 진행률 라우터."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..training import drain_events, get_manager

router = APIRouter(tags=["train"])
logger = logging.getLogger("solar.api.train")


class TrainRequest(BaseModel):
    epochs: Optional[int] = Field(None, ge=1, le=500)
    batch_size: Optional[int] = Field(None, ge=1, le=512)
    learning_rate: Optional[float] = Field(None, gt=0, le=1.0)
    model_type: Optional[str] = Field(None, pattern="^(basic|advanced|transformer)$")
    dropout_rate: Optional[float] = Field(None, ge=0.0, le=0.9)
    early_stopping_patience: Optional[int] = Field(None, ge=1, le=100)

    def overrides(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


def _job_to_status(job) -> Dict[str, Any]:
    return {
        "run_id": job.run_id,
        "status": job.status,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "last_epoch": job.last_epoch,
        "total_epochs": job.total_epochs,
        "error": job.error,
        "metrics": job.metrics,
    }


@router.post("/train")
def start_train(req: TrainRequest):
    manager = get_manager()
    try:
        job = manager.start(req.overrides())
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return _job_to_status(job)


@router.get("/train/status")
def train_status():
    job = get_manager().current
    if job is None:
        return {"status": "idle"}
    return _job_to_status(job)


def _sse_format(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/train/stream/{run_id}")
async def train_stream(run_id: str, request: Request):
    manager = get_manager()
    job = manager.current
    if job is None or job.run_id != run_id:
        raise HTTPException(404, f"run_id {run_id} 미존재 또는 종료")

    async def event_gen():
        # 큐를 별도 스레드에서 polling → SSE 라인으로 변환
        loop = asyncio.get_running_loop()
        gen = drain_events(job, idle_timeout=15.0)
        while True:
            if await request.is_disconnected():
                logger.info("SSE 클라이언트 연결 해제")
                return
            try:
                item = await loop.run_in_executor(None, next, gen, None)
            except StopIteration:
                return
            if item is None:
                return
            yield _sse_format(item["event"], item["data"])
            if item["event"] in ("done", "error"):
                # 마지막 이벤트 송신 후 종료
                return

    return StreamingResponse(event_gen(), media_type="text/event-stream")
