"""학습 작업 관리 (단일 사용자 MVP: 동시 1건, 스레드 실행, SSE 큐)."""
from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import tensorflow as tf

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main_pipeline import SolarPowerPredictionPipeline  # noqa: E402

logger = logging.getLogger("solar.training")

SENTINEL: Dict[str, Any] = {"event": "__close__", "data": {}}


@dataclass
class TrainingJob:
    run_id: str
    config: Dict[str, Any]
    status: str = "pending"  # pending|running|completed|failed
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    total_epochs: int = 0
    last_epoch: int = 0
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2048))


class StreamingCallback(tf.keras.callbacks.Callback):
    """Keras 콜백 → TrainingJob.events 큐에 epoch 이벤트 push."""

    def __init__(self, job: TrainingJob):
        super().__init__()
        self.job = job

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None):
        logs = logs or {}
        self.job.last_epoch = epoch + 1
        payload = {
            "epoch": epoch + 1,
            "total": self.job.total_epochs,
            "loss": float(logs.get("loss", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)) if "val_loss" in logs else None,
            "mae": float(logs.get("mae", 0.0)) if "mae" in logs else None,
            "val_mae": float(logs.get("val_mae", 0.0)) if "val_mae" in logs else None,
            "lr": float(logs.get("lr", 0.0)) if "lr" in logs else None,
        }
        self._safe_put({"event": "epoch", "data": payload})

    def _safe_put(self, item: Dict[str, Any]):
        try:
            self.job.events.put_nowait(item)
        except queue.Full:
            logger.warning("이벤트 큐 가득 — 일부 epoch 이벤트 드롭")


class TrainingManager:
    """싱글톤. 동시 1건의 학습만 허용."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current: Optional[TrainingJob] = None
        self._thread: Optional[threading.Thread] = None
        self._on_complete = None  # type: Optional[callable]

    def set_on_complete(self, fn):
        """학습 정상 종료 후 호출 (모델 hot-swap 등)."""
        self._on_complete = fn

    @property
    def current(self) -> Optional[TrainingJob]:
        return self._current

    def start(self, config_overrides: Dict[str, Any]) -> TrainingJob:
        with self._lock:
            if self._current and self._current.status == "running":
                raise RuntimeError("이미 학습이 진행 중입니다")

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = SolarPowerPredictionPipeline().config
            merged: Dict[str, Any] = {**base, **config_overrides}
            merged["verbose"] = 0  # 백엔드에선 콘솔 진행바 끄기

            job = TrainingJob(
                run_id=run_id,
                config=merged,
                status="pending",
                total_epochs=int(merged.get("epochs", 100)),
            )
            self._current = job

            thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            self._thread = thread
            thread.start()
            return job

    def _run(self, job: TrainingJob):
        job.status = "running"
        job.started_at = datetime.now().isoformat()
        job.events.put({"event": "start", "data": {"run_id": job.run_id, "config": job.config}})

        try:
            pipeline = SolarPowerPredictionPipeline(job.config)
            cb = StreamingCallback(job)
            pipeline.run_full_pipeline(extra_callbacks=[cb])

            metrics = pipeline.results.get("metrics", {}) if pipeline.results else {}
            # numpy 타입 → float 변환
            clean_metrics = {}
            for k, v in metrics.items():
                try:
                    clean_metrics[k] = float(v)
                except (TypeError, ValueError):
                    clean_metrics[k] = None
            job.metrics = clean_metrics
            job.status = "completed"
            job.finished_at = datetime.now().isoformat()
            job.events.put(
                {"event": "done", "data": {"run_id": job.run_id, "metrics": clean_metrics}}
            )

            if self._on_complete:
                try:
                    self._on_complete(job)
                except Exception:
                    logger.exception("on_complete 콜백 실패 (무시하고 계속)")

        except Exception as e:
            logger.exception("학습 실패")
            job.status = "failed"
            job.error = str(e)
            job.finished_at = datetime.now().isoformat()
            job.events.put({"event": "error", "data": {"message": str(e)}})
        finally:
            job.events.put(SENTINEL)


_manager = TrainingManager()


def get_manager() -> TrainingManager:
    return _manager


def drain_events(job: TrainingJob, idle_timeout: float = 30.0):
    """SSE 제너레이터에서 사용. SENTINEL을 만나면 종료."""
    while True:
        try:
            item = job.events.get(timeout=idle_timeout)
        except queue.Empty:
            # keep-alive (SSE 클라이언트 연결 유지)
            yield {"event": "keepalive", "data": {"ts": time.time()}}
            continue
        if item.get("event") == "__close__":
            return
        yield item
