"""학습 이력(runs) 라우터. backend/results/ 파일을 파싱한다."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BACKEND_DIR / "results"

RUN_ID_PATTERN = re.compile(r"^summary_(\d{8}_\d{6})\.json$")

router = APIRouter(tags=["runs"])


def _safe_load_json(path: Path) -> Optional[dict]:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _config_summary(cfg: dict) -> Dict[str, Any]:
    """전체 config에서 UI에 노출할 핵심 필드만 추출."""
    keys = (
        "model_type",
        "sequence_length",
        "epochs",
        "batch_size",
        "learning_rate",
        "loss",
        "optimizer",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "random_seed",
    )
    return {k: cfg.get(k) for k in keys if k in cfg}


def _list_run_ids() -> List[str]:
    ids: List[str] = []
    for p in RESULTS_DIR.glob("summary_*.json"):
        m = RUN_ID_PATTERN.match(p.name)
        if m:
            ids.append(m.group(1))
    ids.sort(reverse=True)  # 최신 순
    return ids


def _read_run(run_id: str, *, with_history: bool = False) -> Dict[str, Any]:
    summary_path = RESULTS_DIR / f"summary_{run_id}.json"
    summary = _safe_load_json(summary_path)
    if summary is None:
        raise HTTPException(404, f"run {run_id} 없음")

    cfg = summary.get("config", {})
    perf = summary.get("performance", {})
    data_info = summary.get("data_info", {})

    out: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": summary.get("timestamp", run_id),
        "metrics": {
            "rmse": perf.get("RMSE"),
            "mae": perf.get("MAE"),
            "r2": perf.get("R²") or perf.get("R2"),
            "mape": perf.get("MAPE"),
            "direction_accuracy": perf.get("Direction_Accuracy"),
        },
        "config_summary": _config_summary(cfg),
        "data_info": data_info,
    }

    if with_history:
        history_path = RESULTS_DIR / f"results_{run_id}_history_{run_id}.csv"
        out["history"] = _read_history(history_path)
        out["full_config"] = cfg

    return out


def _read_history(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for i, raw in enumerate(reader):
            row: Dict[str, float] = {"epoch": i + 1}
            for k, v in raw.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    continue
            rows.append(row)
    return rows


@router.get("/runs")
def list_runs() -> Dict[str, Any]:
    runs = [_read_run(rid) for rid in _list_run_ids()]
    return {"count": len(runs), "runs": runs}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    if not RUN_ID_PATTERN.match(f"summary_{run_id}.json"):
        raise HTTPException(400, "잘못된 run_id 형식")
    return _read_run(run_id, with_history=True)
