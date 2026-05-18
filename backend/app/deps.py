"""모델/데이터 로딩 의존성. lifespan에서 1회 실행되어 app.state에 캐시된다."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

# backend/ 를 sys.path에 추가하여 기존 모듈 임포트
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from data_loader import SolarDataLoader  # noqa: E402
from feature_engineer import SolarFeatureEngineer  # noqa: E402


@dataclass
class AppState:
    model: tf.keras.Model
    engineer: SolarFeatureEngineer
    X_test: np.ndarray
    y_test_norm: np.ndarray  # 정규화된 상태 (역정규화 시 사용)
    n_features: int
    sequence_length: int
    model_path: str


def load_app_state(
    generation_csv: str = "Plant_1_Generation_Data.csv",
    weather_csv: str = "Plant_1_Weather_Sensor_Data.csv",
    model_path: str = "results/best_model.keras",
    sequence_length: int = 24,
    random_seed: int = 42,
) -> AppState:
    """Feature pipeline + 모델 로드. 단일 사용자 MVP용 인메모리 캐시."""
    # 재현성: create_target_variable 내부 np.random 노이즈 고정
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    backend_dir = Path(__file__).resolve().parent.parent
    gen_path = backend_dir / generation_csv
    wx_path = backend_dir / weather_csv
    mdl_path = backend_dir / model_path

    if not mdl_path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {mdl_path}. main_pipeline.py로 먼저 학습하세요."
        )

    loader = SolarDataLoader()
    df = loader.preprocess_pipeline(str(gen_path), str(wx_path))

    engineer = SolarFeatureEngineer()
    splits, _, _ = engineer.feature_engineering_pipeline(
        df, sequence_length=sequence_length
    )

    model = tf.keras.models.load_model(str(mdl_path), compile=False)

    return AppState(
        model=model,
        engineer=engineer,
        X_test=splits["X_test"],
        y_test_norm=splits["y_test"],
        n_features=splits["X_test"].shape[2],
        sequence_length=sequence_length,
        model_path=str(mdl_path),
    )
