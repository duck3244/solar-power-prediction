# 🏗️ 시스템 아키텍처

CNN-LSTM 기반 태양광 발전 / 전력 수요 예측 시스템의 전체 아키텍처 문서입니다.
백엔드(FastAPI + TensorFlow)와 프론트엔드(React + Vite)가 분리된 풀스택 구조이며, ML 파이프라인은 백엔드 내부 모듈로 임베딩되어 있습니다.

---

## 1. 개요

| 항목 | 내용 |
|------|------|
| 도메인 | 태양광 발전량 + 기상 데이터를 활용한 단기 전력 수요 예측 |
| 모델 | CNN-LSTM 하이브리드 (Basic / Advanced / Transformer) |
| 백엔드 | Python 3.9, FastAPI, TensorFlow/Keras |
| 프론트엔드 | React 18, TypeScript, Vite, TailwindCSS, Recharts |
| 통신 | REST (JSON) + Server-Sent Events (학습 진행률 스트리밍) |
| 배포 형태 | 단일 사용자 MVP (인메모리 모델 캐시, 로컬 파일 기반 결과 저장) |

---

## 2. 컴포넌트 다이어그램

```
┌────────────────────────────────────────────────────────────────────────┐
│                          User (Browser)                                │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ HTTP / SSE
                               ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Frontend (Vite Dev Server :5173 / Static build)                       │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  React SPA (BrowserRouter)                                       │  │
│  │  ┌──────────┐  ┌─────────┐  ┌────────┐  ┌─────────┐              │  │
│  │  │Dashboard │  │ Predict │  │ Train  │  │ History │              │  │
│  │  └────┬─────┘  └────┬────┘  └───┬────┘  └────┬────┘              │  │
│  │       └────────────┴───────┬────┴────────────┘                   │  │
│  │                            ▼                                     │  │
│  │                  lib/api.ts (fetch wrapper)                      │  │
│  └─────────────────────────────┬────────────────────────────────────┘  │
└────────────────────────────────┼───────────────────────────────────────┘
                                 │ /api/* (proxy)
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI :8000)                                               │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  app/main.py  — FastAPI app, CORS, lifespan, router 등록       │    │
│  └───────┬─────────────────┬────────────────────┬─────────────────┘    │
│          ▼                 ▼                    ▼                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐       │
│  │ predict.py   │  │ train.py     │  │ runs.py                 │       │
│  │ /api/health  │  │ /api/train   │  │ /api/runs               │       │
│  │ /api/predict │  │ /api/train/  │  │ /api/runs/{id}          │       │
│  │   /recent    │  │   status     │  │                         │       │
│  │              │  │ /api/train/  │  │                         │       │
│  │              │  │   stream/{id}│  │                         │       │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬────────────┘       │
│         │                 │                       │                    │
│         │                 ▼                       │                    │
│         │       ┌────────────────────┐            │                    │
│         │       │ training.py        │            │                    │
│         │       │  TrainingManager   │            │                    │
│         │       │  (싱글톤, 1건)     │            │                    │
│         │       │  StreamingCallback │            │                    │
│         │       └─────────┬──────────┘            │                    │
│         │                 │                       │                    │
│         ▼                 ▼                       ▼                    │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  app/deps.py — AppState (model, engineer, X_test 인메모리)   │      │
│  └────────────────────────────┬─────────────────────────────────┘      │
└───────────────────────────────┼────────────────────────────────────────┘
                                ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ML Pipeline (backend/*.py)                                            │
│                                                                        │
│  ┌─────────────────┐   ┌────────────────────┐   ┌───────────────────┐  │
│  │ data_loader.py  │──▶│ feature_engineer.py│──▶│ cnn_lstm_model.py │  │
│  │ SolarDataLoader │   │ SolarFeatureEngineer│  │ CNNLSTMBuilder    │  │
│  │                 │   │                    │   │ ModelEnsemble     │  │
│  └─────────────────┘   └────────────────────┘   └─────────┬─────────┘  │
│                                                           ▼            │
│  ┌─────────────────────┐   ┌────────────────────────────────────────┐  │
│  │ visualizer.py       │◀──│ model_trainer.py                       │  │
│  │ ResultVisualizer    │   │ ModelTrainer / HyperparameterTuner     │  │
│  └─────────────────────┘   └────────────────────────────────────────┘  │
│                                          │                             │
│                                          ▼                             │
│             ┌────────────────────────────────────────┐                 │
│             │ main_pipeline.py                       │                 │
│             │ SolarPowerPredictionPipeline (오케스트)│                 │
│             └────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────┘
                                ▲              ▲
                                │              │
┌───────────────────────────────┴──────────────┴─────────────────────────┐
│  Storage (Local FS)                                                    │
│  backend/                                                              │
│    ├── Plant_1_Generation_Data.csv      (입력: 발전량)                 │
│    ├── Plant_1_Weather_Sensor_Data.csv  (입력: 기상)                   │
│    └── results/                                                        │
│         ├── best_model.keras            (hot-swap 대상)                │
│         ├── final_model_<ts>.keras                                     │
│         ├── summary_<ts>.json           (run 메타 + 성능)              │
│         └── results_<ts>_history_<ts>.csv (epoch 학습 곡선)            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 레이어별 책임

### 3.1 Frontend Layer (`frontend/`)

| 파일/디렉토리 | 책임 |
|---------------|------|
| `src/App.tsx` | `BrowserRouter`로 4개 페이지 라우팅 + `NavBar` 마운트 |
| `src/components/NavBar.tsx` | 상단 네비게이션 (Dashboard / Predict / Train / History) |
| `src/components/MetricCard.tsx` | RMSE / MAE / R² 등 지표 카드 컴포넌트 |
| `src/pages/DashboardPage.tsx` | 시스템 헬스 + 최근 예측 vs 실제 차트 (Recharts) |
| `src/pages/PredictPage.tsx` | 단일 예측 조회 / 결과 표 |
| `src/pages/TrainPage.tsx` | 학습 트리거 + SSE 진행률 실시간 표시 |
| `src/pages/HistoryPage.tsx` | 과거 run 목록 + 학습 곡선 |
| `src/lib/api.ts` | `/api/*` fetch 래퍼, 타입 정의, SSE URL 빌더 |

**기술 스택**: React 18 + TypeScript, Vite, TailwindCSS, Recharts, react-router-dom v6.
**개발 서버**: Vite (포트 5173). `/api` 경로는 백엔드(8000)로 프록시.

### 3.2 API Layer (`backend/app/`)

| 파일 | 책임 |
|------|------|
| `main.py` | FastAPI 앱 생성, CORS, lifespan(모델 로드/hot-swap), 라우터 등록 |
| `deps.py` | `AppState` 정의 + `load_app_state()` — 모델 + 피처 파이프라인 1회 로드 |
| `schemas.py` | Pydantic 응답 모델 (`HealthResponse`, `PredictionPoint`, …) |
| `training.py` | `TrainingManager` (싱글톤, 동시 1건), `TrainingJob`, `StreamingCallback`, SSE 큐 |
| `routers/predict.py` | `/api/health`, `/api/predict/recent` |
| `routers/train.py` | `/api/train` (POST), `/api/train/status`, `/api/train/stream/{run_id}` (SSE) |
| `routers/runs.py` | `/api/runs`, `/api/runs/{run_id}` — `results/` 파일을 파싱하여 노출 |

### 3.3 ML Pipeline Layer (`backend/*.py`)

| 모듈 | 핵심 클래스 | 역할 |
|------|-------------|------|
| `data_loader.py` | `SolarDataLoader` | CSV 로드, 시간별 집계, 발전/기상 병합, 합성 데이터 생성 |
| `feature_engineer.py` | `SolarFeatureEngineer` | 시간/순환/래그/롤링/효율성/상호작용 특성 + 시퀀스 생성 + 스케일러 |
| `cnn_lstm_model.py` | `CNNLSTMBuilder`, `CustomLosses`, `ModelEnsemble` | Basic / Advanced(Bi-LSTM+Attention) / Transformer 모델 빌더 |
| `model_trainer.py` | `ModelTrainer`, `HyperparameterTuner` | 학습/평가/CV/그리드 서치, 학습 곡선·잔차 시각화 |
| `visualizer.py` | `ResultVisualizer` | 성능 대시보드, 인터랙티브 Plotly 대시보드 |
| `main_pipeline.py` | `SolarPowerPredictionPipeline` | 전체 파이프라인 오케스트레이터(CLI + 라이브러리) |

### 3.4 Storage Layer

파일 시스템 기반(데이터베이스 없음).
- **입력**: `backend/Plant_*_Generation_Data.csv`, `backend/Plant_*_Weather_Sensor_Data.csv`
- **출력**: `backend/results/` 아래 학습 결과물
  - `best_model.keras` — 백엔드가 hot-swap 대상으로 사용
  - `summary_<run_id>.json` — `runs.py`가 파싱해 UI에 노출
  - `results_<run_id>_history_<run_id>.csv` — 에폭별 학습 곡선

---

## 4. 핵심 시퀀스 / 데이터 흐름

### 4.1 부팅 (lifespan)
1. `uvicorn`이 `app.main:app` 기동.
2. `lifespan`에서 `load_app_state()` 호출 → `SolarDataLoader` + `SolarFeatureEngineer` 실행 후 test split + best model 로드.
3. 결과를 `app.state.ml` (`AppState`)에 캐싱.
4. `TrainingManager`에 `on_complete` 콜백 등록 → 학습 완료 시 자동 hot-swap.

### 4.2 예측 (`GET /api/predict/recent?n=…`)
1. `app.state.ml` 에서 `X_test` 마지막 n개 시퀀스 슬라이스.
2. `model.predict(X)` → 정규화된 예측.
3. `engineer.target_scaler.inverse_transform()` 으로 역변환.
4. RMSE/MAE/R² 계산 후 `RecentPredictionResponse`로 응답.

### 4.3 학습 + SSE 스트리밍
```
Client                Backend                              ML Pipeline
  │                     │                                       │
  │ POST /api/train ───▶│ TrainingManager.start(overrides)      │
  │                     │   └─ TrainingJob 생성 (run_id)        │
  │                     │   └─ Thread: pipeline.run_full(...)──▶│ data→feature→build→train
  │ ◀── job status      │                                       │
  │                     │                            (on_epoch_end)
  │ GET /api/train/     │                            StreamingCallback
  │   stream/{run_id}──▶│ drain_events(job)                     │
  │ ◀── SSE: epoch ─────│ ◀───── queue.Queue ───────────────────│
  │ ◀── SSE: epoch ─────│                                       │
  │ ◀── SSE: done ──────│  on_complete → load_app_state(swap)   │
  │                     │  app.state.ml = new_state             │
```

핵심 동기화 메커니즘:
- `threading.Lock`으로 동시 학습 1건 보장.
- `queue.Queue(maxsize=2048)`로 epoch 이벤트 백프레셔.
- `SENTINEL` 메시지로 SSE 정상 종료.
- 클라이언트 disconnect 감지(`request.is_disconnected()`)로 핸들러 조기 종료.

### 4.4 Run 이력 조회
- `runs.py`는 DB가 아닌 `backend/results/summary_*.json` 파일을 glob → 정렬해 응답.
- `/api/runs/{id}` 호출 시 학습 곡선 CSV를 추가로 파싱(`history`) + 전체 config 포함.

---

## 5. 설계 결정 & 트레이드오프

| 결정 | 이유 | 제약 |
|------|------|------|
| 단일 사용자 MVP, 인메모리 캐시 | 단순화, 빠른 응답 | 다중 인스턴스 스케일아웃 불가 |
| 동시 1건 학습 제한 | GPU/메모리 보호, 상태 단순화 | 큐잉/대기열 없음 (409 응답) |
| 결과 저장: 파일 시스템 | 외부 DB 의존성 제거 | 동시 쓰기/잠금 보장 없음 |
| SSE (vs WebSocket) | 단방향 진행률 전송에 충분, HTTP 호환 | 양방향 제어 불가 |
| TF Keras Callback → Queue → SSE | Keras 동기 콜백을 asyncio와 분리 | 큐 가득 시 epoch 드롭 가능 |
| `sys.path.insert` 로 backend 루트 추가 | 기존 ML 모듈을 그대로 재사용 | 패키지 구조 비표준 |

---

## 6. 외부 인터페이스 요약

### REST 엔드포인트
| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/health` | 모델/데이터 로드 상태 |
| GET | `/api/predict/recent?n=` | 최근 n개 시퀀스 예측 vs 실제 |
| POST | `/api/train` | 학습 시작 (overrides) |
| GET | `/api/train/status` | 현재 학습 상태 |
| GET | `/api/train/stream/{run_id}` | SSE 진행률 스트림 |
| GET | `/api/runs` | 과거 run 목록 |
| GET | `/api/runs/{run_id}` | run 상세 + 학습 곡선 |

### CORS
- 허용 origin: `http://localhost:5173`, `http://127.0.0.1:5173`
- 허용 method: `GET`, `POST`

---

## 7. 디렉토리 구조 (전체)

```
solar-power-prediction/
├── backend/
│   ├── app/                       # FastAPI 애플리케이션
│   │   ├── main.py
│   │   ├── deps.py
│   │   ├── schemas.py
│   │   ├── training.py
│   │   └── routers/
│   │       ├── predict.py
│   │       ├── train.py
│   │       └── runs.py
│   ├── data_loader.py             # ML 파이프라인
│   ├── feature_engineer.py
│   ├── cnn_lstm_model.py
│   ├── model_trainer.py
│   ├── visualizer.py
│   ├── main_pipeline.py
│   ├── run_example.py
│   ├── config_sample.json
│   ├── requirements.txt
│   ├── Plant_*.csv                # 입력 데이터
│   └── results/                   # 학습 산출물
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── components/
│   │   ├── pages/
│   │   └── lib/api.ts
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── docs/
│   ├── ARCHITECTURE.md            # 본 문서
│   └── UML.md                     # UML 다이어그램
├── fig/
├── README.md
└── LICENSE
```
