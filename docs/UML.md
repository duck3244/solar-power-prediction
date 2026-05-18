# 📐 UML 다이어그램

CNN-LSTM 태양광 발전 예측 시스템의 주요 UML 다이어그램입니다.
모든 다이어그램은 [Mermaid](https://mermaid.js.org/) 문법으로 작성되어 GitHub / VS Code 등에서 즉시 렌더링됩니다.

---

## 1. 클래스 다이어그램 — ML 파이프라인 (backend)

```mermaid
classDiagram
    direction LR

    class SolarDataLoader {
        +raw_data: dict
        +processed_data: DataFrame
        +load_data(gen_file, weather_file)
        +parse_datetime(df, col, format_type)
        +aggregate_hourly_generation(df)
        +aggregate_hourly_weather(df)
        +merge_data(gen_df, weather_df)
        +filter_daytime_data(df, min_power, min_irradiation)
        +generate_synthetic_data()
        +get_data_summary(df, data_type)
        +preprocess_pipeline(gen_file, weather_file) DataFrame
    }

    class SolarFeatureEngineer {
        +feature_scaler
        +target_scaler
        +feature_names: list
        +create_time_features(df)
        +create_cyclical_features(df)
        +create_lag_features(df, columns, lags)
        +create_rolling_features(df, columns, windows)
        +create_power_efficiency_features(df)
        +create_weather_interaction_features(df)
        +create_target_variable(df, pattern, base_demand_kw)
        +select_features(df, method, target_col)
        +fit_scalers(train_df, feature_cols, target_col, method)
        +transform_split(df, feature_cols, target_col)
        +create_sequences(features, target, sequence_length)
        +feature_engineering_pipeline(df, sequence_length) tuple
        +inverse_transform_target(normalized) ndarray
        +transform_new_features(df) ndarray
    }

    class CNNLSTMBuilder {
        +model: Model
        +history: History
        +build_basic_cnn_lstm(input_shape, ...) Model
        +build_advanced_cnn_lstm(input_shape, ...) Model
        +build_transformer_cnn_lstm(input_shape, ...) Model
        +compile_model(model, optimizer, loss, lr) Model
        +create_callbacks(monitor, patience, ...) list
        +get_model_summary(model) str
        +save_model_architecture(model, filepath)
    }

    class CustomLosses {
        <<utility>>
        +weighted_mse(alpha) function
        +quantile_loss(quantile) function
        +focal_mse(gamma) function
    }

    class ModelEnsemble {
        +models: list
        +weights: list
        +add_model(model, weight)
        +predict(X) ndarray
        +save_ensemble(filepath_prefix)
    }

    class ModelTrainer {
        +training_history
        +best_model
        +scalers
        +prepare_train_data(X, y, ratios) dict
        +train_model(model, X_train, y_train, X_val, y_val, ...) History
        +evaluate_model(model, X_test, y_test, scalers) dict
        +cross_validate(builder, X, y, cv_folds) dict
        +plot_training_history(history)
        +plot_predictions(y_true, y_pred)
        +plot_feature_importance(model, names, top_n)
        +save_results(metrics, y_true, y_pred, prefix)
        +load_and_resume_training(path, ...)
    }

    class HyperparameterTuner {
        +model_builder
        +best_params
        +best_score
        +grid_search(X_train, y_train, X_val, y_val, param_grid) dict
    }

    class ResultVisualizer {
        +style: str
        +figsize: tuple
        +plot_performance_dashboard(metrics)
        +plot_predictions(y_true, y_pred, sample_size)
        +plot_training_history(history)
        +plot_data_distribution(raw_data)
        +plot_comprehensive_results(results, processed_data)
        +create_interactive_dashboard(results)
        +save_all_plots(output_dir)
        +plot_model_comparison(comparison_results)
        +plot_residual_analysis(y_true, y_pred)
    }

    class SolarPowerPredictionPipeline {
        +config: dict
        +data_loader: SolarDataLoader
        +feature_engineer: SolarFeatureEngineer
        +model_builder: CNNLSTMBuilder
        +trainer: ModelTrainer
        +visualizer: ResultVisualizer
        +results: dict
        +_default_config() dict
        +load_config(file)
        +save_config(file)
        +run_data_pipeline()
        +run_model_pipeline(extra_callbacks)
        +run_evaluation_pipeline()
        +run_hyperparameter_tuning()
        +run_ensemble_pipeline(n_models)
        +save_all_results()
        +run_full_pipeline(extra_callbacks)
    }

    SolarPowerPredictionPipeline o--> SolarDataLoader
    SolarPowerPredictionPipeline o--> SolarFeatureEngineer
    SolarPowerPredictionPipeline o--> CNNLSTMBuilder
    SolarPowerPredictionPipeline o--> ModelTrainer
    SolarPowerPredictionPipeline o--> ResultVisualizer
    CNNLSTMBuilder ..> CustomLosses : uses
    ModelTrainer ..> CNNLSTMBuilder : uses
    HyperparameterTuner ..> CNNLSTMBuilder : builder
    ModelEnsemble o--> "many" CNNLSTMBuilder : aggregates models
```

---

## 2. 클래스 다이어그램 — API 레이어 (backend/app)

```mermaid
classDiagram
    direction TB

    class FastAPIApp {
        +state.ml: AppState
        +lifespan()
        +include_router(...)
    }

    class AppState {
        +model: tf.keras.Model
        +engineer: SolarFeatureEngineer
        +X_test: ndarray
        +y_test_norm: ndarray
        +n_features: int
        +sequence_length: int
        +model_path: str
    }

    class HealthResponse {
        +status: str
        +model_loaded: bool
        +model_path: str?
        +n_features: int?
        +sequence_length: int?
        +test_samples: int?
    }

    class PredictionPoint {
        +index: int
        +actual: float
        +predicted: float
        +residual: float
    }

    class RecentPredictionResponse {
        +count: int
        +metrics: dict
        +points: List~PredictionPoint~
    }

    class TrainRequest {
        +epochs: int?
        +batch_size: int?
        +learning_rate: float?
        +model_type: str?
        +dropout_rate: float?
        +early_stopping_patience: int?
        +overrides() dict
    }

    class TrainingJob {
        +run_id: str
        +config: dict
        +status: str
        +started_at: str?
        +finished_at: str?
        +error: str?
        +metrics: dict?
        +total_epochs: int
        +last_epoch: int
        +events: Queue
    }

    class TrainingManager {
        <<singleton>>
        -_lock: Lock
        -_current: TrainingJob?
        -_thread: Thread?
        -_on_complete: callable?
        +set_on_complete(fn)
        +current TrainingJob?
        +start(overrides) TrainingJob
        -_run(job)
    }

    class StreamingCallback {
        +job: TrainingJob
        +on_epoch_end(epoch, logs)
        -_safe_put(item)
    }

    class PredictRouter {
        +health(request) HealthResponse
        +predict_recent(request, n) RecentPredictionResponse
    }

    class TrainRouter {
        +start_train(req) dict
        +train_status() dict
        +train_stream(run_id, request) StreamingResponse
    }

    class RunsRouter {
        +list_runs() dict
        +get_run(run_id) dict
        -_read_run(run_id, with_history)
        -_read_history(path)
    }

    FastAPIApp --> AppState : caches in state
    FastAPIApp --> PredictRouter
    FastAPIApp --> TrainRouter
    FastAPIApp --> RunsRouter
    AppState --> SolarFeatureEngineer
    PredictRouter ..> AppState
    PredictRouter ..> RecentPredictionResponse : returns
    PredictRouter ..> HealthResponse : returns
    RecentPredictionResponse o--> "many" PredictionPoint
    TrainRouter ..> TrainingManager
    TrainRouter ..> TrainRequest : validates
    TrainingManager o--> TrainingJob : owns 1
    TrainingManager ..> StreamingCallback : creates
    StreamingCallback --> TrainingJob : writes events
    TrainRouter ..> TrainingJob : streams events
```

---

## 3. 컴포넌트 다이어그램 (Frontend)

```mermaid
classDiagram
    direction LR

    class App {
        +render()
    }
    class NavBar {
        +links: Dashboard/Predict/Train/History
    }
    class DashboardPage {
        +health: HealthResponse
        +recent: RecentPredictionResponse
        +useEffect() fetch
    }
    class PredictPage {
        +runPrediction()
    }
    class TrainPage {
        +form: TrainRequest
        +status: TrainStatus
        +eventSource: EventSource
        +startTrain()
        +subscribeSSE(run_id)
    }
    class HistoryPage {
        +runs: RunSummary[]
        +selected: RunDetail?
    }
    class MetricCard {
        +label, value, hint
    }
    class api {
        <<module>>
        +health()
        +recent(n)
        +runs()
        +run(id)
        +startTrain(req)
        +trainStatus()
        +trainStreamUrl(run_id)
    }

    App --> NavBar
    App --> DashboardPage : route /dashboard
    App --> PredictPage : route /predict
    App --> TrainPage : route /train
    App --> HistoryPage : route /history
    DashboardPage --> MetricCard
    HistoryPage --> MetricCard
    DashboardPage ..> api
    PredictPage ..> api
    TrainPage ..> api
    HistoryPage ..> api
```

---

## 4. 시퀀스 다이어그램 — 학습 트리거 + SSE 진행률

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as TrainPage (React)
    participant API as FastAPI /api/train
    participant TM as TrainingManager
    participant Job as TrainingJob (Thread)
    participant Pipe as SolarPowerPredictionPipeline
    participant CB as StreamingCallback
    participant SSE as /api/train/stream/{id}

    User->>FE: 학습 시작 (config 입력)
    FE->>API: POST /api/train (TrainRequest)
    API->>TM: start(overrides)
    TM->>TM: Lock acquire, run_id 생성
    TM->>Job: Thread 시작
    TM-->>API: TrainingJob (pending→running)
    API-->>FE: job status JSON

    FE->>SSE: GET /api/train/stream/{run_id}
    SSE->>TM: 현재 job 조회
    SSE->>Job: drain_events(job)

    Job->>Pipe: run_full_pipeline(extra_callbacks=[cb])
    Pipe->>Pipe: data → features → build model
    loop 매 epoch
        Pipe->>CB: on_epoch_end(epoch, logs)
        CB->>Job: events.put({event:'epoch', ...})
        Job->>SSE: queue 소비
        SSE-->>FE: event: epoch / data: {epoch, loss, ...}
        FE->>FE: 차트 갱신
    end

    Pipe-->>Job: metrics 반환
    Job->>Job: status=completed, finished_at
    Job->>TM: on_complete(job)
    TM->>API: app.state.ml = load_app_state() (hot-swap)
    Job->>SSE: SENTINEL
    SSE-->>FE: event: done
    FE-->>User: 완료 토스트 + 새 모델 활성화
```

---

## 5. 시퀀스 다이어그램 — 최근 예측 조회

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant FE as DashboardPage
    participant API as FastAPI
    participant ST as app.state.ml (AppState)
    participant M as Keras Model
    participant E as SolarFeatureEngineer

    User->>FE: /dashboard 진입
    FE->>API: GET /api/health
    API->>ST: getattr(app.state, 'ml')
    API-->>FE: HealthResponse(status, n_features, ...)

    FE->>API: GET /api/predict/recent?n=50
    API->>ST: X_test[-n:], y_test_norm[-n:]
    API->>M: model.predict(X)
    M-->>API: y_pred_norm
    API->>E: target_scaler.inverse_transform()
    E-->>API: y_true, y_pred (원본 스케일)
    API->>API: RMSE / MAE / R² 계산
    API-->>FE: RecentPredictionResponse(points, metrics)
    FE->>FE: Recharts 라인 차트 렌더
```

---

## 6. 상태 다이어그램 — TrainingJob 라이프사이클

```mermaid
stateDiagram-v2
    [*] --> pending : TrainingManager.start()
    pending --> running : Thread 실행 시작
    running --> running : on_epoch_end (events.put)
    running --> completed : pipeline 정상 종료
    running --> failed : Exception 발생
    completed --> [*] : on_complete → 모델 hot-swap
    failed --> [*] : error 메시지 SSE 송신

    note right of running
        SENTINEL 전송 후 SSE generator 종료
        클라이언트 disconnect 감지 시
        서버 측 generator도 조기 종료
    end note
```

---

## 7. 활동 다이어그램 — 전체 파이프라인 (`SolarPowerPredictionPipeline.run_full_pipeline`)

```mermaid
flowchart TD
    A[시작: run_full_pipeline] --> B[set_random_seeds]
    B --> C[run_data_pipeline]
    C --> D{config.use_synthetic?}
    D -- yes --> D1[generate_synthetic_data]
    D -- no  --> D2[load_data + preprocess]
    D1 --> E[run_feature_engineering]
    D2 --> E
    E --> F[create features + sequences + scalers]
    F --> G{config.tune?}
    G -- yes --> G1[run_hyperparameter_tuning]
    G -- no  --> H[run_model_pipeline]
    G1 --> H
    H --> H1[CNNLSTMBuilder.build_*]
    H1 --> H2[compile_model + create_callbacks]
    H2 --> H3[ModelTrainer.train_model + StreamingCallback]
    H3 --> I[run_evaluation_pipeline]
    I --> I1[evaluate_model: RMSE/MAE/R²/MAPE/DA]
    I1 --> J{config.ensemble?}
    J -- yes --> J1[run_ensemble_pipeline]
    J -- no  --> K[save_all_results]
    J1 --> K
    K --> K1[best_model.keras, summary_*.json, history.csv]
    K1 --> L[results dict 반환]
    L --> M[종료]
```

---

## 8. 배포 다이어그램

```mermaid
flowchart LR
    subgraph DevMachine[Developer Machine]
        subgraph Browser
            UI[React SPA<br/>Vite Dev 5173]
        end
        subgraph PythonProc[Uvicorn Process :8000]
            FAPI[FastAPI app]
            TF[TensorFlow Runtime]
            MEM[(AppState<br/>인메모리)]
        end
        subgraph FS[Local File System]
            CSV[Plant_*.csv]
            RES[results/<br/>best_model.keras<br/>summary_*.json]
        end
    end

    UI -- "HTTP /api/* (proxy)" --> FAPI
    UI -- "SSE /api/train/stream" --> FAPI
    FAPI --> TF
    FAPI --> MEM
    TF --> RES
    FAPI -- read --> CSV
    FAPI -- read --> RES
```

---

## 9. 학습/예측 흐름 요약 (Use-Case 관점)

```mermaid
flowchart LR
    actor((User))

    UC1[헬스 확인]
    UC2[최근 예측 조회]
    UC3[학습 시작]
    UC4[학습 진행률 구독 SSE]
    UC5[과거 Run 목록]
    UC6[Run 상세 조회 / 학습곡선]

    actor --> UC1
    actor --> UC2
    actor --> UC3
    actor --> UC4
    actor --> UC5
    actor --> UC6

    UC3 -. triggers .-> UC4
    UC3 -. on complete .-> UC2
```

---

## 10. 데이터 모델 (입력/출력 스키마)

```mermaid
erDiagram
    GENERATION_CSV ||--o{ HOURLY_GEN : aggregates
    WEATHER_CSV   ||--o{ HOURLY_WX  : aggregates
    HOURLY_GEN    ||--|| MERGED     : merge_on_date
    HOURLY_WX     ||--|| MERGED     : merge_on_date
    MERGED        ||--o{ SEQUENCES  : sliding_window
    SEQUENCES     ||--|| MODEL_IO   : "X (N,T,F) / y (N,1)"

    GENERATION_CSV {
        string DATE_TIME
        string PLANT_ID
        string SOURCE_KEY
        float  DC_POWER
        float  AC_POWER
        float  DAILY_YIELD
        float  TOTAL_YIELD
    }
    WEATHER_CSV {
        string DATE_TIME
        string PLANT_ID
        float  AMBIENT_TEMPERATURE
        float  MODULE_TEMPERATURE
        float  IRRADIATION
    }
    MERGED {
        datetime DATE_TIME
        float    AC_POWER
        float    IRRADIATION
        float    MODULE_TEMPERATURE
        float    AMBIENT_TEMPERATURE
        float    POWER_DEMAND
        many     engineered_features
    }
    SEQUENCES {
        ndarray X "shape (N, T=24, F)"
        ndarray y "shape (N, 1)"
    }
```

---

> 본 문서의 다이어그램은 모두 Mermaid이며 코드/구조 변경 시 함께 갱신해야 합니다.
> 클래스 시그니처는 `backend/*.py` / `backend/app/**/*.py` / `frontend/src/**/*.tsx` 의 실제 정의와 동기화되어야 합니다.
