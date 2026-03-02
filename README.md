# 🌞 CNN-LSTM 태양광 발전 기반 전력 수요 예측 시스템

첨부된 태양광 발전 데이터(`Plant_1_Generation_Data.csv`, `Plant_1_Weather_Sensor_Data.csv`)를 활용하여 CNN-LSTM 융합 딥러닝 모델로 전력 수요를 예측하는 Python 시스템입니다.

## 📋 프로젝트 개요

### 🎯 목표
- 태양광 발전량 데이터와 기상 데이터를 융합하여 전력 수요 패턴 학습
- CNN의 패턴 인식과 LSTM의 시계열 예측 능력을 결합한 하이브리드 모델
- 실시간 전력 수요 예측을 통한 전력 계통 운영 최적화

### 🏗️ 시스템 아키텍처
```
데이터 수집 → 전처리 → 특성 엔지니어링 → CNN-LSTM 모델 → 예측 → 시각화
     ↓            ↓           ↓              ↓         ↓        ↓
실제/합성 데이터  시간별 집계  순환인코딩/래그    융합 아키텍처  성능평가  대시보드
```

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

첨부된 CSV 파일을 프로젝트 루트에 배치합니다.
```
├── Plant_1_Generation_Data.csv
├── Plant_1_Weather_Sensor_Data.csv
└── (기타 파일들...)
```

> **데이터 출처**: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data?resource=download

### 3. 기본 실행
```bash
# 전체 파이프라인 실행
python main_pipeline.py

# 설정 파일 사용
python main_pipeline.py --config config_sample.json

# 합성 데이터로 테스트
python main_pipeline.py --synthetic

# 하이퍼파라미터 튜닝 포함
python main_pipeline.py --tune --ensemble
```

## 📁 파일 구조

```
solar-power-prediction/
├── data_loader.py              # 데이터 로드 및 전처리
├── feature_engineer.py         # 특성 엔지니어링
├── cnn_lstm_model.py          # CNN-LSTM 모델 정의
├── model_trainer.py           # 모델 학습 및 평가
├── visualizer.py              # 결과 시각화
├── main_pipeline.py           # 전체 실행 파이프라인
├── run_example.py             # 실행 예제 (메뉴 기반)
├── requirements.txt           # 필수 패키지 목록
├── README.md                  # 프로젝트 문서
├── config_sample.json         # 샘플 설정 파일 (중첩 구조)
└── results/                   # 결과 출력 디렉토리
    ├── models/               # 학습된 모델
    ├── plots/                # 생성된 그래프
    └── logs/                 # 학습 로그
```

## 🧩 모듈별 기능

### 📊 data_loader.py
- **기능**: CSV 데이터 로드, 시간별 집계, 데이터 병합
- **주요 클래스**: `SolarDataLoader`
- **처리 데이터**: 
  - 발전량: 68,778 레코드 (22개 인버터)
  - 기상: 3,182 레코드 (15분 간격)

### ⚙️ feature_engineer.py
- **기능**: 고급 특성 엔지니어링, 정규화, 시퀀스 생성
- **주요 특성**:
  - 시간 순환 인코딩 (sin/cos)
  - 래그 특성 (1, 3, 6, 12시간)
  - 롤링 윈도우 (이동평균, 표준편차)
  - 발전 효율성 지표

### 🧠 cnn_lstm_model.py
- **기능**: 다양한 CNN-LSTM 모델 아키텍처 제공
- **모델 타입**:
  - **Basic**: 표준 CNN-LSTM
  - **Advanced**: Bidirectional + Time-step Attention (Multi-scale CNN)
  - **Transformer**: Multi-head Self-Attention 포함

### 🏃 model_trainer.py
- **기능**: 모델 학습, 평가, 교차검증
- **주요 기능**:
  - 시계열 데이터 분할
  - 조기 종료, 학습률 스케줄링
  - 하이퍼파라미터 그리드 서치
  - 모델 앙상블

### 🎨 visualizer.py
- **기능**: 종합적인 결과 시각화
- **시각화 종류**:
  - 성능 대시보드
  - 시간별 예측 분석
  - 잔차 분석
  - 인터랙티브 Plotly 대시보드

### 🔄 main_pipeline.py
- **기능**: 전체 시스템 통합 실행
- **파이프라인**:
  1. 시드 설정 (재현성 보장)
  2. 데이터 로드 및 전처리
  3. 하이퍼파라미터 튜닝 (선택)
  4. 모델 구축 및 학습
  5. 성능 평가 및 시각화
  6. 앙상블 (선택)
  7. 결과 저장

## ⚙️ 설정 옵션

### config.json 예시

설정 파일은 **평탄(flat) 구조**와 **중첩(nested) 구조** 모두 지원합니다. 중첩 구조 사용 시 `_`로 시작하는 주석 키는 자동으로 무시됩니다. 자세한 중첩 구조 예시는 `config_sample.json`을 참고하세요.

```json
{
  "model_type": "advanced",
  "sequence_length": 24,
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "cnn_filters": [64, 64, 32],
  "lstm_units": [128, 64],
  "dropout_rate": 0.3,
  "use_attention": true,
  "use_bidirectional": true
}
```

### 명령행 인자
```bash
python main_pipeline.py [OPTIONS]

Options:
  -c, --config PATH     설정 파일 경로
  --synthetic          합성 데이터 사용
  --tune              하이퍼파라미터 튜닝
  --ensemble          앙상블 모델 실행
  -o, --output PATH    결과 저장 디렉토리
  --create-config      샘플 설정 파일 생성
```

## 📈 성능 지표

### 실측 성능 (Plant_1 데이터 기준)

Advanced CNN-LSTM (Bidirectional + Time-step Attention) 모델의 실제 학습 결과입니다.

| 지표 | 값 |
|------|-----|
| **RMSE** | 4,387 kW |
| **MAE** | 3,398 kW |
| **R²** | 0.7514 |
| **MAPE** | 75.92% |
| **Direction Accuracy** | 51.42% |

### 학습 조건
- **데이터**: 1,647 시퀀스 (66개 특성, 시퀀스 길이 24)
- **모델**: Advanced CNN-LSTM (Bidirectional + Time-step Attention)
- **학습 설정**: Epochs 100 (조기 종료), Batch Size 32, Learning Rate 0.001
- **손실 함수**: Huber Loss / **옵티마이저**: Adam
- **데이터 분할**: Train 70% / Validation 15% / Test 15%

### 모델 타입 비교 (참고)
| 모델 | 특징 | 복잡도 |
|------|------|--------|
| Basic CNN-LSTM | 표준 CNN + LSTM 구조 | 낮음 |
| Advanced CNN-LSTM | Multi-scale CNN + Bidirectional LSTM + Time-step Attention | 중간 |
| Transformer Hybrid | Multi-head Self-Attention + CNN-LSTM 결합 | 높음 |

## 🔧 고급 사용법

### 개별 모듈 사용
```python
# 데이터 로드
from data_loader import SolarDataLoader
loader = SolarDataLoader()
data = loader.preprocess_pipeline('gen.csv', 'weather.csv')

# 특성 엔지니어링
from feature_engineer import SolarFeatureEngineer
engineer = SolarFeatureEngineer()
X, y, features, scalers = engineer.feature_engineering_pipeline(data)

# 모델 구축
from cnn_lstm_model import CNNLSTMBuilder
builder = CNNLSTMBuilder()
model = builder.build_advanced_cnn_lstm(X.shape[1:])
```

### 커스텀 모델 구성
```python
# 커스텀 아키텍처
custom_config = {
    'cnn_filters': [128, 64, 32],
    'lstm_units': [256, 128, 64],
    'use_attention': True,
    'transformer_heads': 8
}

model = builder.build_transformer_cnn_lstm(input_shape, **custom_config)
```

### 앙상블 모델링
```python
from cnn_lstm_model import ModelEnsemble

ensemble = ModelEnsemble()
ensemble.add_model(basic_model, weight=0.3)
ensemble.add_model(advanced_model, weight=0.4)
ensemble.add_model(transformer_model, weight=0.3)

predictions = ensemble.predict(X_test)
```

## 🐛 문제 해결

### 일반적인 문제들

#### 메모리 부족
```python
# 해결방법
config['batch_size'] = 16  # 기본값 32에서 감소
config['sequence_length'] = 12  # 기본값 24에서 감소
```

#### 과적합
```python
# 해결방법
config['dropout_rate'] = 0.5  # 기본값 0.3에서 증가
config['early_stopping_patience'] = 10  # 더 빠른 조기 종료
```

#### 학습 불안정
```python
# 해결방법
config['learning_rate'] = 0.0001  # 학습률 감소
config['batch_size'] = 64  # 배치 크기 증가
```

### 데이터 문제

#### CSV 파일 없음
```bash
# 합성 데이터로 테스트
python main_pipeline.py --synthetic
```

#### 날짜 형식 오류
- `data_loader.py`의 `parse_datetime()` 함수 수정
- 데이터에 맞는 날짜 형식 지정

## 📊 결과 분석

### 출력 파일들
```
results/
├── best_model.h5                                   # 최적 모델 (조기 종료 기준)
├── final_model_<timestamp>.h5                      # 최종 학습 모델
├── results_<timestamp>_metrics_<timestamp>.json    # 성능 지표
├── results_<timestamp>_predictions_<timestamp>.csv # 예측 결과
├── results_<timestamp>_history_<timestamp>.csv     # 학습 이력
├── summary_<timestamp>.json                        # 전체 요약 (설정 + 결과)
├── training_log.csv                                # 에폭별 학습 로그
└── interactive_dashboard.html                      # 인터랙티브 대시보드
```

### 성능 해석
- **R² > 0.9**: 우수한 성능
- **R² 0.8-0.9**: 양호한 성능  
- **R² < 0.8**: 개선 필요


### API 서버
```python
# Flask API 예제
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('results/best_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, *model.input_shape[1:])
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})
```

---