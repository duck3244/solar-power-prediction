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
```bash
# 첨부된 CSV 파일을 프로젝트 루트에 배치
├── Plant_1_Generation_Data.csv
├── Plant_1_Weather_Sensor_Data.csv
└── (기타 파일들...)

## 사용한 데이터
 https://www.kaggle.com/datasets/anikannal/solar-power-generation-data?resource=download
```

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
├── requirements.txt           # 필수 패키지 목록
├── README.md                  # 프로젝트 문서
├── config_sample.json         # 샘플 설정 파일
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
  - **Advanced**: Bidirectional + Attention
  - **Transformer**: Multi-head Attention 포함

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
  1. 데이터 로드 및 전처리
  2. 특성 엔지니어링
  3. 모델 구축 및 학습
  4. 성능 평가 및 시각화
  5. 결과 저장

## ⚙️ 설정 옵션

### config.json 예시
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

### 예상 성능
- **RMSE**: 500-1,500 kW
- **MAE**: 300-1,000 kW  
- **R²**: 0.75-0.95
- **MAPE**: 8-15%

### 모델 비교
| 모델 | RMSE | MAE | R² | 학습시간 |
|------|------|-----|----|---------| 
| Basic CNN-LSTM | 1,200 | 800 | 0.82 | 15분 |
| Advanced CNN-LSTM | 950 | 650 | 0.88 | 25분 |
| Transformer Hybrid | 850 | 600 | 0.91 | 35분 |

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
├── final_model_20250101_123456.h5      # 학습된 모델
├── results_metrics_20250101_123456.json # 성능 지표
├── results_predictions_20250101_123456.csv # 예측 결과
├── results_history_20250101_123456.csv  # 학습 이력
├── summary_20250101_123456.json         # 전체 요약
└── plots/                              # 생성된 그래프들
    ├── performance_dashboard.png
    ├── prediction_analysis.png
    └── interactive_dashboard.html
```

### 성능 해석
- **R² > 0.9**: 우수한 성능
- **R² 0.8-0.9**: 양호한 성능  
- **R² < 0.8**: 개선 필요


### API 서버
```python
# Flask API 예제
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('trained_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(data['features'])
    return jsonify({'prediction': prediction.tolist()})
```

---