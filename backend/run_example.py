#!/usr/bin/env python3
"""
run_example.py
CNN-LSTM 태양광 발전 기반 전력 수요 예측 시스템 실행 예제
"""

import os
import sys
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_pipeline import SolarPowerPredictionPipeline

def run_basic_example():
    """기본 예제 실행"""
    print("🌟 기본 예제 실행 시작")
    print("="*60)
    
    # 기본 설정으로 파이프라인 생성
    config = {
        'use_synthetic_data': True,  # 합성 데이터 사용
        'model_type': 'basic',
        'epochs': 20,  # 빠른 테스트를 위해 줄임
        'batch_size': 32,
        'sequence_length': 12,  # 메모리 절약
        'verbose': 1,
        'save_model': True,
        'save_results': True,
        'output_dir': 'results_basic_example'
    }
    
    # 파이프라인 실행
    pipeline = SolarPowerPredictionPipeline(config)
    result = pipeline.run_full_pipeline()
    
    if result['success']:
        print("\n🎉 기본 예제 실행 성공!")
        print(f"📁 결과 저장 위치: {result['output_dir']}")
    else:
        print(f"\n❌ 기본 예제 실행 실패: {result.get('error', 'Unknown error')}")
    
    return result

def run_advanced_example():
    """고급 예제 실행"""
    print("🚀 고급 예제 실행 시작")
    print("="*60)
    
    # 고급 설정
    config = {
        'use_synthetic_data': True,
        'model_type': 'advanced',
        'epochs': 50,
        'batch_size': 32,
        'sequence_length': 24,
        'cnn_filters': [64, 64, 32],
        'lstm_units': [128, 64],
        'dense_units': [128, 64, 32],
        'dropout_rate': 0.3,
        'use_attention': True,
        'use_bidirectional': True,
        'learning_rate': 0.001,
        'early_stopping_patience': 15,
        'verbose': 1,
        'save_model': True,
        'save_results': True,
        'output_dir': 'results_advanced_example',
        'run_cross_validation': False,  # 시간 절약을 위해 비활성화
        'run_ensemble': False
    }
    
    # 파이프라인 실행
    pipeline = SolarPowerPredictionPipeline(config)
    result = pipeline.run_full_pipeline()
    
    if result['success']:
        print("\n🎉 고급 예제 실행 성공!")
        print(f"📁 결과 저장 위치: {result['output_dir']}")
        
        # 결과 요약 출력
        if 'results' in result and 'metrics' in result['results']:
            print("\n📊 최종 성능 지표:")
            metrics = result['results']['metrics']
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric:>15}: {value:.4f}")
    else:
        print(f"\n❌ 고급 예제 실행 실패: {result.get('error', 'Unknown error')}")
    
    return result

def run_real_data_example():
    """실제 데이터 예제 실행"""
    print("📊 실제 데이터 예제 실행 시작")
    print("="*60)
    
    # 실제 데이터 파일 존재 확인
    gen_file = 'Plant_1_Generation_Data.csv'
    weather_file = 'Plant_1_Weather_Sensor_Data.csv'
    
    if not (os.path.exists(gen_file) and os.path.exists(weather_file)):
        print(f"❌ 실제 데이터 파일을 찾을 수 없습니다:")
        print(f"  - {gen_file}: {'✅' if os.path.exists(gen_file) else '❌'}")
        print(f"  - {weather_file}: {'✅' if os.path.exists(weather_file) else '❌'}")
        print("💡 합성 데이터 예제를 실행하세요.")
        return None
    
    # 실제 데이터 설정
    config = {
        'generation_file': gen_file,
        'weather_file': weather_file,
        'use_synthetic_data': False,
        'model_type': 'advanced',
        'epochs': 100,
        'batch_size': 32,
        'sequence_length': 24,
        'early_stopping_patience': 15,
        'verbose': 1,
        'save_model': True,
        'save_results': True,
        'output_dir': 'results_real_data_example'
    }
    
    # 파이프라인 실행
    pipeline = SolarPowerPredictionPipeline(config)
    result = pipeline.run_full_pipeline()
    
    if result['success']:
        print("\n🎉 실제 데이터 예제 실행 성공!")
        print(f"📁 결과 저장 위치: {result['output_dir']}")
    else:
        print(f"\n❌ 실제 데이터 예제 실행 실패: {result.get('error', 'Unknown error')}")
    
    return result

def run_hyperparameter_tuning_example():
    """하이퍼파라미터 튜닝 예제"""
    print("🔍 하이퍼파라미터 튜닝 예제 실행 시작")
    print("="*60)
    
    config = {
        'use_synthetic_data': True,
        'model_type': 'basic',  # 튜닝에는 기본 모델 사용
        'epochs': 30,  # 튜닝용이므로 줄임
        'sequence_length': 12,
        'run_hyperparameter_tuning': True,
        'verbose': 1,
        'output_dir': 'results_tuning_example'
    }
    
    pipeline = SolarPowerPredictionPipeline(config)
    result = pipeline.run_full_pipeline()
    
    if result['success']:
        print("\n🎉 하이퍼파라미터 튜닝 완료!")
        print(f"📁 결과 저장 위치: {result['output_dir']}")
    else:
        print(f"\n❌ 튜닝 실패: {result.get('error', 'Unknown error')}")
    
    return result

def run_ensemble_example():
    """앙상블 모델 예제"""
    print("🎭 앙상블 모델 예제 실행 시작")
    print("="*60)
    
    config = {
        'use_synthetic_data': True,
        'model_type': 'advanced',
        'epochs': 30,
        'sequence_length': 24,
        'run_ensemble': True,
        'verbose': 1,
        'output_dir': 'results_ensemble_example'
    }
    
    pipeline = SolarPowerPredictionPipeline(config)
    result = pipeline.run_full_pipeline()
    
    if result['success']:
        print("\n🎉 앙상블 모델 완료!")
        print(f"📁 결과 저장 위치: {result['output_dir']}")
        
        # 앙상블 성능 출력
        if 'results' in result and 'ensemble' in result['results']:
            ensemble_metrics = result['results']['ensemble']['ensemble_metrics']
            print(f"\n🏆 앙상블 모델 성능:")
            for metric, value in ensemble_metrics.items():
                print(f"  {metric:>6}: {value:.4f}")
    else:
        print(f"\n❌ 앙상블 실행 실패: {result.get('error', 'Unknown error')}")
    
    return result

def run_individual_modules_test():
    """개별 모듈 테스트"""
    print("🧪 개별 모듈 테스트 시작")
    print("="*60)
    
    try:
        # 1. 데이터 로더 테스트
        print("\n1️⃣ 데이터 로더 테스트...")
        from data_loader import SolarDataLoader
        
        loader = SolarDataLoader()
        # 합성 데이터 생성 테스트
        gen_df, weather_df = loader.generate_synthetic_data()
        print(f"✅ 합성 데이터 생성: 발전량 {len(gen_df)}, 기상 {len(weather_df)} 레코드")
        
        # 2. 특성 엔지니어링 테스트
        print("\n2️⃣ 특성 엔지니어링 테스트...")
        from feature_engineer import SolarFeatureEngineer
        
        # 간단한 데이터프레임 생성
        import pandas as pd
        import numpy as np
        
        test_data = pd.DataFrame({
            'DATE_TIME': pd.date_range('2020-05-15', periods=100, freq='H'),
            'AC_POWER': np.random.exponential(1000, 100),
            'IRRADIATION': np.random.gamma(2, 0.5, 100),
            'AMBIENT_TEMPERATURE': 25 + np.random.normal(0, 5, 100)
        })
        
        engineer = SolarFeatureEngineer()
        splits, features, scalers = engineer.feature_engineering_pipeline(
            test_data, sequence_length=12
        )
        X_train, y_train = splits['X_train'], splits['y_train']
        X_val, y_val = splits['X_val'], splits['y_val']
        X_test, y_test = splits['X_test'], splits['y_test']
        print(f"✅ 특성 엔지니어링: train={X_train.shape[0]} 시퀀스, {len(features)} 특성")

        # 3. 모델 빌더 테스트
        print("\n3️⃣ 모델 빌더 테스트...")
        from cnn_lstm_model import CNNLSTMBuilder

        builder = CNNLSTMBuilder()

        # 기본 모델
        basic_model = builder.build_basic_cnn_lstm(X_train.shape[1:])
        basic_model = builder.compile_model(basic_model)
        print(f"✅ 기본 CNN-LSTM 모델: {basic_model.count_params():,} 파라미터")

        # 고급 모델
        advanced_model = builder.build_advanced_cnn_lstm(X_train.shape[1:])
        advanced_model = builder.compile_model(advanced_model)
        print(f"✅ 고급 CNN-LSTM 모델: {advanced_model.count_params():,} 파라미터")

        # 4. 트레이너 테스트
        print("\n4️⃣ 모델 트레이너 테스트...")
        from model_trainer import ModelTrainer

        trainer = ModelTrainer()
        print(f"✅ 데이터 분할: 훈련({len(X_train)}), 검증({len(X_val)}), 테스트({len(X_test)})")
        
        # 간단한 학습 테스트 (1 에포크)
        history = trainer.train_model(
            basic_model, X_train, y_train, X_val, y_val,
            epochs=1, batch_size=16, verbose=0
        )
        print("✅ 모델 학습 테스트 완료")
        
        # 5. 시각화 테스트
        print("\n5️⃣ 시각화 테스트...")
        
        # 간단한 예측 수행
        y_pred = basic_model.predict(X_test, verbose=0)
        y_true = y_test
        
        # 역정규화
        if 'target_scaler' in scalers:
            y_true = scalers['target_scaler'].inverse_transform(y_true)
            y_pred = scalers['target_scaler'].inverse_transform(y_pred)
        
        # 성능 지표 계산
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        }
        
        print(f"✅ 테스트 성능: RMSE={metrics['RMSE']:.2f}, R²={metrics['R²']:.3f}")
        
        print("\n🎉 모든 개별 모듈 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 모듈 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_menu():
    """메뉴 표시"""
    print("\n" + "="*60)
    print("🌞 CNN-LSTM 태양광 발전 기반 전력 수요 예측 시스템")
    print("="*60)
    print("실행할 예제를 선택하세요:")
    print()
    print("1️⃣  기본 예제 (빠른 테스트)")
    print("2️⃣  고급 예제 (완전한 기능)")
    print("3️⃣  실제 데이터 예제")
    print("4️⃣  하이퍼파라미터 튜닝 예제")
    print("5️⃣  앙상블 모델 예제")
    print("6️⃣  개별 모듈 테스트")
    print("7️⃣  모든 예제 순차 실행")
    print("0️⃣  종료")
    print()
    print("="*60)

def run_all_examples():
    """모든 예제 순차 실행"""
    print("🚀 모든 예제 순차 실행 시작")
    print("="*80)
    
    examples = [
        ("개별 모듈 테스트", run_individual_modules_test),
        ("기본 예제", run_basic_example),
        ("고급 예제", run_advanced_example),
        ("실제 데이터 예제", run_real_data_example),
        ("하이퍼파라미터 튜닝", run_hyperparameter_tuning_example),
        ("앙상블 모델", run_ensemble_example)
    ]
    
    results = {}
    
    for name, func in examples:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = func()
            results[name] = result
            print(f"✅ {name} 완료")
        except Exception as e:
            print(f"❌ {name} 실패: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # 전체 결과 요약
    print("\n" + "="*80)
    print("📊 전체 실행 결과 요약")
    print("="*80)
    
    for name, result in results.items():
        if isinstance(result, dict) and 'success' in result:
            status = "✅ 성공" if result['success'] else "❌ 실패"
        elif result is True:
            status = "✅ 성공"
        elif result is False:
            status = "❌ 실패"
        else:
            status = "❓ 알 수 없음"
        
        print(f"{name:25}: {status}")
    
    print("="*80)
    print("🎉 모든 예제 실행 완료!")

def main():
    """메인 함수"""
    while True:
        show_menu()
        
        try:
            choice = input("선택하세요 (0-7): ").strip()
            
            if choice == '0':
                print("👋 프로그램을 종료합니다.")
                break
            elif choice == '1':
                run_basic_example()
            elif choice == '2':
                run_advanced_example()
            elif choice == '3':
                run_real_data_example()
            elif choice == '4':
                run_hyperparameter_tuning_example()
            elif choice == '5':
                run_ensemble_example()
            elif choice == '6':
                run_individual_modules_test()
            elif choice == '7':
                run_all_examples()
            else:
                print("❌ 잘못된 선택입니다. 0-7 사이의 숫자를 입력하세요.")
                
        except KeyboardInterrupt:
            print("\n\n👋 사용자가 프로그램을 중단했습니다.")
            break
        except Exception as e:
            print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
            
        input("\n계속하려면 Enter 키를 누르세요...")

if __name__ == "__main__":
    print("🌟 CNN-LSTM 태양광 발전 예측 시스템 시작")
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 기본 디렉토리 생성
    os.makedirs('results', exist_ok=True)
    
    main()
