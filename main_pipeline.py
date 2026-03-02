#!/usr/bin/env python3
"""
main_pipeline.py
CNN-LSTM 태양광 발전 기반 전력 수요 예측 전체 실행 파이프라인
"""

import os
import sys
import argparse
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 로컬 모듈 임포트
from data_loader import SolarDataLoader
from feature_engineer import SolarFeatureEngineer
from cnn_lstm_model import CNNLSTMBuilder, ModelEnsemble
from model_trainer import ModelTrainer, HyperparameterTuner
from visualizer import ResultVisualizer

class SolarPowerPredictionPipeline:
    """태양광 발전 기반 전력 수요 예측 전체 파이프라인"""
    
    def __init__(self, config=None):
        """
        파이프라인 초기화
        
        Args:
            config (dict): 설정 딕셔너리
        """
        self.config = self._default_config()
        if config:
            self.config.update(config)
        self.loader = SolarDataLoader()
        self.engineer = SolarFeatureEngineer()
        self.builder = CNNLSTMBuilder()
        self.trainer = ModelTrainer()
        self.visualizer = None  # 나중에 초기화
        
        self.processed_data = None
        self.model = None
        self.results = {}
        
        # 결과 저장 디렉토리 생성
        self.output_dir = self.config.get('output_dir', 'results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _default_config(self):
        """기본 설정 반환"""
        return {
            # 데이터 설정
            'generation_file': 'Plant_1_Generation_Data.csv',
            'weather_file': 'Plant_1_Weather_Sensor_Data.csv',
            'use_synthetic_data': False,
            
            # 특성 엔지니어링 설정
            'sequence_length': 24,
            'feature_selection_method': 'correlation',
            'normalization_method': 'minmax',
            
            # 모델 설정
            'model_type': 'advanced',  # 'basic', 'advanced', 'transformer'
            'cnn_filters': [64, 64, 32],
            'lstm_units': [128, 64],
            'dense_units': [128, 64, 32],
            'dropout_rate': 0.3,
            'use_attention': True,
            'use_bidirectional': True,
            
            # 학습 설정
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'loss': 'huber',
            'early_stopping_patience': 15,
            
            # 데이터 분할 설정
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'shuffle': False,  # 시계열 데이터이므로 False
            
            # 기타 설정
            'random_seed': 42,
            'verbose': 1,
            'save_model': True,
            'save_results': True,
            'output_dir': 'results',
            'run_cross_validation': False,
            'run_hyperparameter_tuning': False
        }
    
    def load_config(self, config_file):
        """
        설정 파일 로드 (중첩 구조의 JSON도 평탄화하여 적용)

        Args:
            config_file (str): 설정 파일 경로
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            # 중첩 구조 평탄화 (_로 시작하는 주석 키 제외)
            flat_config = {}
            for key, value in loaded_config.items():
                if key.startswith('_'):
                    continue
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if not sub_key.startswith('_'):
                            flat_config[sub_key] = sub_value
                else:
                    flat_config[key] = value

            self.config.update(flat_config)
            print(f"✅ 설정 파일 로드 완료: {config_file}")

        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
            print("💡 기본 설정을 사용합니다.")
    
    def save_config(self, config_file=None):
        """
        현재 설정 저장
        
        Args:
            config_file (str): 저장할 파일명
        """
        if config_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_file = os.path.join(self.output_dir, f'config_{timestamp}.json')
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"✅ 설정 파일 저장: {config_file}")
            
        except Exception as e:
            print(f"❌ 설정 파일 저장 실패: {e}")
    
    def run_data_pipeline(self):
        """데이터 로드 및 전처리 파이프라인"""
        print("🚀 데이터 파이프라인 시작")
        print("="*60)
        
        # 1. 데이터 로드
        if self.config['use_synthetic_data']:
            print("📊 합성 데이터 사용")
            gen_df, weather_df = self.loader.generate_synthetic_data()
            gen_df = self.loader.parse_datetime(gen_df, format_type='generation')
            weather_df = self.loader.parse_datetime(weather_df, format_type='weather')
            processed_df = self.loader.merge_data(
                self.loader.aggregate_hourly_generation(gen_df),
                self.loader.aggregate_hourly_weather(weather_df)
            )
            processed_df = self.loader.filter_daytime_data(processed_df)
        else:
            processed_df = self.loader.preprocess_pipeline(
                self.config['generation_file'],
                self.config['weather_file']
            )
        
        # 2. 특성 엔지니어링
        X, y, features, scalers = self.engineer.feature_engineering_pipeline(
            processed_df,
            sequence_length=self.config['sequence_length'],
            feature_selection_method=self.config['feature_selection_method']
        )
        
        self.processed_data = {
            'X': X,
            'y': y,
            'features': features,
            'scalers': scalers,
            'raw_data': processed_df
        }
        
        print("✅ 데이터 파이프라인 완료")
        return self.processed_data
    
    def run_model_pipeline(self):
        """모델 구축 및 학습 파이프라인"""
        print("🧠 모델 파이프라인 시작")
        print("="*60)
        
        if self.processed_data is None:
            raise ValueError("데이터 파이프라인을 먼저 실행하세요.")
        
        X, y = self.processed_data['X'], self.processed_data['y']
        
        # 1. 데이터 분할
        X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_train_data(
            X, y,
            train_ratio=self.config['train_ratio'],
            val_ratio=self.config['val_ratio'],
            test_ratio=self.config['test_ratio'],
            shuffle=self.config['shuffle']
        )
        
        # 2. 모델 구축
        input_shape = X.shape[1:]
        
        if self.config['model_type'] == 'basic':
            model = self.builder.build_basic_cnn_lstm(
                input_shape,
                cnn_filters=self.config['cnn_filters'][:2],  # 기본 모델은 2개 레이어
                lstm_units=self.config['lstm_units'],
                dense_units=self.config['dense_units'],
                dropout_rate=self.config['dropout_rate']
            )
        elif self.config['model_type'] == 'transformer':
            model = self.builder.build_transformer_cnn_lstm(
                input_shape,
                cnn_filters=self.config['cnn_filters'],
                lstm_units=self.config['lstm_units'],
                dense_units=self.config['dense_units'],
                dropout_rate=self.config['dropout_rate']
            )
        else:  # advanced
            model = self.builder.build_advanced_cnn_lstm(
                input_shape,
                cnn_filters=self.config['cnn_filters'],
                lstm_units=self.config['lstm_units'],
                dense_units=self.config['dense_units'],
                dropout_rate=self.config['dropout_rate'],
                use_attention=self.config['use_attention'],
                use_bidirectional=self.config['use_bidirectional']
            )
        
        # 3. 모델 컴파일
        model = self.builder.compile_model(
            model,
            optimizer=self.config['optimizer'],
            learning_rate=self.config['learning_rate'],
            loss=self.config['loss']
        )
        
        # 4. 콜백 설정
        model_save_path = os.path.join(self.output_dir, 'best_model.h5')
        callbacks = self.builder.create_callbacks(
            patience=self.config['early_stopping_patience'],
            save_path=model_save_path
        )
        
        # 5. 모델 학습
        history = self.trainer.train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=self.config['verbose']
        )
        
        self.model = model
        
        # 6. 모델 평가
        metrics, y_true, y_pred = self.trainer.evaluate_model(
            model, X_test, y_test, 
            scalers=self.processed_data['scalers'],
            verbose=self.config['verbose']
        )
        
        self.results = {
            'metrics': metrics,
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred
            },
            'data_splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            },
            'model_config': self.builder.model_config
        }
        
        print("✅ 모델 파이프라인 완료")
        return self.results
    
    def run_evaluation_pipeline(self):
        """평가 및 시각화 파이프라인"""
        print("📊 평가 파이프라인 시작")
        print("="*60)
        
        if self.model is None or not self.results:
            raise ValueError("모델 파이프라인을 먼저 실행하세요.")
        
        # 시각화기 초기화
        try:
            from visualizer import ResultVisualizer
            self.visualizer = ResultVisualizer()
        except ImportError:
            print("⚠️ visualizer 모듈을 찾을 수 없습니다. 기본 시각화만 실행합니다.")
            self.visualizer = None
        
        # 1. 학습 과정 시각화
        self.trainer.plot_training_history()
        
        # 2. 예측 결과 시각화
        y_true = self.results['predictions']['y_true']
        y_pred = self.results['predictions']['y_pred']
        self.trainer.plot_predictions(y_true, y_pred)
        
        # 3. 특성 중요도 시각화
        feature_names = self.processed_data['features']
        self.trainer.plot_feature_importance(self.model, feature_names)
        
        # 4. 고급 시각화 (visualizer 모듈이 있는 경우)
        if self.visualizer:
            self.visualizer.plot_comprehensive_results(self.results, self.processed_data)
        
        # 5. 교차 검증 (설정에서 활성화된 경우)
        if self.config.get('run_cross_validation', False):
            print("\n🔄 교차 검증 실행 중...")
            
            # 간단한 모델 빌더 함수 정의
            def simple_model_builder(input_shape, **kwargs):
                return self.builder.build_basic_cnn_lstm(input_shape, **kwargs)
            
            cv_results = self.trainer.cross_validate(
                simple_model_builder, 
                self.processed_data['X'], 
                self.processed_data['y']
            )
            self.results['cross_validation'] = cv_results
        
        print("✅ 평가 파이프라인 완료")
        return self.results
    
    def run_hyperparameter_tuning(self):
        """하이퍼파라미터 튜닝 실행"""
        print("🔍 하이퍼파라미터 튜닝 시작")
        print("="*60)
        
        if self.processed_data is None:
            raise ValueError("데이터 파이프라인을 먼저 실행하세요.")
        
        X, y = self.processed_data['X'], self.processed_data['y']
        
        # 데이터 분할 (튜닝용)
        X_train, X_val, _, y_train, y_val, _ = self.trainer.prepare_train_data(X, y)
        
        # 튜닝할 파라미터 그리드 정의
        param_grid = {
            'cnn_filters': [[32, 16], [64, 32], [64, 64, 32]],
            'lstm_units': [[32, 16], [64, 32], [128, 64]],
            'dropout_rate': [0.2, 0.3, 0.4],
            'dense_units': [[32, 16], [64, 32], [128, 64]]
        }
        
        # 모델 빌더 함수
        def tuning_model_builder(input_shape, **params):
            return self.builder.build_basic_cnn_lstm(input_shape, **params)
        
        # 튜너 초기화 및 실행
        tuner = HyperparameterTuner(tuning_model_builder)
        tuning_results = tuner.grid_search(
            X_train, y_train, X_val, y_val, 
            param_grid, scoring='rmse'
        )
        
        # 최적 파라미터로 설정 업데이트
        self.config.update(tuning_results['best_params'])
        
        print("✅ 하이퍼파라미터 튜닝 완료")
        return tuning_results
    
    def run_ensemble_pipeline(self, n_models=3):
        """앙상블 모델 파이프라인"""
        print(f"🎭 앙상블 모델 파이프라인 시작 ({n_models}개 모델)")
        print("="*60)
        
        if self.processed_data is None:
            raise ValueError("데이터 파이프라인을 먼저 실행하세요.")
        
        X, y = self.processed_data['X'], self.processed_data['y']
        X_train, X_val, X_test, y_train, y_val, y_test = self.trainer.prepare_train_data(X, y)
        
        ensemble = ModelEnsemble()
        models_info = []
        
        # 다양한 모델 구성으로 앙상블 생성
        model_configs = [
            {
                'type': 'basic',
                'cnn_filters': [64, 32],
                'lstm_units': [64, 32],
                'weight': 0.3
            },
            {
                'type': 'advanced',
                'cnn_filters': [64, 64, 32],
                'lstm_units': [128, 64],
                'use_attention': True,
                'weight': 0.4
            },
            {
                'type': 'advanced',
                'cnn_filters': [128, 64, 32],
                'lstm_units': [64, 32],
                'use_bidirectional': True,
                'weight': 0.3
            }
        ]
        
        for i, config in enumerate(model_configs[:n_models]):
            print(f"\n🔨 앙상블 모델 {i+1} 학습 중...")
            
            # 모델 구축
            if config['type'] == 'basic':
                model = self.builder.build_basic_cnn_lstm(
                    X.shape[1:],
                    cnn_filters=config['cnn_filters'],
                    lstm_units=config['lstm_units']
                )
            else:
                model = self.builder.build_advanced_cnn_lstm(
                    X.shape[1:],
                    cnn_filters=config['cnn_filters'],
                    lstm_units=config['lstm_units'],
                    use_attention=config.get('use_attention', False),
                    use_bidirectional=config.get('use_bidirectional', False)
                )
            
            # 컴파일 및 학습
            model = self.builder.compile_model(model)
            
            # 빠른 학습 (앙상블용)
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=0
            )
            
            # 앙상블에 추가
            ensemble.add_model(model, weight=config['weight'])
            
            # 개별 모델 성능 평가
            metrics, _, _ = self.trainer.evaluate_model(model, X_test, y_test, verbose=0)
            models_info.append({
                'model_id': i+1,
                'config': config,
                'metrics': metrics
            })
            
            print(f"  모델 {i+1} RMSE: {metrics['RMSE']:.4f}")
        
        # 앙상블 예측 및 평가
        print(f"\n🎯 앙상블 예측 수행 중...")
        y_pred_ensemble = ensemble.predict(X_test)
        
        # 앙상블 성능 평가
        if self.processed_data['scalers'] and 'target_scaler' in self.processed_data['scalers']:
            scaler = self.processed_data['scalers']['target_scaler']
            y_true_unscaled = scaler.inverse_transform(y_test)
            y_pred_unscaled = scaler.inverse_transform(y_pred_ensemble)
        else:
            y_true_unscaled = y_test
            y_pred_unscaled = y_pred_ensemble
        
        # 앙상블 지표 계산
        ensemble_metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled)),
            'MAE': mean_absolute_error(y_true_unscaled, y_pred_unscaled),
            'R²': r2_score(y_true_unscaled, y_pred_unscaled)
        }
        
        print("="*60)
        print("🏆 앙상블 vs 개별 모델 성능 비교")
        print("="*60)
        
        for info in models_info:
            print(f"모델 {info['model_id']} RMSE: {info['metrics']['RMSE']:.4f}")
        
        print(f"앙상블 RMSE: {ensemble_metrics['RMSE']:.4f}")
        print("="*60)
        
        # 앙상블 저장
        ensemble_dir = os.path.join(self.output_dir, 'ensemble')
        os.makedirs(ensemble_dir, exist_ok=True)
        ensemble.save_ensemble(os.path.join(ensemble_dir, 'ensemble_model'))
        
        ensemble_results = {
            'ensemble_metrics': ensemble_metrics,
            'individual_models': models_info,
            'predictions': {
                'y_true': y_true_unscaled,
                'y_pred': y_pred_unscaled
            }
        }
        
        print("✅ 앙상블 파이프라인 완료")
        return ensemble_results
    
    def save_all_results(self):
        """모든 결과 저장"""
        print("💾 결과 저장 중...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 설정 저장
        self.save_config(os.path.join(self.output_dir, f'config_{timestamp}.json'))
        
        # 2. 모델 저장
        if self.model and self.config.get('save_model', True):
            model_path = os.path.join(self.output_dir, f'final_model_{timestamp}.h5')
            self.model.save(model_path)
            print(f"✅ 모델 저장: {model_path}")
        
        # 3. 결과 저장
        if self.results and self.config.get('save_results', True):
            saved_files = self.trainer.save_results(
                self.results['metrics'],
                self.results['predictions']['y_true'],
                self.results['predictions']['y_pred'],
                filepath_prefix=os.path.join(self.output_dir, f'results_{timestamp}')
            )
        
        # 4. 전체 결과 요약 저장
        summary = {
            'timestamp': timestamp,
            'config': self.config,
            'data_info': {
                'total_samples': len(self.processed_data['X']) if self.processed_data else 0,
                'features_count': len(self.processed_data['features']) if self.processed_data else 0,
                'sequence_length': self.config['sequence_length']
            },
            'model_info': self.builder.model_config if hasattr(self.builder, 'model_config') else {},
            'performance': self.results.get('metrics', {}) if self.results else {}
        }
        
        summary_file = os.path.join(self.output_dir, f'summary_{timestamp}.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 결과 요약 저장: {summary_file}")
        print(f"📁 모든 결과가 '{self.output_dir}' 디렉토리에 저장되었습니다.")
        
        return {
            'summary_file': summary_file,
            'output_directory': self.output_dir
        }
    
    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        print("🌟 CNN-LSTM 태양광 발전 기반 전력 수요 예측 시스템 시작")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # 재현성을 위한 시드 설정
            seed = self.config.get('random_seed', 42)
            np.random.seed(seed)
            import tensorflow as tf
            tf.random.set_seed(seed)

            # 1. 데이터 파이프라인
            self.run_data_pipeline()

            # 2. 하이퍼파라미터 튜닝 (선택사항, 데이터 로드 후 실행)
            if self.config.get('run_hyperparameter_tuning', False):
                tuning_results = self.run_hyperparameter_tuning()
                print(f"🏆 최적 하이퍼파라미터 적용됨")

            # 3. 모델 파이프라인
            self.run_model_pipeline()
            
            # 4. 평가 파이프라인
            self.run_evaluation_pipeline()
            
            # 5. 앙상블 실행 (선택사항)
            if self.config.get('run_ensemble', False):
                ensemble_results = self.run_ensemble_pipeline()
                self.results['ensemble'] = ensemble_results
            
            # 6. 결과 저장
            self.save_all_results()
            
            # 실행 시간 계산
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            # 최종 결과 요약
            print("\n" + "="*80)
            print("🎉 전체 파이프라인 실행 완료!")
            print("="*80)
            print(f"⏰ 실행 시간: {execution_time}")
            print(f"📊 최종 성능 지표:")
            
            if self.results and 'metrics' in self.results:
                for metric, value in self.results['metrics'].items():
                    if isinstance(value, float):
                        print(f"  {metric:>18}: {value:.4f}")
                    else:
                        print(f"  {metric:>18}: {value}")
            
            print(f"📁 결과 저장 위치: {self.output_dir}")
            print("="*80)
            
            return {
                'success': True,
                'results': self.results,
                'execution_time': str(execution_time),
                'output_dir': self.output_dir
            }
            
        except Exception as e:
            print(f"\n❌ 파이프라인 실행 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': str(datetime.now() - start_time)
            }

def create_sample_config():
    """샘플 설정 파일 생성"""
    sample_config = {
        "generation_file": "Plant_1_Generation_Data.csv",
        "weather_file": "Plant_1_Weather_Sensor_Data.csv",
        "use_synthetic_data": False,
        
        "sequence_length": 24,
        "feature_selection_method": "correlation",
        "normalization_method": "minmax",
        
        "model_type": "advanced",
        "cnn_filters": [64, 64, 32],
        "lstm_units": [128, 64],
        "dense_units": [128, 64, 32],
        "dropout_rate": 0.3,
        "use_attention": True,
        "use_bidirectional": True,
        
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "huber",
        "early_stopping_patience": 15,
        
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "shuffle": False,
        
        "random_seed": 42,
        "verbose": 1,
        "save_model": True,
        "save_results": True,
        "output_dir": "results",
        
        "run_cross_validation": False,
        "run_hyperparameter_tuning": False,
        "run_ensemble": False
    }
    
    with open('config_sample.json', 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    print("✅ 샘플 설정 파일 생성: config_sample.json")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="CNN-LSTM 태양광 발전 기반 전력 수요 예측 시스템"
    )
    parser.add_argument(
        '--config', '-c', 
        help='설정 파일 경로',
        default=None
    )
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='샘플 설정 파일 생성'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='합성 데이터 사용'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='하이퍼파라미터 튜닝 실행'
    )
    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='앙상블 모델 실행'
    )
    parser.add_argument(
        '--output', '-o',
        help='결과 저장 디렉토리',
        default='results'
    )
    
    args = parser.parse_args()
    
    # 샘플 설정 파일 생성
    if args.create_config:
        create_sample_config()
        return
    
    # 파이프라인 초기화
    pipeline = SolarPowerPredictionPipeline()
    
    # 설정 로드
    if args.config:
        pipeline.load_config(args.config)
    
    # 명령행 인자로 설정 오버라이드
    if args.synthetic:
        pipeline.config['use_synthetic_data'] = True
    
    if args.tune:
        pipeline.config['run_hyperparameter_tuning'] = True
    
    if args.ensemble:
        pipeline.config['run_ensemble'] = True
    
    if args.output:
        pipeline.config['output_dir'] = args.output
    
    # 전체 파이프라인 실행
    result = pipeline.run_full_pipeline()
    
    # 실행 결과에 따른 종료 코드
    sys.exit(0 if result['success'] else 1)

if __name__ == "__main__":
    main()
