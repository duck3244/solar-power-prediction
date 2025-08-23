#!/usr/bin/env python3
"""
model_trainer.py
CNN-LSTM 모델 학습 및 평가 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import History
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """CNN-LSTM 모델 학습 및 평가 클래스"""

    def __init__(self):
        self.model = None
        self.history = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scalers = None

    def prepare_train_data(self, X, y,
                           train_ratio=0.7,
                           val_ratio=0.15,
                           test_ratio=0.15,
                           shuffle=False,
                           random_state=42):
        """
        학습/검증/테스트 데이터 분할

        Args:
            X (np.array): 입력 특성
            y (np.array): 타겟 변수
            train_ratio (float): 학습 데이터 비율
            val_ratio (float): 검증 데이터 비율
            test_ratio (float): 테스트 데이터 비율
            shuffle (bool): 데이터 셔플 여부 (시계열의 경우 False 권장)
            random_state (int): 랜덤 시드

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("📊 데이터 분할 중...")

        # 비율 검증
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio = 1.0이어야 합니다.")

        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)

        if shuffle:
            # 랜덤 분할 (비시계열 데이터용)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=(1 - train_ratio), random_state=random_state, shuffle=True
            )

            val_test_ratio = val_ratio / (val_ratio + test_ratio)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1 - val_test_ratio), random_state=random_state, shuffle=True
            )
        else:
            # 순차적 분할 (시계열 데이터용)
            X_train = X[:train_size]
            X_val = X[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]

            y_train = y[:train_size]
            y_val = y[train_size:train_size + val_size]
            y_test = y[train_size + val_size:]

        # 데이터 정보 저장
        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)

        print(f"✅ 데이터 분할 완료:")
        print(f"  - 학습: {len(X_train)} 샘플")
        print(f"  - 검증: {len(X_val)} 샘플")
        print(f"  - 테스트: {len(X_test)} 샘플")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, model, X_train, y_train, X_val, y_val,
                    epochs=100, batch_size=32, callbacks=None,
                    verbose=1, validation_freq=1):
        """
        모델 학습

        Args:
            model (tf.keras.Model): 학습할 모델
            X_train (np.array): 학습 입력 데이터
            y_train (np.array): 학습 타겟 데이터
            X_val (np.array): 검증 입력 데이터
            y_val (np.array): 검증 타겟 데이터
            epochs (int): 에포크 수
            batch_size (int): 배치 크기
            callbacks (list): 콜백 함수 리스트
            verbose (int): 출력 레벨
            validation_freq (int): 검증 주기

        Returns:
            tf.keras.callbacks.History: 학습 이력
        """
        print("🚀 모델 학습 시작...")
        print("=" * 60)

        self.model = model

        # 학습 실행
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            validation_freq=validation_freq,
            shuffle=True
        )

        self.history = history

        print("✅ 모델 학습 완료")
        return history

    def evaluate_model(self, model, X_test, y_test, scalers=None, verbose=1):
        """
        모델 성능 평가

        Args:
            model (tf.keras.Model): 평가할 모델
            X_test (np.array): 테스트 입력 데이터
            y_test (np.array): 테스트 타겟 데이터
            scalers (dict): 스케일러 딕셔너리
            verbose (int): 출력 레벨

        Returns:
            tuple: (metrics_dict, y_true, y_pred)
        """
        print("📊 모델 성능 평가 중...")

        # 예측
        y_pred_scaled = model.predict(X_test, verbose=verbose)

        # 역정규화 (스케일러가 있는 경우)
        if scalers and 'target_scaler' in scalers:
            y_pred = scalers['target_scaler'].inverse_transform(y_pred_scaled)
            y_true = scalers['target_scaler'].inverse_transform(y_test)
        else:
            y_pred = y_pred_scaled
            y_true = y_test

        # 평가 지표 계산
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE 계산 (0으로 나누기 방지)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.inf

        # 방향성 정확도 (Direction Accuracy)
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true.flatten())
            y_pred_diff = np.diff(y_pred.flatten())
            direction_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
        else:
            direction_accuracy = 0

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy
        }

        # 결과 출력
        print("=" * 60)
        print("📈 모델 성능 지표")
        print("=" * 60)
        for metric_name, value in metrics.items():
            if metric_name == 'Direction_Accuracy':
                print(f"{metric_name:>18}: {value:.2f}%")
            elif metric_name in ['MAPE']:
                print(f"{metric_name:>18}: {value:.2f}%")
            else:
                print(f"{metric_name:>18}: {value:.4f}")
        print("=" * 60)

        return metrics, y_true, y_pred

    def cross_validate(self, model_builder, X, y, cv_folds=5, **model_params):
        """
        시계열 교차 검증

        Args:
            model_builder (function): 모델 생성 함수
            X (np.array): 입력 데이터
            y (np.array): 타겟 데이터
            cv_folds (int): 교차 검증 폴드 수
            **model_params: 모델 생성 파라미터

        Returns:
            dict: 교차 검증 결과
        """
        print(f"🔄 {cv_folds}-폴드 시계열 교차 검증 시작...")

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = {
            'RMSE': [], 'MAE': [], 'R²': [], 'MAPE': []
        }

        fold = 1
        for train_idx, test_idx in tscv.split(X):
            print(f"\n📝 Fold {fold}/{cv_folds}")

            X_train_cv, X_test_cv = X[train_idx], X[test_idx]
            y_train_cv, y_test_cv = y[train_idx], y[test_idx]

            # 모델 생성 및 컴파일
            model = model_builder(input_shape=X.shape[1:], **model_params)

            # 학습 (적은 에포크로 빠르게)
            model.fit(
                X_train_cv, y_train_cv,
                epochs=20,
                batch_size=32,
                verbose=0,
                validation_split=0.2
            )

            # 평가
            y_pred = model.predict(X_test_cv, verbose=0)

            rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred))
            mae = mean_absolute_error(y_test_cv, y_pred)
            r2 = r2_score(y_test_cv, y_pred)

            # MAPE 계산
            mask = y_test_cv != 0
            mape = np.mean(np.abs((y_test_cv[mask] - y_pred[mask]) / y_test_cv[mask])) * 100 if mask.any() else np.inf

            cv_scores['RMSE'].append(rmse)
            cv_scores['MAE'].append(mae)
            cv_scores['R²'].append(r2)
            cv_scores['MAPE'].append(mape)

            print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

            fold += 1

        # 교차 검증 결과 요약
        cv_results = {}
        for metric in cv_scores.keys():
            scores = cv_scores[metric]
            cv_results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }

        print("\n" + "=" * 60)
        print("📊 교차 검증 결과 요약")
        print("=" * 60)
        for metric, result in cv_results.items():
            print(f"{metric:>6}: {result['mean']:.4f} ± {result['std']:.4f}")
        print("=" * 60)

        return cv_results

    def plot_training_history(self, history=None, figsize=(15, 10)):
        """
        학습 과정 시각화

        Args:
            history (tf.keras.callbacks.History): 학습 이력 (None이면 self.history 사용)
            figsize (tuple): 그래프 크기
        """
        if history is None:
            history = self.history

        if history is None:
            print("❌ 학습 이력이 없습니다.")
            return

        history_dict = history.history
        epochs = range(1, len(history_dict['loss']) + 1)

        # 그래프 설정
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('CNN-LSTM model training process', fontsize=16, fontweight='bold')

        # Loss
        axes[0, 0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history_dict:
            axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('📉 Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MAE
        if 'mae' in history_dict:
            axes[0, 1].plot(epochs, history_dict['mae'], 'g-', label='Training MAE', linewidth=2)
            if 'val_mae' in history_dict:
                axes[0, 1].plot(epochs, history_dict['val_mae'], 'orange', label='Validation MAE', linewidth=2)
            axes[0, 1].set_title('Mean Absolute Error')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # MAPE
        if 'mape' in history_dict:
            axes[1, 0].plot(epochs, history_dict['mape'], 'purple', label='Training MAPE', linewidth=2)
            if 'val_mape' in history_dict:
                axes[1, 0].plot(epochs, history_dict['val_mape'], 'brown', label='Validation MAPE', linewidth=2)
            axes[1, 0].set_title('Mean Absolute Percentage Error')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('MAPE (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Learning Rate (if available)
        if 'lr' in history_dict:
            axes[1, 1].plot(epochs, history_dict['lr'], 'red', linewidth=2)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # 학습 시간 또는 기타 정보 표시
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nScheduling Info\nNot Available',
                            ha='center', va='center', transform=axes[1, 1].transAxes,
                            fontsize=12, color='gray')
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, y_true, y_pred,
                         sample_range=None, figsize=(16, 12)):
        """
        예측 결과 시각화

        Args:
            y_true (np.array): 실제값
            y_pred (np.array): 예측값
            sample_range (tuple): 표시할 샘플 범위 (start, end)
            figsize (tuple): 그래프 크기
        """
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('CNN-LSTM power demand prediction results', fontsize=16, fontweight='bold')

        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # 샘플 범위 설정
        if sample_range is None:
            start, end = 0, min(100, len(y_true_flat))
        else:
            start, end = sample_range
            end = min(end, len(y_true_flat))

        # 1. 시계열 비교
        indices = range(start, end)
        axes[0, 0].plot(indices, y_true_flat[start:end], 'b-',
                        label='Actual value', linewidth=2, alpha=0.8)
        axes[0, 0].plot(indices, y_pred_flat[start:end], 'r-',
                        label='Predicted value', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Time series forecast comparison')
        axes[0, 0].set_xlabel('Time (Hours)')
        axes[0, 0].set_ylabel('Electricity demand (kW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 산점도
        axes[0, 1].scatter(y_true_flat, y_pred_flat, alpha=0.6, color='green', s=20)
        min_val = min(y_true_flat.min(), y_pred_flat.min())
        max_val = max(y_true_flat.max(), y_pred_flat.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 1].set_title('Actual vs. predicted values')
        axes[0, 1].set_xlabel('Actual value (kW)')
        axes[0, 1].set_ylabel('Predicted value (kW)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 잔차 분석
        residuals = y_true_flat - y_pred_flat
        axes[1, 0].scatter(y_pred_flat, residuals, alpha=0.6, color='orange', s=20)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Residual analysis')
        axes[1, 0].set_xlabel('Actual value (kW)')
        axes[1, 0].set_ylabel('Predicted value (kW)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 잔차 히스토그램
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].axvline(x=np.mean(residuals), color='green', linestyle='-', linewidth=2,
                           label=f'Average: {np.mean(residuals):.2f}')
        axes[1, 1].set_title('Residual distribution')
        axes[1, 1].set_xlabel('Residual (kW)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 추가 통계 정보 출력
        print("\nResidual statistics:")
        print(f"  - Average: {np.mean(residuals):.4f}")
        print(f"  - Standard deviation: {np.std(residuals):.4f}")
        print(f"  - Minimum value: {np.min(residuals):.4f}")
        print(f"  - Maximum value: {np.max(residuals):.4f}")

    def plot_feature_importance(self, model, feature_names, top_n=20):
        """
        특성 중요도 시각화 (가중치 기반 근사)

        Args:
            model (tf.keras.Model): 학습된 모델
            feature_names (list): 특성명 리스트
            top_n (int): 표시할 상위 특성 개수
        """
        try:
            # 첫 번째 Dense 레이어의 가중치를 특성 중요도로 근사
            dense_layers = [layer for layer in model.layers if 'dense' in layer.name.lower()]

            if not dense_layers:
                print("❌ Dense 레이어를 찾을 수 없습니다.")
                return

            # 첫 번째 융합 Dense 레이어 찾기
            fusion_layer = None
            for layer in model.layers:
                if 'fusion' in layer.name or 'concatenate' in layer.name:
                    # Concatenate 다음의 Dense 레이어 찾기
                    layer_idx = model.layers.index(layer)
                    for i in range(layer_idx + 1, len(model.layers)):
                        if isinstance(model.layers[i], tf.keras.layers.Dense):
                            fusion_layer = model.layers[i]
                            break
                    break

            if fusion_layer is None:
                fusion_layer = dense_layers[0]

            weights = fusion_layer.get_weights()[0]  # [input_dim, output_dim]

            # 가중치의 절댓값 평균을 중요도로 사용
            if len(weights.shape) == 2:
                importance_scores = np.mean(np.abs(weights), axis=1)
            else:
                importance_scores = np.abs(weights)

            # 특성명과 매칭 (길이가 다를 수 있으므로 조정)
            if len(importance_scores) != len(feature_names):
                print(f"⚠️ 가중치 차원({len(importance_scores)})과 특성 수({len(feature_names)})가 다릅니다.")
                # CNN-LSTM 융합 후의 차원이므로 특성명 매칭 불가
                feature_names = [f'Feature_{i}' for i in range(len(importance_scores))]

            # 중요도 정렬
            feature_importance = pd.Series(importance_scores, index=feature_names)
            feature_importance = feature_importance.sort_values(ascending=True)

            # 상위 N개 특성만 표시
            top_features = feature_importance.tail(top_n)

            # 시각화
            plt.figure(figsize=(12, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            bars = plt.barh(range(len(top_features)), top_features.values, color=colors)

            plt.yticks(range(len(top_features)), top_features.index)
            plt.xlabel('Weighted importance (absolute mean)')
            plt.title(f'Top {top_n} feature importance (weighted approximation)')
            plt.grid(True, alpha=0.3)

            # 값 표시
            for i, (bar, value) in enumerate(zip(bars, top_features.values)):
                plt.text(value + max(top_features.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{value:.4f}', va='center', fontsize=9)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ 특성 중요도 계산 실패: {e}")

    def save_results(self, metrics, y_true, y_pred, filepath_prefix='model_results'):
        """
        결과 저장

        Args:
            metrics (dict): 평가 지표
            y_true (np.array): 실제값
            y_pred (np.array): 예측값
            filepath_prefix (str): 저장 파일명 접두사
        """
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 평가 지표 저장
        metrics_file = f"{filepath_prefix}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            # numpy 타입을 일반 타입으로 변환
            metrics_serializable = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_serializable[key] = value.tolist()
                elif isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                    metrics_serializable[key] = float(value)
                else:
                    metrics_serializable[key] = value

            json.dump(metrics_serializable, f, indent=2)

        print(f"✅ 평가 지표 저장: {metrics_file}")

        # 2. 예측 결과 저장
        results_df = pd.DataFrame({
            'actual': y_true.flatten(),
            'predicted': y_pred.flatten(),
            'residual': y_true.flatten() - y_pred.flatten()
        })

        results_file = f"{filepath_prefix}_predictions_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"✅ 예측 결과 저장: {results_file}")

        # 3. 학습 이력 저장 (있는 경우)
        if self.history is not None:
            history_df = pd.DataFrame(self.history.history)
            history_file = f"{filepath_prefix}_history_{timestamp}.csv"
            history_df.to_csv(history_file, index=False)
            print(f"✅ 학습 이력 저장: {history_file}")
        else:
            history_file = None

        return {
            'metrics_file': metrics_file,
            'predictions_file': results_file,
            'history_file': history_file
        }

    def load_and_resume_training(self, model_path, X_train, y_train, X_val, y_val,
                                 additional_epochs=50, **training_params):
        """
        저장된 모델을 로드하여 학습 재개

        Args:
            model_path (str): 모델 파일 경로
            X_train, y_train: 학습 데이터
            X_val, y_val: 검증 데이터
            additional_epochs (int): 추가 학습 에포크
            **training_params: 기타 학습 파라미터

        Returns:
            tf.keras.callbacks.History: 추가 학습 이력
        """
        print(f"📂 모델 로드 중: {model_path}")

        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ 모델 로드 완료")

            # 학습 재개
            print(f"🔄 {additional_epochs} 에포크 추가 학습 시작...")

            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=additional_epochs,
                **training_params
            )

            # 기존 이력과 병합 (있는 경우)
            if self.history is not None:
                for key in history.history.keys():
                    if key in self.history.history:
                        self.history.history[key].extend(history.history[key])
                    else:
                        self.history.history[key] = history.history[key]
            else:
                self.history = history

            print("✅ 추가 학습 완료")
            return history

        except Exception as e:
            print(f"❌ 모델 로드 및 학습 재개 실패: {e}")
            raise


class HyperparameterTuner:
    """하이퍼파라미터 튜닝 클래스"""

    def __init__(self, model_builder):
        self.model_builder = model_builder
        self.best_params = None
        self.best_score = float('inf')
        self.tuning_results = []

    def grid_search(self, X_train, y_train, X_val, y_val, param_grid,
                    scoring='rmse', max_epochs=30):
        """
        그리드 서치를 통한 하이퍼파라미터 튜닝

        Args:
            X_train, y_train: 학습 데이터
            X_val, y_val: 검증 데이터
            param_grid (dict): 파라미터 그리드
            scoring (str): 평가 지표
            max_epochs (int): 최대 에포크 (빠른 평가를 위해)

        Returns:
            dict: 최적 파라미터와 결과
        """
        print("🔍 그리드 서치 하이퍼파라미터 튜닝 시작...")
        print(f"탐색할 조합 수: {np.prod([len(v) for v in param_grid.values()])}")

        from itertools import product

        # 모든 파라미터 조합 생성
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        combination_count = 0
        total_combinations = np.prod([len(v) for v in param_values])

        for combination in product(*param_values):
            combination_count += 1
            params = dict(zip(param_names, combination))

            print(f"\n🧪 조합 {combination_count}/{total_combinations}: {params}")

            try:
                # 모델 생성
                model = self.model_builder(input_shape=X_train.shape[1:], **params)

                # 간단한 컴파일 및 학습
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=max_epochs,
                    batch_size=32,
                    verbose=0
                )

                # 성능 평가
                y_pred = model.predict(X_val, verbose=0)

                if scoring == 'rmse':
                    score = np.sqrt(mean_squared_error(y_val, y_pred))
                elif scoring == 'mae':
                    score = mean_absolute_error(y_val, y_pred)
                elif scoring == 'r2':
                    score = -r2_score(y_val, y_pred)  # 음수로 변환 (최소화 목표)
                else:
                    score = np.sqrt(mean_squared_error(y_val, y_pred))

                # 결과 저장
                result = {
                    'params': params,
                    'score': score,
                    'final_val_loss': history.history['val_loss'][-1]
                }
                self.tuning_results.append(result)

                print(f"  Score ({scoring}): {score:.4f}")

                # 최적 파라미터 업데이트
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = params
                    print(f"  🏆 새로운 최적 파라미터! Score: {score:.4f}")

            except Exception as e:
                print(f"  ❌ 조합 실패: {e}")
                continue

        print("\n" + "=" * 60)
        print("🏆 하이퍼파라미터 튜닝 결과")
        print("=" * 60)
        print(f"최적 파라미터: {self.best_params}")
        print(f"최적 점수 ({scoring}): {self.best_score:.4f}")

        # 상위 5개 결과 출력
        sorted_results = sorted(self.tuning_results, key=lambda x: x['score'])[:5]
        print(f"\n📊 상위 5개 결과:")
        for i, result in enumerate(sorted_results, 1):
            print(f"  {i}. Score: {result['score']:.4f}, Params: {result['params']}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.tuning_results
        }


# 사용 예제
if __name__ == "__main__":
    from cnn_lstm_model import CNNLSTMBuilder

    print("🧪 모델 학습 및 평가 테스트")
    print("=" * 60)

    # 예제 데이터 생성
    np.random.seed(42)
    n_samples, seq_length, n_features = 1000, 24, 20
    X = np.random.randn(n_samples, seq_length, n_features)
    y = np.random.randn(n_samples, 1)

    # 트레이너 초기화
    trainer = ModelTrainer()

    # 데이터 분할
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_train_data(X, y)

    # 모델 구축
    builder = CNNLSTMBuilder()
    model = builder.build_basic_cnn_lstm(input_shape=X.shape[1:])
    model = builder.compile_model(model)

    # 콜백 생성
    callbacks = builder.create_callbacks()

    # 모델 학습
    history = trainer.train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=20, batch_size=32, callbacks=callbacks
    )

    # 모델 평가
    metrics, y_true, y_pred = trainer.evaluate_model(model, X_test, y_test)

    # 결과 시각화
    trainer.plot_training_history()
    trainer.plot_predictions(y_true, y_pred)

    # 결과 저장
    saved_files = trainer.save_results(metrics, y_true, y_pred)

    print("🎉 모든 테스트 완료!")