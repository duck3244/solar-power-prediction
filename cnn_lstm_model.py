#!/usr/bin/env python3
"""
cnn_lstm_model.py
CNN-LSTM 융합 모델 정의 및 구축 모듈
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Dropout, BatchNormalization, Concatenate, Reshape, Flatten,
    Bidirectional, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, CSVLogger
)
import numpy as np

class CNNLSTMBuilder:
    """CNN-LSTM 융합 모델 빌더 클래스"""
    
    def __init__(self):
        self.model = None
        self.model_config = {}
        
    def build_basic_cnn_lstm(self, input_shape, 
                            cnn_filters=[64, 32], 
                            lstm_units=[64, 32],
                            dense_units=[64, 32],
                            dropout_rate=0.2):
        """
        기본 CNN-LSTM 모델 구축
        
        Args:
            input_shape (tuple): 입력 형태 (sequence_length, features)
            cnn_filters (list): CNN 필터 수 리스트
            lstm_units (list): LSTM 유닛 수 리스트
            dense_units (list): Dense 레이어 유닛 수 리스트
            dropout_rate (float): 드롭아웃 비율
            
        Returns:
            tf.keras.Model: 구축된 모델
        """
        print("🧠 기본 CNN-LSTM 모델 구축 중...")
        
        # 입력 레이어
        inputs = Input(shape=input_shape, name='input_layer')
        
        # CNN 브랜치 (패턴 인식)
        cnn_branch = inputs
        for i, filters in enumerate(cnn_filters):
            cnn_branch = Conv1D(
                filters=filters, 
                kernel_size=3, 
                activation='relu', 
                padding='same',
                name=f'conv1d_{i+1}'
            )(cnn_branch)
            cnn_branch = BatchNormalization(name=f'batch_norm_cnn_{i+1}')(cnn_branch)
            cnn_branch = Dropout(dropout_rate, name=f'dropout_cnn_{i+1}')(cnn_branch)
        
        # CNN 출력을 위한 Global Max Pooling
        cnn_output = GlobalMaxPooling1D(name='global_maxpool')(cnn_branch)
        
        # LSTM 브랜치 (시계열 의존성)
        lstm_branch = inputs
        for i, units in enumerate(lstm_units[:-1]):
            lstm_branch = LSTM(
                units=units, 
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                name=f'lstm_{i+1}'
            )(lstm_branch)
        
        # 마지막 LSTM 레이어 (return_sequences=False)
        lstm_output = LSTM(
            units=lstm_units[-1],
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name=f'lstm_{len(lstm_units)}'
        )(lstm_branch)
        
        # 브랜치 융합
        merged = Concatenate(name='concatenate')([cnn_output, lstm_output])
        
        # 융합 레이어
        fusion = merged
        for i, units in enumerate(dense_units):
            fusion = Dense(
                units, 
                activation='relu', 
                name=f'dense_{i+1}'
            )(fusion)
            fusion = Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(fusion)
        
        # 출력 레이어
        outputs = Dense(1, activation='linear', name='output')(fusion)
        
        # 모델 생성
        model = Model(inputs=inputs, outputs=outputs, name='BasicCNN_LSTM')
        
        self.model_config = {
            'type': 'basic_cnn_lstm',
            'input_shape': input_shape,
            'cnn_filters': cnn_filters,
            'lstm_units': lstm_units,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate
        }
        
        print(f"✅ 기본 모델 구축 완료 - 파라미터: {model.count_params():,}")
        return model
    
    def build_advanced_cnn_lstm(self, input_shape,
                               cnn_filters=[64, 64, 32],
                               lstm_units=[128, 64],
                               dense_units=[128, 64, 32],
                               dropout_rate=0.3,
                               use_attention=True,
                               use_bidirectional=True):
        """
        고급 CNN-LSTM 모델 구축 (Attention, Bidirectional 포함)
        
        Args:
            input_shape (tuple): 입력 형태
            cnn_filters (list): CNN 필터 수
            lstm_units (list): LSTM 유닛 수
            dense_units (list): Dense 레이어 유닛 수
            dropout_rate (float): 드롭아웃 비율
            use_attention (bool): Attention 메커니즘 사용 여부
            use_bidirectional (bool): Bidirectional LSTM 사용 여부
            
        Returns:
            tf.keras.Model: 구축된 모델
        """
        print("🚀 고급 CNN-LSTM 모델 구축 중...")
        
        inputs = Input(shape=input_shape, name='input_layer')
        
        # Multi-scale CNN 브랜치
        cnn_branches = []
        kernel_sizes = [3, 5, 7]
        
        for i, kernel_size in enumerate(kernel_sizes):
            branch = inputs
            for j, filters in enumerate(cnn_filters):
                branch = Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    padding='same',
                    name=f'conv1d_k{kernel_size}_{j+1}'
                )(branch)
                branch = BatchNormalization(name=f'bn_k{kernel_size}_{j+1}')(branch)
                branch = Dropout(dropout_rate, name=f'dropout_k{kernel_size}_{j+1}')(branch)
            
            # 각 브랜치의 출력
            branch_output = GlobalMaxPooling1D(name=f'global_pool_k{kernel_size}')(branch)
            cnn_branches.append(branch_output)
        
        # CNN 브랜치 결합
        if len(cnn_branches) > 1:
            cnn_combined = Concatenate(name='cnn_concat')(cnn_branches)
        else:
            cnn_combined = cnn_branches[0]
        
        # LSTM 브랜치
        lstm_branch = inputs
        
        for i, units in enumerate(lstm_units[:-1]):
            if use_bidirectional:
                lstm_branch = Bidirectional(
                    LSTM(
                        units=units//2,  # Bidirectional이므로 절반 크기 사용
                        return_sequences=True,
                        dropout=dropout_rate,
                        recurrent_dropout=dropout_rate
                    ),
                    name=f'bi_lstm_{i+1}'
                )(lstm_branch)
            else:
                lstm_branch = LSTM(
                    units=units,
                    return_sequences=True,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    name=f'lstm_{i+1}'
                )(lstm_branch)
        
        # 마지막 LSTM 레이어 + Attention 메커니즘
        if use_attention:
            # Attention 사용 시 시퀀스 출력 필요 (return_sequences=True)
            if use_bidirectional:
                lstm_seq = Bidirectional(
                    LSTM(
                        units=lstm_units[-1]//2,
                        return_sequences=True,
                        dropout=dropout_rate,
                        recurrent_dropout=dropout_rate
                    ),
                    name=f'bi_lstm_{len(lstm_units)}'
                )(lstm_branch)
            else:
                lstm_seq = LSTM(
                    units=lstm_units[-1],
                    return_sequences=True,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    name=f'lstm_{len(lstm_units)}'
                )(lstm_branch)

            # 각 타임스텝에 대한 attention score 계산 → softmax → 가중합
            attention_scores = Dense(1, activation='tanh', name='attention_score')(lstm_seq)
            attention_weights = tf.keras.layers.Softmax(axis=1, name='attention_softmax')(attention_scores)
            context = tf.keras.layers.Multiply(name='attention_applied')([lstm_seq, attention_weights])
            lstm_output = tf.keras.layers.Lambda(
                lambda x: tf.reduce_sum(x, axis=1), name='attention_reduce'
            )(context)
        else:
            if use_bidirectional:
                lstm_output = Bidirectional(
                    LSTM(
                        units=lstm_units[-1]//2,
                        dropout=dropout_rate,
                        recurrent_dropout=dropout_rate
                    ),
                    name=f'bi_lstm_{len(lstm_units)}'
                )(lstm_branch)
            else:
                lstm_output = LSTM(
                    units=lstm_units[-1],
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    name=f'lstm_{len(lstm_units)}'
                )(lstm_branch)
        
        # 브랜치 융합
        merged = Concatenate(name='final_concat')([cnn_combined, lstm_output])
        
        # 융합 레이어들
        fusion = merged
        for i, units in enumerate(dense_units):
            fusion = Dense(units, activation='relu', name=f'fusion_dense_{i+1}')(fusion)
            fusion = BatchNormalization(name=f'fusion_bn_{i+1}')(fusion)
            fusion = Dropout(dropout_rate, name=f'fusion_dropout_{i+1}')(fusion)
        
        # 출력 레이어
        outputs = Dense(1, activation='linear', name='output')(fusion)
        
        # 모델 생성
        model = Model(inputs=inputs, outputs=outputs, name='AdvancedCNN_LSTM')
        
        self.model_config = {
            'type': 'advanced_cnn_lstm',
            'input_shape': input_shape,
            'cnn_filters': cnn_filters,
            'lstm_units': lstm_units,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate,
            'use_attention': use_attention,
            'use_bidirectional': use_bidirectional
        }
        
        print(f"✅ 고급 모델 구축 완료 - 파라미터: {model.count_params():,}")
        return model
    
    def build_transformer_cnn_lstm(self, input_shape,
                                  cnn_filters=[64, 32],
                                  lstm_units=[64, 32],
                                  transformer_heads=4,
                                  transformer_dim=64,
                                  dense_units=[64, 32],
                                  dropout_rate=0.2):
        """
        Transformer + CNN-LSTM 하이브리드 모델 구축
        
        Args:
            input_shape (tuple): 입력 형태
            cnn_filters (list): CNN 필터 수
            lstm_units (list): LSTM 유닛 수
            transformer_heads (int): Multi-head Attention의 헤드 수
            transformer_dim (int): Transformer 차원
            dense_units (list): Dense 레이어 유닛 수
            dropout_rate (float): 드롭아웃 비율
            
        Returns:
            tf.keras.Model: 구축된 모델
        """
        print("🔮 Transformer-CNN-LSTM 하이브리드 모델 구축 중...")
        
        inputs = Input(shape=input_shape, name='input_layer')
        
        # Transformer 브랜치
        transformer_branch = inputs
        
        # Multi-head Self-attention
        attention_output = MultiHeadAttention(
            num_heads=transformer_heads,
            key_dim=transformer_dim,
            name='multi_head_attention'
        )(transformer_branch, transformer_branch)
        
        # Add & Norm
        attention_output = Add(name='add_1')([inputs, attention_output])
        attention_output = LayerNormalization(name='layer_norm_1')(attention_output)
        
        # Feed Forward Network
        ffn = Dense(transformer_dim * 2, activation='relu', name='ffn_1')(attention_output)
        ffn = Dense(input_shape[-1], name='ffn_2')(ffn)
        
        # Add & Norm
        transformer_output = Add(name='add_2')([attention_output, ffn])
        transformer_output = LayerNormalization(name='layer_norm_2')(transformer_output)
        
        # CNN 브랜치
        cnn_branch = transformer_output  # Transformer 출력을 CNN 입력으로 사용
        for i, filters in enumerate(cnn_filters):
            cnn_branch = Conv1D(
                filters=filters,
                kernel_size=3,
                activation='relu',
                padding='same',
                name=f'conv1d_{i+1}'
            )(cnn_branch)
            cnn_branch = BatchNormalization(name=f'cnn_bn_{i+1}')(cnn_branch)
            cnn_branch = Dropout(dropout_rate, name=f'cnn_dropout_{i+1}')(cnn_branch)
        
        cnn_output = GlobalMaxPooling1D(name='cnn_global_pool')(cnn_branch)
        
        # LSTM 브랜치
        lstm_branch = transformer_output
        for i, units in enumerate(lstm_units[:-1]):
            lstm_branch = LSTM(
                units=units,
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=dropout_rate,
                name=f'lstm_{i+1}'
            )(lstm_branch)
        
        lstm_output = LSTM(
            units=lstm_units[-1],
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name=f'lstm_{len(lstm_units)}'
        )(lstm_branch)
        
        # 모든 브랜치 융합
        merged = Concatenate(name='final_merge')([cnn_output, lstm_output])
        
        # 융합 레이어
        fusion = merged
        for i, units in enumerate(dense_units):
            fusion = Dense(units, activation='relu', name=f'fusion_{i+1}')(fusion)
            fusion = Dropout(dropout_rate, name=f'fusion_dropout_{i+1}')(fusion)
        
        # 출력
        outputs = Dense(1, activation='linear', name='output')(fusion)
        
        model = Model(inputs=inputs, outputs=outputs, name='TransformerCNN_LSTM')
        
        self.model_config = {
            'type': 'transformer_cnn_lstm',
            'input_shape': input_shape,
            'cnn_filters': cnn_filters,
            'lstm_units': lstm_units,
            'transformer_heads': transformer_heads,
            'transformer_dim': transformer_dim,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate
        }
        
        print(f"✅ Transformer 하이브리드 모델 구축 완료 - 파라미터: {model.count_params():,}")
        return model
    
    def compile_model(self, model, 
                     optimizer='adam',
                     learning_rate=0.001,
                     loss='mse',
                     metrics=['mae', 'mape']):
        """
        모델 컴파일
        
        Args:
            model (tf.keras.Model): 컴파일할 모델
            optimizer (str): 옵티마이저
            learning_rate (float): 학습률
            loss (str): 손실 함수
            metrics (list): 평가 지표
            
        Returns:
            tf.keras.Model: 컴파일된 모델
        """
        print(f"⚙️ 모델 컴파일 중 (optimizer: {optimizer}, lr: {learning_rate})...")
        
        # 옵티마이저 설정
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        # 손실 함수 설정 (Huber loss for robustness)
        if loss == 'huber':
            loss_fn = tf.keras.losses.Huber(delta=1.0)
        else:
            loss_fn = loss
        
        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=metrics
        )
        
        print("✅ 모델 컴파일 완료")
        return model
    
    def create_callbacks(self, monitor='val_loss',
                        patience=15,
                        reduce_lr_patience=10,
                        min_lr=1e-6,
                        save_best_only=True,
                        save_path='best_model.h5'):
        """
        콜백 함수들 생성
        
        Args:
            monitor (str): 모니터링할 지표
            patience (int): Early stopping patience
            reduce_lr_patience (int): Learning rate reduction patience
            min_lr (float): 최소 학습률
            save_best_only (bool): 최고 성능 모델만 저장
            save_path (str): 모델 저장 경로
            
        Returns:
            list: 콜백 함수 리스트
        """
        callbacks = []
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode='min' if 'loss' in monitor else 'max'
        )
        callbacks.append(early_stopping)
        
        # Reduce Learning Rate on Plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
            mode='min' if 'loss' in monitor else 'max'
        )
        callbacks.append(reduce_lr)
        
        # Model Checkpoint
        checkpoint = ModelCheckpoint(
            filepath=save_path,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            verbose=1,
            mode='min' if 'loss' in monitor else 'max'
        )
        callbacks.append(checkpoint)
        
        # CSV Logger
        import os
        log_dir = os.path.dirname(save_path) or '.'
        csv_logger = CSVLogger(os.path.join(log_dir, 'training_log.csv'), append=True)
        callbacks.append(csv_logger)
        
        print(f"✅ 콜백 함수 생성 완료: {len(callbacks)}개")
        return callbacks
    
    def get_model_summary(self, model):
        """
        모델 구조 요약 정보 출력
        
        Args:
            model (tf.keras.Model): 요약할 모델
        """
        print("="*60)
        print("🏗️ 모델 구조 요약")
        print("="*60)
        
        model.summary()
        
        print(f"\n📊 모델 정보:")
        print(f"  - 모델명: {model.name}")
        print(f"  - 총 파라미터: {model.count_params():,}")
        print(f"  - 학습 가능 파라미터: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
        print(f"  - 비학습 파라미터: {sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
        print(f"  - 레이어 수: {len(model.layers)}")
        
        # 설정 정보 출력
        if self.model_config:
            print(f"\n⚙️ 모델 설정:")
            for key, value in self.model_config.items():
                print(f"  - {key}: {value}")
    
    def save_model_architecture(self, model, filepath='model_architecture.png'):
        """
        모델 구조 다이어그램 저장
        
        Args:
            model (tf.keras.Model): 저장할 모델
            filepath (str): 저장 경로
        """
        try:
            tf.keras.utils.plot_model(
                model,
                to_file=filepath,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=150
            )
            print(f"✅ 모델 구조 다이어그램이 '{filepath}'에 저장되었습니다.")
        except Exception as e:
            print(f"⚠️ 모델 구조 다이어그램 저장 실패: {e}")
            print("💡 graphviz 패키지 설치가 필요할 수 있습니다: pip install graphviz")

class CustomLosses:
    """커스텀 손실 함수들"""
    
    @staticmethod
    def weighted_mse(alpha=1.0):
        """
        시간대별 가중치를 적용한 MSE 손실
        
        Args:
            alpha (float): 가중치 강도
            
        Returns:
            function: 손실 함수
        """
        def loss(y_true, y_pred):
            # 높은 값에 더 큰 가중치 적용
            weights = 1 + alpha * tf.abs(y_true) / tf.reduce_max(tf.abs(y_true))
            mse = tf.square(y_true - y_pred)
            weighted_mse = mse * weights
            return tf.reduce_mean(weighted_mse)
        
        return loss
    
    @staticmethod
    def quantile_loss(quantile=0.5):
        """
        분위수 손실 (Quantile Loss)
        
        Args:
            quantile (float): 분위수 (0~1)
            
        Returns:
            function: 손실 함수
        """
        def loss(y_true, y_pred):
            error = y_true - y_pred
            return tf.reduce_mean(
                tf.maximum(quantile * error, (quantile - 1) * error)
            )
        
        return loss
    
    @staticmethod
    def focal_mse(gamma=2.0):
        """
        Focal MSE 손실 (어려운 샘플에 집중)
        
        Args:
            gamma (float): 포커싱 파라미터
            
        Returns:
            function: 손실 함수
        """
        def loss(y_true, y_pred):
            mse = tf.square(y_true - y_pred)
            # 정규화된 오차
            normalized_error = mse / (tf.reduce_max(mse) + 1e-8)
            # Focal weight
            focal_weight = tf.pow(normalized_error, gamma)
            return tf.reduce_mean(focal_weight * mse)
        
        return loss

class ModelEnsemble:
    """모델 앙상블 클래스"""
    
    def __init__(self):
        self.models = []
        self.weights = []
    
    def add_model(self, model, weight=1.0):
        """
        앙상블에 모델 추가
        
        Args:
            model (tf.keras.Model): 추가할 모델
            weight (float): 모델 가중치
        """
        self.models.append(model)
        self.weights.append(weight)
    
    def predict(self, X):
        """
        앙상블 예측
        
        Args:
            X (np.array): 입력 데이터
            
        Returns:
            np.array: 가중 평균 예측값
        """
        if not self.models:
            raise ValueError("앙상블에 모델이 없습니다.")
        
        predictions = []
        total_weight = sum(self.weights)
        
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X, verbose=0)
            predictions.append(pred * (weight / total_weight))
        
        return np.sum(predictions, axis=0)
    
    def save_ensemble(self, filepath_prefix='ensemble_model'):
        """
        앙상블 모델들 저장
        
        Args:
            filepath_prefix (str): 파일명 접두사
        """
        for i, model in enumerate(self.models):
            filepath = f"{filepath_prefix}_{i}.h5"
            model.save(filepath)
            print(f"모델 {i} 저장: {filepath}")
        
        # 가중치 정보 저장
        weights_info = {
            'weights': self.weights,
            'model_count': len(self.models)
        }
        
        import json
        with open(f"{filepath_prefix}_weights.json", 'w') as f:
            json.dump(weights_info, f)
        
        print(f"앙상블 가중치 저장: {filepath_prefix}_weights.json")

# 사용 예제
if __name__ == "__main__":
    # 모델 빌더 초기화
    builder = CNNLSTMBuilder()
    
    # 예제 입력 형태 (24 시간 시퀀스, 20개 특성)
    input_shape = (24, 20)
    
    print("🔨 다양한 CNN-LSTM 모델 구축 테스트")
    print("="*60)
    
    # 1. 기본 CNN-LSTM 모델
    print("\n1️⃣ 기본 CNN-LSTM 모델")
    basic_model = builder.build_basic_cnn_lstm(input_shape)
    basic_model = builder.compile_model(basic_model)
    builder.get_model_summary(basic_model)
    
    # 2. 고급 CNN-LSTM 모델  
    print("\n2️⃣ 고급 CNN-LSTM 모델")
    advanced_model = builder.build_advanced_cnn_lstm(
        input_shape,
        use_attention=True,
        use_bidirectional=True
    )
    advanced_model = builder.compile_model(advanced_model, loss='huber')
    builder.get_model_summary(advanced_model)
    
    # 3. Transformer-CNN-LSTM 하이브리드
    print("\n3️⃣ Transformer-CNN-LSTM 하이브리드")
    hybrid_model = builder.build_transformer_cnn_lstm(input_shape)
    hybrid_model = builder.compile_model(hybrid_model)
    builder.get_model_summary(hybrid_model)
    
    # 4. 콜백 함수 생성
    print("\n4️⃣ 콜백 함수 생성")
    callbacks = builder.create_callbacks()
    
    # 5. 앙상블 예제
    print("\n5️⃣ 모델 앙상블 예제")
    ensemble = ModelEnsemble()
    ensemble.add_model(basic_model, weight=0.4)
    ensemble.add_model(advanced_model, weight=0.6)
    
    print("🎉 모든 모델 구축 완료!")
