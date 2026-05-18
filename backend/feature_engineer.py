#!/usr/bin/env python3
"""
feature_engineer.py
태양광 발전 데이터 특성 엔지니어링 모듈
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class SolarFeatureEngineer:
    """태양광 발전 데이터 특성 엔지니어링 클래스"""
    
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_columns = []
        
    def create_time_features(self, df):
        """
        시간 기반 특성 생성
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            
        Returns:
            pd.DataFrame: 시간 특성이 추가된 데이터프레임
        """
        print("🕐 시간 특성 생성 중...")
        
        df = df.copy()
        
        # 기본 시간 특성
        df['hour'] = df['DATE_TIME'].dt.hour
        df['day_of_week'] = df['DATE_TIME'].dt.dayofweek  # 0=월요일
        df['day_of_month'] = df['DATE_TIME'].dt.day
        df['month'] = df['DATE_TIME'].dt.month
        df['day_of_year'] = df['DATE_TIME'].dt.dayofyear
        df['week_of_year'] = df['DATE_TIME'].dt.isocalendar().week
        
        # 주말 여부
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 계절 구분
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,    # 겨울
            3: 1, 4: 1, 5: 1,     # 봄
            6: 2, 7: 2, 8: 2,     # 여름
            9: 3, 10: 3, 11: 3    # 가을
        })
        
        print("✅ 기본 시간 특성 생성 완료")
        return df
    
    def create_cyclical_features(self, df):
        """
        순환 특성 생성 (sin/cos 변환)
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            
        Returns:
            pd.DataFrame: 순환 특성이 추가된 데이터프레임
        """
        print("🔄 순환 특성 생성 중...")
        
        df = df.copy()
        
        # 시간 순환 인코딩 (24시간)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 요일 순환 인코딩 (7일)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 월 순환 인코딩 (12개월)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 일년 중 날짜 순환 인코딩 (365일)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        print("✅ 순환 특성 생성 완료")
        return df
    
    def create_lag_features(self, df, columns=None, lags=[1, 3, 6, 12]):
        """
        래그 특성 생성 (이전 시점의 값)
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            columns (list): 래그를 생성할 컬럼명 리스트
            lags (list): 래그 시점 리스트
            
        Returns:
            pd.DataFrame: 래그 특성이 추가된 데이터프레임
        """
        print("📉 래그 특성 생성 중...")
        
        if columns is None:
            columns = ['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE']
        
        df = df.copy()
        df = df.sort_values('DATE_TIME')  # 시간순 정렬 확인
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_LAG{lag}'] = df[col].shift(lag)
        
        print(f"✅ 래그 특성 생성 완료: {len(columns)}개 변수 × {len(lags)}개 시점")
        return df
    
    def create_rolling_features(self, df, columns=None, windows=[3, 6, 12, 24]):
        """
        롤링 윈도우 특성 생성 (이동평균, 표준편차 등)
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            columns (list): 롤링 특성을 생성할 컬럼명 리스트
            windows (list): 윈도우 크기 리스트
            
        Returns:
            pd.DataFrame: 롤링 특성이 추가된 데이터프레임
        """
        print("📊 롤링 윈도우 특성 생성 중...")
        
        if columns is None:
            columns = ['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE']
        
        df = df.copy()
        df = df.sort_values('DATE_TIME')
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    # 이동평균
                    df[f'{col}_MA{window}'] = df[col].rolling(
                        window=window, min_periods=1
                    ).mean()
                    
                    # 이동표준편차
                    df[f'{col}_STD{window}'] = df[col].rolling(
                        window=window, min_periods=1
                    ).std()
                    
                    # 이동최대값
                    df[f'{col}_MAX{window}'] = df[col].rolling(
                        window=window, min_periods=1
                    ).max()
                    
                    # 이동최소값
                    df[f'{col}_MIN{window}'] = df[col].rolling(
                        window=window, min_periods=1
                    ).min()
        
        print(f"✅ 롤링 특성 생성 완료: {len(columns)}개 변수 × {len(windows)}개 윈도우")
        return df
    
    def create_power_efficiency_features(self, df):
        """
        발전 효율성 관련 특성 생성
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            
        Returns:
            pd.DataFrame: 효율성 특성이 추가된 데이터프레임
        """
        print("⚡ 발전 효율성 특성 생성 중...")
        
        df = df.copy()
        
        # DC-AC 변환 효율
        df['DC_AC_EFFICIENCY'] = np.where(
            df['DC_POWER'] > 0,
            df['AC_POWER'] / df['DC_POWER'],
            0
        )
        
        # 일조량 대비 발전 효율
        df['POWER_PER_IRRADIATION'] = np.where(
            df['IRRADIATION'] > 0,
            df['AC_POWER'] / df['IRRADIATION'],
            0
        )
        
        # 온도 대비 발전량
        df['POWER_PER_TEMPERATURE'] = np.where(
            df['AMBIENT_TEMPERATURE'] > 0,
            df['AC_POWER'] / df['AMBIENT_TEMPERATURE'],
            0
        )
        
        # 모듈 온도와 외기온도 차이
        if 'MODULE_TEMPERATURE' in df.columns:
            df['TEMP_DIFFERENCE'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']
        
        # 발전량 정규화 (시간대별)
        hourly_max = df.groupby('hour')['AC_POWER'].transform('max')
        df['NORMALIZED_POWER'] = np.where(
            hourly_max > 0,
            df['AC_POWER'] / hourly_max,
            0
        )
        
        print("✅ 발전 효율성 특성 생성 완료")
        return df
    
    def create_weather_interaction_features(self, df):
        """
        기상 변수 간 상호작용 특성 생성
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            
        Returns:
            pd.DataFrame: 상호작용 특성이 추가된 데이터프레임
        """
        print("🌤️ 기상 상호작용 특성 생성 중...")
        
        df = df.copy()
        
        # 일조량 × 온도 상호작용
        df['IRRADIATION_TEMP_INTERACTION'] = df['IRRADIATION'] * df['AMBIENT_TEMPERATURE']
        
        # 온도 제곱 (비선형 효과)
        df['TEMPERATURE_SQUARED'] = df['AMBIENT_TEMPERATURE'] ** 2
        
        # 일조량 제곱
        df['IRRADIATION_SQUARED'] = df['IRRADIATION'] ** 2
        
        # 기상 지수 (종합 지수)
        df['WEATHER_INDEX'] = (
            df['IRRADIATION'] * 0.6 + 
            df['AMBIENT_TEMPERATURE'] * 0.4
        ) / 100
        
        print("✅ 기상 상호작용 특성 생성 완료")
        return df
    
    def create_target_variable(self, df, demand_pattern='realistic', base_demand_kw=8000.0):
        """
        전력 수요 타겟 변수 생성 (실제 환경에서는 외부 데이터 사용)

        주의: 타겟은 입력 특성(AC_POWER, DC_POWER 등 발전량)과
        독립적으로 합성되어야 한다. 그렇지 않으면 모델이 타겟을
        역산할 수 있어 데이터 누수가 발생한다.

        Args:
            df (pd.DataFrame): 입력 데이터프레임
            demand_pattern (str): 수요 패턴 타입 ('realistic', 'simple')
            base_demand_kw (float): 시간대/요일 보정 전의 기본 수요(kW)

        Returns:
            pd.DataFrame: 타겟 변수가 추가된 데이터프레임
        """
        print("🎯 전력 수요 타겟 변수 생성 중...")

        df = df.copy()

        if demand_pattern == 'realistic':
            # 시간대별 패턴 (아침/저녁 피크, 정오 밸리)
            morning_peak = np.where((df['hour'] >= 7) & (df['hour'] <= 9), 1.4, 1.0)
            evening_peak = np.where((df['hour'] >= 17) & (df['hour'] <= 19), 1.6, 1.0)
            midday_valley = np.where((df['hour'] >= 13) & (df['hour'] <= 15), 0.8, 1.0)
            time_factor = morning_peak * evening_peak * midday_valley

            # 요일별 패턴 (주말 수요 감소)
            weekend_factor = np.where(df['is_weekend'] == 1, 0.7, 1.0)

            # 계절별 패턴 (여름철 냉방 수요 증가)
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * df['day_of_year'] / 365 + np.pi/2)

            # 기상 영향 (온도가 높을수록 냉방 수요 증가)
            weather_factor = 1 + 0.01 * np.maximum(df['AMBIENT_TEMPERATURE'] - 25, 0)

            # 노이즈 추가
            noise_factor = np.random.normal(1, 0.05, len(df))

            # 최종 전력 수요 — AC_POWER에 의존하지 않음
            df['POWER_DEMAND'] = (
                base_demand_kw * time_factor * weekend_factor *
                seasonal_factor * weather_factor * noise_factor
            )

        else:
            # 간단한 패턴 (시간별 sin + 노이즈)
            hourly_shape = 1.0 + 0.4 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)
            df['POWER_DEMAND'] = base_demand_kw * hourly_shape * (
                1.0 + 0.1 * np.random.standard_normal(len(df))
            )

        # 음수값 제거
        df['POWER_DEMAND'] = np.maximum(df['POWER_DEMAND'], 0)

        print("✅ 전력 수요 타겟 변수 생성 완료")
        return df
    
    def select_features(self, df, feature_selection_method='correlation', target_col='POWER_DEMAND'):
        """
        특성 선택
        
        Args:
            df (pd.DataFrame): 입력 데이터프레임
            feature_selection_method (str): 특성 선택 방법
            target_col (str): 타겟 변수명
            
        Returns:
            tuple: (selected_features, feature_importance)
        """
        print("🎯 특성 선택 중...")
        
        # 숫자형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 타겟 변수 제외
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # 시간 관련 정수형 컬럼 제외 (이미 순환 인코딩된 버전 사용)
        exclude_cols = ['hour', 'day_of_week', 'day_of_month', 'month', 
                       'day_of_year', 'week_of_year', 'season']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if feature_selection_method == 'correlation':
            # 상관관계 기반 특성 선택
            corr_with_target = df[feature_cols + [target_col]].corr()[target_col].abs()
            selected_features = corr_with_target[corr_with_target > 0.1].index.tolist()
            if target_col in selected_features:
                selected_features.remove(target_col)
            feature_importance = corr_with_target[selected_features].sort_values(ascending=False)
            
        elif feature_selection_method == 'variance':
            # 분산 기반 특성 선택 (분산이 너무 작은 특성 제거)
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            X = df[feature_cols].fillna(0)
            selector.fit(X)
            selected_features = [feature_cols[i] for i in range(len(feature_cols)) 
                               if selector.variances_[i] > 0.01]
            feature_importance = pd.Series(selector.variances_, 
                                         index=feature_cols).sort_values(ascending=False)
            
        else:
            # 기본: 모든 특성 사용
            selected_features = feature_cols
            feature_importance = pd.Series(1.0, index=feature_cols)
        
        self.feature_columns = selected_features
        print(f"✅ 특성 선택 완료: {len(selected_features)}개 특성 선택됨")
        
        return selected_features, feature_importance
    
    def fit_scalers(self, train_df, feature_cols, target_col='POWER_DEMAND', method='minmax'):
        """
        train 데이터에 스케일러를 fit (val/test는 transform만 적용해야 함).

        Args:
            train_df (pd.DataFrame): 학습 구간 데이터 (결측 없음 가정)
            feature_cols (list): 정규화할 특성 컬럼
            target_col (str): 타겟 변수명
            method (str): 정규화 방법 ('minmax', 'standard')

        Returns:
            tuple: (feature_scaler, target_scaler)
        """
        if method == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        else:
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()

        self.feature_scaler.fit(train_df[feature_cols])
        self.target_scaler.fit(train_df[[target_col]])

        return self.feature_scaler, self.target_scaler

    def transform_split(self, df, feature_cols, target_col='POWER_DEMAND'):
        """이미 fit된 스케일러를 사용해 특성/타겟 변환."""
        features_normalized = self.feature_scaler.transform(df[feature_cols])
        target_normalized = self.target_scaler.transform(df[[target_col]])
        return features_normalized, target_normalized
    
    def create_sequences(self, features, target, sequence_length=24):
        """
        시계열 시퀀스 생성
        
        Args:
            features (np.array): 정규화된 특성 배열
            target (np.array): 정규화된 타겟 배열
            sequence_length (int): 시퀀스 길이
            
        Returns:
            tuple: (X_sequences, y_sequences)
        """
        print(f"🔗 시계열 시퀀스 생성 중 (length: {sequence_length})...")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(features)):
            # 특성 시퀀스
            X_sequences.append(features[i-sequence_length:i])
            # 타겟값 (현재 시점)
            y_sequences.append(target[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"✅ 시퀀스 생성 완료: X shape {X_sequences.shape}, y shape {y_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def get_feature_importance_plot(self, feature_importance, top_n=20):
        """
        특성 중요도 시각화
        
        Args:
            feature_importance (pd.Series): 특성 중요도
            top_n (int): 표시할 상위 특성 개수
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('중요도')
        plt.title(f'상위 {top_n}개 특성 중요도')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def feature_engineering_pipeline(self, df, sequence_length=24,
                                   feature_selection_method='correlation',
                                   train_ratio=0.7, val_ratio=0.15,
                                   normalization_method='minmax'):
        """
        전체 특성 엔지니어링 파이프라인.

        시간 순서 분할(train→val→test) 후 train 구간에서만 스케일러를 fit하여
        검증/테스트 통계 누수를 방지한다. lag/rolling으로 생긴 결측 행은
        dropna로 제거한다.

        Args:
            df (pd.DataFrame): 입력 데이터프레임
            sequence_length (int): 시퀀스 길이
            feature_selection_method (str): 특성 선택 방법
            train_ratio (float): 학습 구간 비율
            val_ratio (float): 검증 구간 비율 (test는 1 - train - val)
            normalization_method (str): 'minmax' 또는 'standard'

        Returns:
            tuple: (splits, selected_features, scalers)
                splits: dict {'X_train', 'y_train', 'X_val', 'y_val',
                              'X_test', 'y_test'}
                scalers: {'feature_scaler', 'target_scaler'}
        """
        print("🚀 특성 엔지니어링 파이프라인 시작")
        print("="*60)

        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio <= 0:
            raise ValueError("train_ratio + val_ratio 가 1.0 미만이어야 합니다.")

        # 1~6. 특성 생성
        df = self.create_time_features(df)
        df = self.create_cyclical_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_power_efficiency_features(df)
        df = self.create_weather_interaction_features(df)

        # 7. 타겟 변수 생성
        df = self.create_target_variable(df)

        # 8. 시간순 정렬 및 lag/rolling으로 생긴 결측 제거
        df = df.sort_values('DATE_TIME').reset_index(drop=True)
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"🧹 결측 행 제거: {before} → {len(df)} (lag/rolling NaN 포함)")

        # 9. 특성 선택 (전체 분포 기준 — 누수가 아닌, 컬럼 결정용)
        selected_features, feature_importance = self.select_features(
            df, feature_selection_method
        )

        # 10. 시간 기준 split
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        print(f"📂 시간순 분할: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # 11. 스케일러 fit (train only) 후 각 split 변환
        self.fit_scalers(train_df, selected_features, method=normalization_method)
        train_X, train_y = self.transform_split(train_df, selected_features)
        val_X, val_y = self.transform_split(val_df, selected_features)
        test_X, test_y = self.transform_split(test_df, selected_features)

        # 12. 각 split 내부에서 시퀀스 생성 (경계 누수 없음)
        X_train, y_train = self.create_sequences(train_X, train_y, sequence_length)
        X_val, y_val = self.create_sequences(val_X, val_y, sequence_length)
        X_test, y_test = self.create_sequences(test_X, test_y, sequence_length)

        print("✅ 특성 엔지니어링 파이프라인 완료")
        print(f"📊 최종 결과:")
        print(f"  - 선택된 특성: {len(selected_features)}개")
        print(f"  - 시퀀스 길이: {sequence_length}")
        print(f"  - 시퀀스 수: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        print(f"\n🏆 상위 10개 중요 특성:")
        for i, (feature, importance) in enumerate(feature_importance.head(10).items()):
            print(f"  {i+1:2d}. {feature}: {importance:.4f}")

        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
        }
        scalers = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
        }
        return splits, selected_features, scalers
    
    def inverse_transform_target(self, normalized_target):
        """
        타겟 변수 역정규화
        
        Args:
            normalized_target (np.array): 정규화된 타겟값
            
        Returns:
            np.array: 원래 스케일의 타겟값
        """
        return self.target_scaler.inverse_transform(normalized_target)
    
    def transform_new_features(self, df):
        """
        새로운 데이터에 대해 학습된 스케일러로 특성 변환.

        lag/rolling으로 생기는 결측은 dropna로 제거한다.

        Args:
            df (pd.DataFrame): 새로운 데이터

        Returns:
            np.array: 변환된 특성
        """
        if not self.feature_columns:
            raise ValueError("특성이 선택되지 않았습니다. fit을 먼저 실행하세요.")

        df = self.create_time_features(df)
        df = self.create_cyclical_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_power_efficiency_features(df)
        df = self.create_weather_interaction_features(df)

        df_clean = df[self.feature_columns].dropna()
        features_normalized = self.feature_scaler.transform(df_clean)

        return features_normalized

# 사용 예제
if __name__ == "__main__":
    # 예제 데이터 로드 (data_loader 모듈 사용)
    from data_loader import SolarDataLoader
    
    loader = SolarDataLoader()
    df = loader.preprocess_pipeline(
        'Plant_1_Generation_Data.csv',
        'Plant_1_Weather_Sensor_Data.csv'
    )
    
    # 특성 엔지니어링 실행
    engineer = SolarFeatureEngineer()
    splits, features, scalers = engineer.feature_engineering_pipeline(df)

    print(f"🎉 특성 엔지니어링 완료!")
    print(f"  - X_train: {splits['X_train'].shape}, y_train: {splits['y_train'].shape}")
    print(f"  - X_val:   {splits['X_val'].shape},   y_val:   {splits['y_val'].shape}")
    print(f"  - X_test:  {splits['X_test'].shape},  y_test:  {splits['y_test'].shape}")
