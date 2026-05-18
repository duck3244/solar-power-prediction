#!/usr/bin/env python3
"""
data_loader.py
태양광 발전 데이터 로드 및 전처리 모듈
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SolarDataLoader:
    """태양광 발전 데이터 로더 클래스"""
    
    def __init__(self):
        self.generation_df = None
        self.weather_df = None
        self.merged_df = None
        
    def load_data(self, generation_file, weather_file):
        """
        CSV 파일에서 태양광 발전 및 기상 데이터 로드
        
        Args:
            generation_file (str): 발전량 데이터 CSV 파일 경로
            weather_file (str): 기상 데이터 CSV 파일 경로
            
        Returns:
            tuple: (generation_df, weather_df) 데이터프레임 튜플
        """
        print("🔄 데이터 로드 시작...")
        
        try:
            # 발전량 데이터 로드
            self.generation_df = pd.read_csv(generation_file)
            print(f"✅ 발전량 데이터: {len(self.generation_df)} 행, {len(self.generation_df.columns)} 열")
            
            # 기상 데이터 로드
            self.weather_df = pd.read_csv(weather_file)
            print(f"✅ 기상 데이터: {len(self.weather_df)} 행, {len(self.weather_df.columns)} 열")
            
            return self.generation_df, self.weather_df
            
        except FileNotFoundError as e:
            print(f"❌ 파일을 찾을 수 없습니다: {e}")
            print("💡 합성 데이터를 생성합니다...")
            return self.generate_synthetic_data()
            
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생: {e}")
            raise
    
    def parse_datetime(self, df, datetime_col='DATE_TIME', format_type='generation'):
        """
        날짜/시간 컬럼 파싱
        
        Args:
            df (pd.DataFrame): 대상 데이터프레임
            datetime_col (str): 날짜/시간 컬럼명
            format_type (str): 'generation' 또는 'weather' 포맷 타입
            
        Returns:
            pd.DataFrame: 날짜/시간이 파싱된 데이터프레임
        """
        df = df.copy()
        
        if format_type == 'generation':
            # "15-05-2020 00:00" 형식 처리
            df[datetime_col] = pd.to_datetime(df[datetime_col], 
                                            format='%d-%m-%Y %H:%M', errors='coerce')
        else:
            # 표준 datetime 형식 처리
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        
        # 잘못된 날짜 제거
        invalid_dates = df[datetime_col].isna().sum()
        if invalid_dates > 0:
            print(f"⚠️ 잘못된 날짜 {invalid_dates}개 제거")
            df = df.dropna(subset=[datetime_col])
            
        return df
    
    def aggregate_hourly_generation(self, df):
        """
        발전량 데이터를 시간별로 집계 (다중 인버터 합계)
        
        Args:
            df (pd.DataFrame): 발전량 데이터프레임
            
        Returns:
            pd.DataFrame: 시간별 집계된 발전량 데이터
        """
        print("🔧 발전량 데이터 시간별 집계 중...")
        
        hourly_gen = df.groupby('DATE_TIME').agg({
            'DC_POWER': 'sum',
            'AC_POWER': 'sum',
            'DAILY_YIELD': 'sum',
            'SOURCE_KEY': 'count'  # 인버터 개수
        }).reset_index()
        
        # 컬럼명 변경
        hourly_gen.rename(columns={'SOURCE_KEY': 'INVERTER_COUNT'}, inplace=True)
        
        print(f"✅ {len(hourly_gen)}개 시간 포인트로 집계 완료")
        return hourly_gen
    
    def aggregate_hourly_weather(self, df):
        """
        기상 데이터를 시간별로 집계 (평균값)
        
        Args:
            df (pd.DataFrame): 기상 데이터프레임
            
        Returns:
            pd.DataFrame: 시간별 집계된 기상 데이터
        """
        print("🔧 기상 데이터 시간별 집계 중...")
        
        hourly_weather = df.groupby('DATE_TIME').agg({
            'AMBIENT_TEMPERATURE': 'mean',
            'MODULE_TEMPERATURE': 'mean',
            'IRRADIATION': 'mean'
        }).reset_index()
        
        print(f"✅ {len(hourly_weather)}개 시간 포인트로 집계 완료")
        return hourly_weather
    
    def merge_data(self, gen_df, weather_df):
        """
        발전량과 기상 데이터 병합
        
        Args:
            gen_df (pd.DataFrame): 시간별 발전량 데이터
            weather_df (pd.DataFrame): 시간별 기상 데이터
            
        Returns:
            pd.DataFrame: 병합된 데이터프레임
        """
        print("🔧 데이터 병합 중...")
        
        merged_df = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner')
        
        print(f"✅ 병합 완료: {len(merged_df)} 행")
        return merged_df
    
    def filter_daytime_data(self, df, min_power=100, min_irradiation=0):
        """
        낮시간 데이터 필터링 (태양광 발전이 있는 시간)
        
        Args:
            df (pd.DataFrame): 병합된 데이터프레임
            min_power (float): 최소 발전량 (W)
            min_irradiation (float): 최소 일조량
            
        Returns:
            pd.DataFrame: 필터링된 낮시간 데이터
        """
        print("🔧 낮시간 데이터 필터링 중...")
        
        # 시간 특성 추가
        df['hour'] = df['DATE_TIME'].dt.hour
        
        # 필터링 조건
        daytime_filter = (
            (df['AC_POWER'] > min_power) &
            (df['IRRADIATION'] > min_irradiation) &
            (df['hour'] >= 6) &
            (df['hour'] <= 18)
        )
        
        filtered_df = df[daytime_filter].copy()
        
        print(f"✅ 필터링 완료: {len(filtered_df)} 행의 낮시간 데이터")
        return filtered_df
    
    def generate_synthetic_data(self):
        """
        합성 태양광 발전 데이터 생성 (파일이 없을 경우 사용)
        
        Returns:
            tuple: (generation_df, weather_df) 합성 데이터
        """
        print("📊 합성 데이터 생성 중...")
        
        # 날짜 범위 설정 (35일간)
        start_date = datetime(2020, 5, 15)
        end_date = start_date + timedelta(days=35)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        generation_data = []
        weather_data = []
        
        # 가상 인버터 키 생성
        inverter_keys = [f'INV_{i:03d}' for i in range(1, 23)]  # 22개 인버터
        
        for dt in date_range:
            hour = dt.hour
            
            # 낮시간만 생성 (6-18시)
            if 6 <= hour <= 18:
                # 태양광 발전 패턴 (종 모양 곡선)
                solar_factor = np.exp(-((hour - 12) ** 2) / 8)
                base_irradiation = solar_factor * 1.2 * (0.8 + 0.4 * np.random.random())
                base_temp = 25 + 10 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 2)
                
                # 각 인버터별 발전량 생성
                for inv_key in inverter_keys:
                    individual_factor = 0.8 + 0.4 * np.random.random()
                    dc_power = solar_factor * individual_factor * 3000  # 인버터당 최대 3kW
                    ac_power = dc_power * 0.95  # 변환 효율 95%
                    daily_yield = ac_power * 0.1  # 임시값
                    
                    generation_data.append({
                        'DATE_TIME': dt.strftime('%d-%m-%Y %H:%M'),
                        'PLANT_ID': 4135001,
                        'SOURCE_KEY': inv_key,
                        'DC_POWER': max(0, dc_power),
                        'AC_POWER': max(0, ac_power),
                        'DAILY_YIELD': daily_yield,
                        'TOTAL_YIELD': np.random.randint(6000000, 8000000)
                    })
                
                # 기상 데이터 생성 (시간당 1개)
                weather_data.append({
                    'DATE_TIME': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'PLANT_ID': 4135001,
                    'SOURCE_KEY': 'WEATHER_01',
                    'AMBIENT_TEMPERATURE': base_temp,
                    'MODULE_TEMPERATURE': base_temp + 5 + np.random.normal(0, 1),
                    'IRRADIATION': max(0, base_irradiation)
                })
        
        # 데이터프레임 생성
        self.generation_df = pd.DataFrame(generation_data)
        self.weather_df = pd.DataFrame(weather_data)
        
        print(f"✅ 합성 발전량 데이터: {len(self.generation_df)} 행")
        print(f"✅ 합성 기상 데이터: {len(self.weather_df)} 행")
        
        return self.generation_df, self.weather_df
    
    def get_data_summary(self, df, data_type="merged"):
        """
        데이터 요약 정보 출력
        
        Args:
            df (pd.DataFrame): 분석할 데이터프레임
            data_type (str): 데이터 타입 ("generation", "weather", "merged")
        """
        print(f"\n{'='*50}")
        print(f"📊 {data_type.upper()} 데이터 요약")
        print(f"{'='*50}")
        
        print(f"행 수: {len(df):,}")
        print(f"열 수: {len(df.columns)}")
        
        if 'DATE_TIME' in df.columns:
            print(f"날짜 범위: {df['DATE_TIME'].min()} ~ {df['DATE_TIME'].max()}")
        
        if data_type == "generation" and 'AC_POWER' in df.columns:
            print(f"AC Power 통계:")
            print(f"  - 평균: {df['AC_POWER'].mean():.2f} W")
            print(f"  - 최대: {df['AC_POWER'].max():.2f} W")
            print(f"  - 0보다 큰 값: {(df['AC_POWER'] > 0).sum():,} 개")
            
        if data_type == "weather" and 'IRRADIATION' in df.columns:
            print(f"Irradiation 통계:")
            print(f"  - 평균: {df['IRRADIATION'].mean():.2f}")
            print(f"  - 최대: {df['IRRADIATION'].max():.2f}")
            
        print(f"결측값:")
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                print(f"  - {col}: {count}")
    
    def preprocess_pipeline(self, generation_file, weather_file):
        """
        전체 전처리 파이프라인 실행
        
        Args:
            generation_file (str): 발전량 데이터 파일
            weather_file (str): 기상 데이터 파일
            
        Returns:
            pd.DataFrame: 전처리 완료된 데이터프레임
        """
        print("🚀 전체 전처리 파이프라인 시작")
        print("="*60)
        
        # 1. 데이터 로드
        gen_df, weather_df = self.load_data(generation_file, weather_file)
        
        # 2. 날짜/시간 파싱
        gen_df = self.parse_datetime(gen_df, format_type='generation')
        weather_df = self.parse_datetime(weather_df, format_type='weather')
        
        # 3. 데이터 요약 출력
        self.get_data_summary(gen_df, "generation")
        self.get_data_summary(weather_df, "weather")
        
        # 4. 시간별 집계
        hourly_gen = self.aggregate_hourly_generation(gen_df)
        hourly_weather = self.aggregate_hourly_weather(weather_df)
        
        # 5. 데이터 병합
        merged_df = self.merge_data(hourly_gen, hourly_weather)
        
        # 6. 낮시간 데이터 필터링
        daytime_df = self.filter_daytime_data(merged_df)
        
        # 7. 최종 요약
        self.get_data_summary(daytime_df, "merged")
        
        print("✅ 전처리 파이프라인 완료")
        self.merged_df = daytime_df
        
        return daytime_df

# 사용 예제
if __name__ == "__main__":
    # 데이터 로더 초기화
    loader = SolarDataLoader()
    
    # 전처리 파이프라인 실행
    try:
        processed_data = loader.preprocess_pipeline(
            'Plant_1_Generation_Data.csv',
            'Plant_1_Weather_Sensor_Data.csv'
        )
        print(f"🎉 전처리 완료: {len(processed_data)} 행의 데이터 준비됨")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
