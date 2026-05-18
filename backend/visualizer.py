#!/usr/bin/env python3
"""
visualizer.py
CNN-LSTM 결과 시각화 모듈 (새로 작성)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """CNN-LSTM 결과 시각화 클래스"""

    def __init__(self, style='seaborn-v0_8', figsize=(12, 8)):
        """
        시각화기 초기화

        Args:
            style (str): matplotlib 스타일
            figsize (tuple): 기본 그래프 크기
        """
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('default')

        self.figsize = figsize
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#28A745',
            'info': '#17A2B8',
            'warning': '#FFC107',
            'danger': '#DC3545'
        }

    def plot_performance_dashboard(self, metrics):
        """
        성능 지표 대시보드

        Args:
            metrics (dict): 성능 지표
        """
        print("📊 성능 대시보드 생성 중...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold')

        # 주요 지표 바 차트
        main_metrics = []
        main_values = []
        colors = []

        for metric in ['RMSE', 'MAE', 'R²', 'MAPE']:
            if metric in metrics:
                main_metrics.append(metric)
                main_values.append(metrics[metric])
                if metric == 'R²':
                    colors.append(self.colors['success'])
                else:
                    colors.append(self.colors['primary'])

        if main_metrics:
            bars = axes[0, 0].bar(main_metrics, main_values, color=colors, alpha=0.8)
            axes[0, 0].set_title('Key Performance Indicators')
            axes[0, 0].set_ylabel('Value')

            # 값 표시
            for bar, value in zip(bars, main_values):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # R² 스코어 게이지
        r2_score = metrics.get('R²', 0)

        # 간단한 게이지 차트
        angles = np.linspace(0, np.pi, 100)
        x = np.cos(angles)
        y = np.sin(angles)

        axes[0, 1].plot(x, y, 'k-', linewidth=3)
        axes[0, 1].fill_between(x, 0, y, alpha=0.3)

        # R² 점수 표시
        if r2_score >= 0.9:
            color = self.colors['success']
        elif r2_score >= 0.7:
            color = self.colors['warning']
        else:
            color = self.colors['danger']

        axes[0, 1].text(0, 0.5, f'R² = {r2_score:.3f}',
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        axes[0, 1].set_title('R² Score')
        axes[0, 1].set_xlim(-1.2, 1.2)
        axes[0, 1].set_ylim(0, 1.2)
        axes[0, 1].axis('off')

        # MAPE 도넛 차트
        mape = min(metrics.get('MAPE', 0), 100)
        accuracy = max(0, 100 - mape)

        sizes = [accuracy, mape]
        labels = ['Accuracy', 'MAPE']
        colors_pie = [self.colors['success'], self.colors['danger']]

        wedges, texts, autotexts = axes[1, 0].pie(sizes, labels=labels, colors=colors_pie,
                                                  autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('MAPE-based on accuracy')

        # 성능 등급
        if r2_score >= 0.9:
            grade, grade_color = 'A', self.colors['success']
        elif r2_score >= 0.8:
            grade, grade_color = 'B', self.colors['info']
        elif r2_score >= 0.7:
            grade, grade_color = 'C', self.colors['warning']
        else:
            grade, grade_color = 'D', self.colors['danger']

        axes[1, 1].text(0.5, 0.5, grade, ha='center', va='center',
                        fontsize=80, fontweight='bold', color=grade_color,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Model Performance Rating')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()
        print("✅ Performance Dashboard Complete")

    def plot_predictions(self, y_true, y_pred, sample_size=100):
        """
        예측 결과 시각화

        Args:
            y_true (np.array): 실제값
            y_pred (np.array): 예측값
            sample_size (int): 표시할 샘플 수
        """
        print("🎯 예측 결과 시각화 중...")

        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analysis of prediction results', fontsize=16, fontweight='bold')

        # 1. 시계열 비교 (최근 데이터)
        if len(y_true_flat) > sample_size:
            indices = range(len(y_true_flat) - sample_size, len(y_true_flat))
            true_sample = y_true_flat[-sample_size:]
            pred_sample = y_pred_flat[-sample_size:]
        else:
            indices = range(len(y_true_flat))
            true_sample = y_true_flat
            pred_sample = y_pred_flat

        axes[0, 0].plot(indices, true_sample, 'o-', label='Actual value',
                        color=self.colors['primary'], linewidth=2, markersize=4)
        axes[0, 0].plot(indices, pred_sample, 's-', label='Predicted value',
                        color=self.colors['secondary'], linewidth=2, markersize=4)
        axes[0, 0].set_title('Time series forecast comparison')
        axes[0, 0].set_xlabel('Time index')
        axes[0, 0].set_ylabel('Electricity demand (kW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 산점도
        axes[0, 1].scatter(y_true_flat, y_pred_flat, alpha=0.6, color=self.colors['accent'], s=20)

        # 완벽한 예측 라인
        min_val = min(y_true_flat.min(), y_pred_flat.min())
        max_val = max(y_true_flat.max(), y_pred_flat.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

        # R² 표시
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true_flat, y_pred_flat)
        axes[0, 1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 1].transAxes,
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        axes[0, 1].set_title('Actual vs. predicted values')
        axes[0, 1].set_xlabel('Actual value (kW)')
        axes[0, 1].set_ylabel('Predicted value (kW)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 잔차 분석
        residuals = y_true_flat - y_pred_flat
        axes[1, 0].scatter(y_pred_flat, residuals, alpha=0.6, color=self.colors['info'], s=20)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Residual analysis')
        axes[1, 0].set_xlabel('Predicted value (kW)')
        axes[1, 0].set_ylabel('Residual (kW)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 잔차 히스토그램
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color=self.colors['success'], edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='0')
        axes[1, 1].axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2,
                           label=f'Avg.: {np.mean(residuals):.2f}')
        axes[1, 1].set_title('Residual distribution')
        axes[1, 1].set_xlabel('Residual (kW)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 잔차 통계
        print("📊 잔차 통계:")
        print(f"  - 평균: {np.mean(residuals):.4f}")
        print(f"  - 표준편차: {np.std(residuals):.4f}")
        print(f"  - 최소값: {np.min(residuals):.4f}")
        print(f"  - 최대값: {np.max(residuals):.4f}")
        print("✅ 예측 결과 시각화 완료")

    def plot_training_history(self, history):
        """
        학습 과정 시각화

        Args:
            history: 학습 이력
        """
        print("📈 학습 과정 시각화 중...")

        if history is None:
            print("❌ 학습 이력이 없습니다.")
            return

        history_dict = history.history if hasattr(history, 'history') else history
        epochs = range(1, len(history_dict['loss']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model learning process', fontsize=16, fontweight='bold')

        # Loss
        axes[0, 0].plot(epochs, history_dict['loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history_dict:
            axes[0, 0].plot(epochs, history_dict['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss')
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
        else:
            axes[0, 1].text(0.5, 0.5, 'No MAPE data', ha='center', va='center',
                            transform=axes[0, 1].transAxes, fontsize=14, color='gray')
            axes[0, 1].axis('off')

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
        else:
            axes[1, 0].text(0.5, 0.5, 'No MAPE data', ha='center', va='center',
                            transform=axes[1, 0].transAxes, fontsize=14, color='gray')
            axes[1, 0].axis('off')

        # Learning Rate (if available)
        if 'lr' in history_dict:
            axes[1, 1].plot(epochs, history_dict['lr'], 'red', linewidth=2)
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # 학습 요약 정보 표시
            total_epochs = len(epochs)
            final_loss = history_dict['loss'][-1] if 'loss' in history_dict else 0
            final_val_loss = history_dict['val_loss'][-1] if 'val_loss' in history_dict else 0

            summary_text = f"Learning Summary\n\n"
            summary_text += f"Total epochs: {total_epochs}\n"
            summary_text += f"Final Loss: {final_loss:.4f}\n"
            if 'val_loss' in history_dict:
                summary_text += f"Final Val Loss: {final_val_loss:.4f}"

            axes[1, 1].text(0.5, 0.5, summary_text, ha='center', va='center',
                            transform=axes[1, 1].transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            axes[1, 1].set_title('Learning Summary')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()
        print("✅ 학습 과정 시각화 완료")

    def plot_data_distribution(self, raw_data):
        """
        데이터 분포 분석

        Args:
            raw_data (pd.DataFrame): 원본 데이터
        """
        print("📈 데이터 분포 분석 중...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data distribution analysis', fontsize=16, fontweight='bold')

        # 1. AC Power 분포
        if 'AC_POWER' in raw_data.columns:
            axes[0, 0].hist(raw_data['AC_POWER'], bins=50, alpha=0.7,
                            color=self.colors['primary'], edgecolor='black')
            axes[0, 0].set_xlabel('AC Power (W)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('AC power generation distribution')
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Irradiation 분포
        if 'IRRADIATION' in raw_data.columns:
            axes[0, 1].hist(raw_data['IRRADIATION'], bins=50, alpha=0.7,
                            color=self.colors['secondary'], edgecolor='black')
            axes[0, 1].set_xlabel('Irradiation (W/m²)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Sunlight distribution')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 온도 분포
        if 'AMBIENT_TEMPERATURE' in raw_data.columns:
            axes[0, 2].hist(raw_data['AMBIENT_TEMPERATURE'], bins=50, alpha=0.7,
                            color=self.colors['accent'], edgecolor='black')
            axes[0, 2].set_xlabel('Temperature (°C)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Outside temperature distribution')
            axes[0, 2].grid(True, alpha=0.3)

        # 4. 상관관계 히트맵
        numeric_cols = []
        for col in ['AC_POWER', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'POWER_DEMAND']:
            if col in raw_data.columns:
                numeric_cols.append(col)

        if len(numeric_cols) >= 2:
            corr_matrix = raw_data[numeric_cols].corr()

            im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_xticks(range(len(corr_matrix.columns)))
            axes[1, 0].set_yticks(range(len(corr_matrix.columns)))
            axes[1, 0].set_xticklabels(corr_matrix.columns, rotation=45)
            axes[1, 0].set_yticklabels(corr_matrix.columns)
            axes[1, 0].set_title('Correlation between variables')

            # 상관계수 값 표시
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    color = "white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black"
                    axes[1, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                    ha="center", va="center", color=color, fontweight='bold')

            fig.colorbar(im, ax=axes[1, 0])

        # 5. 시간별 패턴
        if 'DATE_TIME' in raw_data.columns and 'AC_POWER' in raw_data.columns:
            try:
                raw_data_copy = raw_data.copy()
                raw_data_copy['hour'] = pd.to_datetime(raw_data_copy['DATE_TIME']).dt.hour
                hourly_avg = raw_data_copy.groupby('hour')['AC_POWER'].mean()

                axes[1, 1].plot(hourly_avg.index, hourly_avg.values, 'o-',
                                color=self.colors['info'], linewidth=2, markersize=6)
                axes[1, 1].fill_between(hourly_avg.index, hourly_avg.values,
                                        alpha=0.3, color=self.colors['info'])
                axes[1, 1].set_xlabel('Time')
                axes[1, 1].set_ylabel('Avg. AC Power (W)')
                axes[1, 1].set_title('Average power generation by time zone')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_xticks(range(0, 24, 4))
            except Exception:
                axes[1, 1].text(0.5, 0.5, 'No hourly pattern\nanalysis possible', ha='center', va='center',
                                transform=axes[1, 1].transAxes, fontsize=14, color='gray')
                axes[1, 1].axis('off')

        # 6. Box plot
        if len(numeric_cols) > 0:
            box_data = []
            box_labels = []
            for col in numeric_cols[:4]:  # 상위 4개만
                if col in raw_data.columns:
                    data = raw_data[col].dropna()
                    if len(data) > 0:
                        box_data.append(data)
                        box_labels.append(col)

            if box_data:
                bp = axes[1, 2].boxplot(box_data, labels=box_labels, patch_artist=True)

                colors = [self.colors['primary'], self.colors['secondary'],
                          self.colors['accent'], self.colors['info']]
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                axes[1, 2].set_title('Key Variable Box Plot')
                axes[1, 2].tick_params(axis='x', rotation=45)
                axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        print("✅ 데이터 분포 분석 완료")

    def plot_comprehensive_results(self, results, processed_data, save_plots=False):
        """
        종합 결과 시각화

        Args:
            results (dict): 모델 결과
            processed_data (dict): 처리된 데이터
            save_plots (bool): 플롯 저장 여부
        """
        print("🎨 종합 결과 시각화 시작...")

        # 1. 성능 지표 대시보드
        if 'metrics' in results:
            self.plot_performance_dashboard(results['metrics'])

        # 2. 예측 결과 분석
        if 'predictions' in results:
            y_true = results['predictions']['y_true']
            y_pred = results['predictions']['y_pred']
            self.plot_predictions(y_true, y_pred)

        # 3. 데이터 분포 분석
        if processed_data and 'raw_data' in processed_data:
            self.plot_data_distribution(processed_data['raw_data'])

        # 4. 인터랙티브 대시보드 (Plotly가 있는 경우)
        try:
            self.create_interactive_dashboard(results)
        except ImportError:
            print("⚠️ Plotly가 설치되지 않아 인터랙티브 대시보드를 건너뜁니다.")
        except Exception as e:
            print(f"⚠️ 인터랙티브 대시보드 생성 실패: {e}")

        if save_plots:
            self.save_all_plots()

        print("✅ 종합 시각화 완료")

    def create_interactive_dashboard(self, results):
        """
        인터랙티브 대시보드 생성 (Plotly)

        Args:
            results (dict): 모델 결과
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            print("🌐 인터랙티브 대시보드 생성 중...")

            if 'predictions' not in results:
                print("❌ 예측 데이터가 없습니다.")
                return

            y_true = results['predictions']['y_true'].flatten()
            y_pred = results['predictions']['y_pred'].flatten()

            # 서브플롯 생성
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Time series comparison', 'Actual vs. predicted values', 'Residual distribution', 'Performance indicators'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )

            # 1. 시계열 비교 (최근 100개)
            recent_size = min(100, len(y_true))
            indices = list(range(len(y_true) - recent_size, len(y_true)))

            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=y_true[-recent_size:],
                    mode='lines+markers',
                    name='Actual value',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=y_pred[-recent_size:],
                    mode='lines+markers',
                    name='Predicted value',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )

            # 2. 산점도
            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode='markers',
                    name='Prediction results',
                    marker=dict(size=5, opacity=0.6, color='green')
                ),
                row=1, col=2
            )

            # 완벽한 예측 라인
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect prediction',
                    line=dict(color='red', dash='dash', width=2)
                ),
                row=1, col=2
            )

            # 3. 잔차 히스토그램
            residuals = y_true - y_pred
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=30,
                    name='Residual distribution',
                    marker=dict(color='purple', opacity=0.7)
                ),
                row=2, col=1
            )

            # 4. 성능 지표 테이블
            if 'metrics' in results:
                metrics = results['metrics']
                metric_names = list(metrics.keys())
                metric_values = [f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                                 for v in metrics.values()]

                fig.add_trace(
                    go.Table(
                        header=dict(values=['indicators', 'value'],
                                    fill_color='lightblue',
                                    align='center'),
                        cells=dict(values=[metric_names, metric_values],
                                   fill_color='lightgray',
                                   align='center')
                    ),
                    row=2, col=2
                )

            # 레이아웃 업데이트
            fig.update_layout(
                title_text="CNN-LSTM Power Demand Forecasting Interactive Dashboard",
                title_x=0.5,
                height=800
            )

            # HTML 파일로 저장
            fig.write_html("interactive_dashboard.html")
            print("✅ 인터랙티브 대시보드 저장: interactive_dashboard.html")

            # Jupyter에서 실행 시 표시
            try:
                fig.show()
            except Exception:
                print("💡 대시보드가 HTML 파일로 저장되었습니다.")

        except ImportError:
            print("⚠️ Plotly가 설치되지 않았습니다.")
        except Exception as e:
            print(f"❌ 인터랙티브 대시보드 생성 실패: {e}")

    def save_all_plots(self, output_dir='plots'):
        """
        모든 플롯을 파일로 저장

        Args:
            output_dir (str): 저장 디렉토리
        """
        import os

        print(f"💾 플롯 저장 중... ({output_dir})")

        os.makedirs(output_dir, exist_ok=True)

        # 현재 열려있는 모든 figure 저장
        figs = [plt.figure(n) for n in plt.get_fignums()]

        for i, fig in enumerate(figs):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plot_{i + 1:02d}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)

            try:
                fig.savefig(filepath, dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                print(f"✅ 저장: {filepath}")
            except Exception as e:
                print(f"❌ 저장 실패 {filepath}: {e}")

        print(f"📁 총 {len(figs)}개 플롯이 '{output_dir}' 디렉토리에 저장되었습니다.")

    def plot_model_comparison(self, comparison_results):
        """
        모델 비교 결과 시각화

        Args:
            comparison_results (dict): 여러 모델의 결과 비교
        """
        print("🏆 모델 비교 시각화 중...")

        if not comparison_results:
            print("❌ 비교할 모델 결과가 없습니다.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model performance comparison', fontsize=16, fontweight='bold')

        models = list(comparison_results.keys())
        metrics = ['RMSE', 'MAE', 'R²', 'MAPE']

        # 각 지표별 비교
        for idx, metric in enumerate(metrics):
            if idx >= 4:  # 최대 4개 지표만
                break

            ax = axes[idx // 2, idx % 2]

            values = []
            for model in models:
                if 'metrics' in comparison_results[model] and metric in comparison_results[model]['metrics']:
                    values.append(comparison_results[model]['metrics'][metric])
                else:
                    values.append(0)

            if not values:
                ax.text(0.5, 0.5, f'{metric}\nNo data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=14, color='gray')
                ax.axis('off')
                continue

            colors = [self.colors['primary'], self.colors['secondary'],
                      self.colors['accent'], self.colors['info']][:len(models)]

            bars = ax.bar(models, values, color=colors, alpha=0.8)

            # 값 표시
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

            ax.set_title(f'{metric} comparison')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

            # 최적값 강조
            if values:
                if metric == 'R²':
                    best_idx = np.argmax(values)
                else:
                    best_idx = np.argmin([v for v in values if v > 0])

                if 0 <= best_idx < len(bars):
                    bars[best_idx].set_color(self.colors['success'])

        plt.tight_layout()
        plt.show()
        print("✅ 모델 비교 완료")

    def plot_residual_analysis(self, y_true, y_pred):
        """
        상세 잔차 분석

        Args:
            y_true (np.array): 실제값
            y_pred (np.array): 예측값
        """
        print("🔍 잔차 분석 중...")

        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        residuals = y_true_flat - y_pred_flat

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed residual analysis', fontsize=16, fontweight='bold')

        # 1. 잔차 vs 예측값
        axes[0, 0].scatter(y_pred_flat, residuals, alpha=0.6,
                           color=self.colors['primary'], s=20)
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)

        # 추세선 추가
        try:
            z = np.polyfit(y_pred_flat, residuals, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(sorted(y_pred_flat), p(sorted(y_pred_flat)),
                            'r-', alpha=0.8, linewidth=2,
                            label=f'Trend line (slope: {z[0]:.4f})')
            axes[0, 0].legend()
        except Exception:
            pass

        axes[0, 0].set_xlabel('Predicted value (kW)')
        axes[0, 0].set_ylabel('Residual (kW)')
        axes[0, 0].set_title('Residual vs Predicted value')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Q-Q plot (정규성 검정)
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (Normality test)')
            axes[0, 1].grid(True, alpha=0.3)

            # 정규성 검정
            if len(residuals) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                axes[0, 1].text(0.05, 0.95, f'Shapiro-Wilk\nStat: {shapiro_stat:.4f}\np-value: {shapiro_p:.4f}',
                                transform=axes[0, 1].transAxes, fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except ImportError:
            axes[0, 1].text(0.5, 0.5, 'Q-Q Plot\n(scipy necessary)', ha='center', va='center',
                            transform=axes[0, 1].transAxes, fontsize=14, color='gray')
            axes[0, 1].axis('off')

        # 3. 잔차 히스토그램 + 정규분포 곡선
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, density=True,
                        color=self.colors['accent'], edgecolor='black')

        # 정규분포 곡선 오버레이
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        try:
            from scipy import stats
            y = stats.norm.pdf(x, mu, sigma)
            axes[1, 0].plot(x, y, 'r-', linewidth=2, label='Normal distribution')
            axes[1, 0].legend()
        except ImportError:
            pass

        axes[1, 0].set_xlabel('residual')
        axes[1, 0].set_ylabel('density')
        axes[1, 0].set_title('Residual distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 잔차 통계 요약
        residual_stats = {
            'Average': np.mean(residuals),
            'Standard deviation': np.std(residuals),
            'Minimum value': np.min(residuals),
            'Maximum value': np.max(residuals),
            'Median': np.median(residuals),
            '25% quarter': np.percentile(residuals, 25),
            '75% quarter': np.percentile(residuals, 75)
        }

        # 통계 표 생성
        stats_text = []
        for stat, value in residual_stats.items():
            stats_text.append([stat, f'{value:.4f}'])

        # 표 형태로 표시
        table_data = []
        for i in range(0, len(stats_text), 2):
            row = [stats_text[i][0], stats_text[i][1]]
            if i + 1 < len(stats_text):
                row.extend([stats_text[i + 1][0], stats_text[i + 1][1]])
            else:
                row.extend(['', ''])
            table_data.append(row)

        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['statistic', 'value', 'statistic', 'value'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        axes[1, 1].axis('off')
        axes[1, 1].set_title('Residual Statistics Summary')

        plt.tight_layout()
        plt.show()
        print("✅ 잔차 분석 완료")


# 사용 예제
if __name__ == "__main__":
    print("🎨 시각화 모듈 테스트 시작...")

    # 예제 데이터 생성
    np.random.seed(42)
    n_samples = 500

    y_true = np.random.randn(n_samples, 1) * 100 + 1000
    y_pred = y_true + np.random.randn(n_samples, 1) * 50

    # 가상의 결과 데이터
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        results = {
            'metrics': {
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'R²': r2_score(y_true, y_pred),
                'MAPE': np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
            },
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred
            }
        }
    except ImportError:
        print("⚠️ scikit-learn이 설치되지 않아 간단한 지표만 계산합니다.")
        results = {
            'metrics': {
                'RMSE': np.sqrt(np.mean((y_true - y_pred) ** 2)),
                'MAE': np.mean(np.abs(y_true - y_pred)),
                'R²': 0.85,  # 예시값
                'MAPE': 15.0  # 예시값
            },
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred
            }
        }

    # 가상의 원본 데이터
    dates = pd.date_range('2020-05-15', periods=n_samples, freq='H')
    raw_data = pd.DataFrame({
        'DATE_TIME': dates,
        'AC_POWER': np.random.exponential(3000, n_samples),
        'IRRADIATION': np.random.gamma(2, 0.5, n_samples),
        'AMBIENT_TEMPERATURE': 25 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 3,
                                                                                                          n_samples),
        'MODULE_TEMPERATURE': 28 + 12 * np.sin(np.arange(n_samples) * 2 * np.pi / 24) + np.random.normal(0, 2,
                                                                                                         n_samples),
        'POWER_DEMAND': y_true.flatten()
    })

    processed_data = {
        'raw_data': raw_data
    }

    # 가상의 학습 이력
    history = {
        'loss': np.random.exponential(1, 50)[::-1] * 0.1 + 0.01,
        'val_loss': np.random.exponential(1, 50)[::-1] * 0.12 + 0.015,
        'mae': np.random.exponential(1, 50)[::-1] * 0.05 + 0.005,
        'val_mae': np.random.exponential(1, 50)[::-1] * 0.06 + 0.007
    }

    # 시각화기 테스트
    visualizer = ResultVisualizer()

    print("\n1️⃣ 성능 대시보드 테스트...")
    visualizer.plot_performance_dashboard(results['metrics'])

    print("\n2️⃣ 예측 결과 테스트...")
    visualizer.plot_predictions(y_true, y_pred)

    print("\n3️⃣ 학습 과정 테스트...")
    visualizer.plot_training_history(history)

    print("\n4️⃣ 데이터 분포 테스트...")
    visualizer.plot_data_distribution(raw_data)

    print("\n5️⃣ 잔차 분석 테스트...")
    visualizer.plot_residual_analysis(y_true, y_pred)

    print("\n6️⃣ 종합 시각화 테스트...")
    visualizer.plot_comprehensive_results(results, processed_data, save_plots=False)

    print("\n🎉 모든 시각화 테스트 완료!")

    # 모델 비교 테스트
    print("\n7️⃣ 모델 비교 테스트...")
    comparison_results = {
        'Basic CNN-LSTM': results,
        'Advanced CNN-LSTM': {
            'metrics': {
                'RMSE': results['metrics']['RMSE'] * 0.9,
                'MAE': results['metrics']['MAE'] * 0.9,
                'R²': min(1.0, results['metrics']['R²'] * 1.1),
                'MAPE': results['metrics']['MAPE'] * 0.8
            }
        },
        'Transformer': {
            'metrics': {
                'RMSE': results['metrics']['RMSE'] * 0.85,
                'MAE': results['metrics']['MAE'] * 0.85,
                'R²': min(1.0, results['metrics']['R²'] * 1.15),
                'MAPE': results['metrics']['MAPE'] * 0.75
            }
        }
    }

    visualizer.plot_model_comparison(comparison_results)

    print("\n✨ visualizer.py 모듈 테스트 완료!")