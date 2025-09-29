# src/config.py

from pathlib import Path

# 경로 설정 
ROOT_DATA_DIR = Path(r"C:\Users\shoot\OneDrive\Desktop\mlft_data") 
# 외부 피처 파일 경로
EXTERNAL_FEATURE_FILE = Path(r"C:\Users\shoot\Downloads\movstats_bars_noHIGHT_barvalue.csv")

# --- 출력 경로 설정 ---
OUTPUT_DIR = Path("./output")
PROCESSED_DATA_PATH = OUTPUT_DIR / "final_featured_df.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# 분석할 채널 및 종속변수 소스 컬럼 정의
CHANNELS = ['CH-A1', 'CH-A2', 'CH-A3', 'CH-A4', 'CH-A5',
            'CH-B1', 'CH-B2', 'CH-B3', 'CH-B4', 'CH-B5']
TARGET_SOURCE_COLS = [f'{ch}_movmin30_mean' for ch in CHANNELS]

# 최적 하이퍼파라미터
BEST_LGBM_PARAMS = {
    'n_estimators': 800,
    'learning_rate': 0.024850134583659714,
    'num_leaves': 92,
    'max_depth': 3,
    'subsample': 0.8008128713976412,
    'colsample_bytree': 0.7420726031676225,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

# 피처 선택 단계에서 선택된 피처 목록 (100개)
SELECTED_FEATURES = [
    'CH-A1_median', 'CH-A2_median', 'CH-A2_min', 'CH-A3_median', 'CH-A3_min',
    'CH-A4_median', 'CH-A5_median', 'CH-B1_median', 'CH-B2_median', 'CH-B3_median',
    'micro_CH-A1_rolling_min_mean_11', 'micro_CH-A1_rolling_min_std_11', 'micro_CH-A1_rolling_min_mean_33',
    'micro_CH-A2_rolling_min_mean_11', 'micro_CH-A2_rolling_min_mean_33', 'micro_CH-A3_rolling_min_mean_11',
    'micro_CH-A3_rolling_min_std_11', 'micro_CH-A3_rolling_min_mean_33', 'micro_CH-A4_rolling_min_mean_11',
    'micro_CH-A4_rolling_min_std_11', 'micro_CH-A4_rolling_min_mean_33', 'micro_CH-A5_rolling_min_mean_11',
    'micro_CH-A5_rolling_min_std_11', 'micro_CH-A5_rolling_min_mean_33', 'micro_CH-B1_rolling_min_mean_11',
    'micro_CH-B1_rolling_min_mean_33', 'micro_CH-B2_rolling_min_mean_11', 'micro_CH-B2_rolling_min_mean_33',
    'micro_CH-B3_rolling_min_mean_11', 'micro_CH-B3_rolling_min_std_11', 'micro_CH-B3_rolling_min_mean_33',
    'micro_CH-B4_rolling_min_mean_11', 'micro_CH-B4_rolling_min_mean_33', 'micro_CH-B5_rolling_min_mean_11',
    'micro_CH-B5_rolling_min_mean_33', 'CH-A1_perm_entropy', 'CH-A2_perm_entropy', 'CH-A4_perm_entropy',
    'CH-A5_perm_entropy', 'CH-B1_perm_entropy', 'CH-B2_perm_entropy', 'CH-B3_perm_entropy', 'CH-B4_perm_entropy',
    'CH-B5_perm_entropy', 'bar_in_lot_sequence', 'STEEL_TYPE', 'SIZE', 'CH-B2_median_lag_1',
    'CH-B3_median_lag_5', 'CH-B4_median_lag_5', 'macro_CH-A1_median_rolling_mean_11',
    'macro_CH-A2_median_rolling_mean_11', 'macro_CH-A3_median_rolling_mean_11', 'macro_CH-A4_median_rolling_mean_11',
    'macro_CH-A5_median_rolling_mean_11', 'macro_CH-B1_median_rolling_mean_11', 'macro_CH-B2_median_rolling_mean_11',
    'macro_CH-B2_median_rolling_std_11', 'macro_CH-B3_median_rolling_std_3', 'macro_CH-B3_median_rolling_mean_11',
    'macro_CH-B4_median_rolling_mean_11', 'macro_CH-B5_median_rolling_mean_11', 'macro_CH-A1_median_ewm_mean_11',
    'macro_CH-A1_median_ewm_std_11', 'macro_CH-A2_mean_ewm_mean_11', 'macro_CH-A2_median_ewm_mean_11',
    'macro_CH-A2_median_ewm_std_11', 'macro_CH-A3_mean_ewm_std_11', 'macro_CH-A3_std_ewm_mean_11',
    'macro_CH-A3_median_ewm_mean_11', 'macro_CH-A3_median_ewm_std_11', 'macro_CH-A4_std_ewm_mean_11',
    'macro_CH-A4_median_ewm_mean_11', 'macro_CH-A4_median_ewm_std_11', 'macro_CH-A5_mean_ewm_mean_11',
    'macro_CH-A5_mean_ewm_std_11', 'macro_CH-A5_median_ewm_mean_11', 'macro_CH-A5_median_ewm_std_11',
    'macro_CH-B1_mean_ewm_mean_11', 'macro_CH-B1_std_ewm_mean_11', 'macro_CH-B1_median_ewm_mean_11',
    'macro_CH-B2_std_ewm_mean_11', 'macro_CH-B2_std_ewm_std_11', 'macro_CH-B2_median_ewm_mean_11',
    'macro_CH-B2_median_ewm_std_11', 'macro_CH-B3_std_ewm_std_11', 'macro_CH-B3_median_ewm_mean_11',
    'macro_CH-B3_median_ewm_std_11', 'macro_CH-B4_mean_ewm_mean_11', 'macro_CH-B4_median_ewm_mean_11',
    'macro_CH-B4_median_ewm_std_11', 'macro_CH-B5_std_ewm_mean_11', 'macro_CH-B5_median_ewm_mean_11',
    'macro_CH-B5_median_ewm_std_11', 'normalized_sequence', 'noise_delta_from_start_CH-A3'
]