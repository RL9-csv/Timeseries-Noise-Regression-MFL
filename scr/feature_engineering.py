# src/feature_engineering.py

import pandas as pd
import numpy as np
import re
import antropy as ant
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

# Micro Feature Extraction
def extract_micro_features(df_raw: pd.DataFrame, bar_file_path: Path) -> Dict[str, Any]:
    """하나의 BAR 데이터프레임에서 미시적 피처들을 추출합니다."""
    if df_raw is None or df_raw.empty:
        return {}
        
    df = df_raw.copy()
    features_dict = {}
    ch_cols = [col for col in df.columns if col.startswith('CH')]

    # BAR 시간 정보
    try:
        with open(bar_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            header_content = "".join([next(f) for _ in range(7)])
        date_match = re.search(r"DATE,(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2})", header_content)
        if date_match:
            features_dict['BAR_datetime'] = pd.to_datetime(date_match.group(1), format='%Y.%m.%d %H:%M')
    except Exception:
        features_dict['BAR_datetime'] = None

    # 기초 통계량
    for col in ch_cols:
        features_dict[f'{col}_mean'] = df[col].mean()
        features_dict[f'{col}_std'] = df[col].std()
        features_dict[f'{col}_median'] = df[col].median()
        features_dict[f'{col}_min'] = df[col].min()

    # ... (Notebook에 있는 나머지 미시적 피처 생성 블록들: Event, Peak, Rolling, EWM, Entropy, Katz FD)
    # 블록 4: 미시적 Rolling 
    for col in ch_cols:
        signal = df_raw[col].dropna()
        if not signal.empty:
            features_dict[f'micro_{col}_rolling_std_mean_11'] = signal.rolling(window=11, min_periods=1).std().mean()
            features_dict[f'micro_{col}_rolling_mean_std_11'] = signal.rolling(window=11, min_periods=1).mean().std()
            features_dict[f'micro_{col}_rolling_std_std_11'] = signal.rolling(window=11, min_periods=1).std().std()
            features_dict[f'micro_{col}_rolling_std_mean_33'] = signal.rolling(window=33, min_periods=1).std().mean()
            features_dict[f'micro_{col}_rolling_mean_std_33'] = signal.rolling(window=33, min_periods=1).mean().std()
            features_dict[f'micro_{col}_rolling_std_std_33'] = signal.rolling(window=33, min_periods=1).std().std()
            features_dict[f'micro_{col}_rolling_min_mean_11'] = signal.rolling(window=11, min_periods=1).min().mean()
            features_dict[f'micro_{col}_rolling_min_std_11'] = signal.rolling(window=11, min_periods=1).min().std()
            features_dict[f'micro_{col}_rolling_min_mean_33'] = signal.rolling(window=33, min_periods=1).min().mean()
            features_dict[f'micro_{col}_rolling_min_std_33'] = signal.rolling(window=33, min_periods=1).min().std()

    # 블록 5: 미시적 EWM
    span_sizes = [11, 33]
    agg_funcs = ['mean', 'std']
    for col in ch_cols:
        signal = df_raw[col].dropna()
        if not signal.empty:
            for span in span_sizes:
                for agg in agg_funcs:
                    ewm = signal.ewm(span=span, min_periods=1)
                    ewm_col = getattr(ewm, agg)()
                    features_dict[f'micro_{col}_ewm_{span}_{agg}'] = ewm_col.fillna(0).iloc[-1]
    
    # 블록 6: 순열 엔트로피 & 블록 7: 카츠 프랙탈 차원
    for col in ch_cols:
        signal = df_raw[col].dropna()
        if len(signal) >= 50:
            features_dict[f'{col}_perm_entropy'] = ant.perm_entropy(signal, normalize=True)
        if len(signal) > 1:
            features_dict[f'{col}_katz_fd'] = ant.katz_fd(signal)

    return features_dict

# Macro Feature Engineering
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag 피처를 생성합니다."""
    target_cols = [col for col in df.columns if (col.endswith('std') or col.endswith('mean') or col.endswith('median')) and not any(k in col for k in ['rolling', 'ewm', 'entropy', 'micro', 'movmin30'])]
    lag_num = [1, 3, 5, 9]
    for col in tqdm(target_cols, desc="  - Adding Lag Features"):
        for num in lag_num:
            df[f'{col}_lag_{num}'] = df.groupby(['LOT_ID'])[col].shift(num)
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """거시적 Rolling 피처를 생성합니다."""
    target_cols = [col for col in df.columns if (col.endswith('std') or col.endswith('mean') or col.endswith('median')) and not any(k in col for k in ['rolling', 'ewm', 'entropy', 'micro', 'movmin30'])]
    windows = [3, 11]
    agg_funcs = ['mean', 'std']
    for col in tqdm(target_cols, desc="  - Adding Rolling Features"):
        for num in windows:
            for agg in agg_funcs:
                rolling_col = getattr(df.groupby(['LOT_ID'])[col].rolling(window=num), agg)()
                df[f'macro_{col}_rolling_{agg}_{num}'] = rolling_col.reset_index(0, drop=True)
    return df

def add_ewm_features(df: pd.DataFrame) -> pd.DataFrame:
    """거시적 EWM 피처를 생성합니다."""
    target_cols = [col for col in df.columns if (col.endswith('std') or col.endswith('mean') or col.endswith('median')) and not any(k in col for k in ['rolling', 'ewm', 'entropy', 'micro', 'movmin30'])]
    spans = [3, 11]
    agg_funcs = ['mean', 'std']
    for col in tqdm(target_cols, desc="  - Adding EWM Features"):
        for num in spans:
            for agg in agg_funcs:
                ewm_col = getattr(df.groupby(['LOT_ID'])[col].ewm(span=num), agg)()
                df[f'macro_{col}_ewm_{agg}_{num}'] = ewm_col.reset_index(0, drop=True)
    return df

def add_lot_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """LOT 내의 순서 정보를 활용한 컨텍스트 피처를 생성합니다."""
    noise_cols = [col for col in df.columns if 'micro' in col and 'rolling_std_mean_11' in col]
    for noise_col in tqdm(noise_cols, desc="  - Adding LOT Context Features"):
        channel_name = noise_col.split('_')[1]
        if 'normalized_sequence' not in df.columns:
            max_sequence = df.groupby('LOT_ID')['bar_in_lot_sequence'].transform('max')
            df['normalized_sequence'] = (df['bar_in_lot_sequence'] - 1) / (max_sequence - 1).replace(0, 1)
            df['normalized_sequence'].fillna(0, inplace=True)
        start_noise = df.groupby('LOT_ID')[noise_col].transform('first')
        df[f'noise_delta_from_start_{channel_name}'] = df[noise_col] - start_noise
        df[f'noise_ratio_from_start_{channel_name}'] = df[noise_col] / (start_noise + 1e-6)
        df[f'seq_x_volatility_{channel_name}'] = df['bar_in_lot_sequence'] * df[noise_col]
    return df

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """모든 거시적 피처 생성 함수들을 순차적으로 실행하는 래퍼 함수입니다."""
    print("\nStep 5: Generating Macro Features...")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_ewm_features(df)
    df = add_lot_context_features(df)
    print("-> Macro feature generation complete.")
    return df