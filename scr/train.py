# src/train.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import List, Dict, Any, Tuple

from src import config

def prepare_channel_data(df: pd.DataFrame, channel_name: str, selected_features: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """특정 채널에 대한 X, y 데이터를 준비하고 선택된 피처만 반환합니다."""
    channel_df = df.copy()
    target_source = f'{channel_name}_movmin30_mean'
    
    channel_df['y_target'] = channel_df.groupby('LOT_ID')[target_source].shift(-10)
    channel_df.dropna(subset=['y_target'], inplace=True)
    channel_df['y_log'] = np.log1p(channel_df['y_target'])
    
    # STEEL_TYPE은 범주형 피처로 항상 필요하므로, selected_features에 없으면 추가합니다.
    features_to_use = list(set(selected_features + ['STEEL_TYPE']))
    features_to_use = [f for f in features_to_use if f in channel_df.columns]
    
    X = channel_df[features_to_use]
    y = channel_df['y_log']
    
    return X, y, channel_df

def train_and_evaluate_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    모든 채널에 대해 최종 모델을 훈련하고, 예측을 생성하며,
    심층적인 폴드 분석을 수행합니다.
    """
    print("\n===== Stage 6: Final Fold Analysis & Prediction Generation =====\n")
    
    final_predictions_df = pd.DataFrame(index=df.index)
    final_predictions_df[['LOT_ID', 'bar_in_lot_sequence']] = df[['LOT_ID', 'bar_in_lot_sequence']]
    fold_analysis_results = []

    for channel in config.CHANNELS:
        print(f"--- Processing Final Model for Channel: {channel} ---")
        X, y, channel_df = prepare_channel_data(df, channel, config.SELECTED_FEATURES)

        # 파이프라인 최종 정의
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = [col for col in ['STEEL_TYPE'] if col in X.columns]
        
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ], 
            remainder='passthrough'
        )
        
        final_model = lgb.LGBMRegressor(**config.BEST_LGBM_PARAMS)
        final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', final_model)])

        # 훈련 데이터 정의 (LOT 기준 80%)
        lot_start_times = channel_df.groupby('LOT_ID')['BAR_datetime'].min()
        sorted_lots = lot_start_times.sort_values().index.tolist()
        split_point = int(len(sorted_lots) * 0.8)
        train_lots = sorted_lots[:split_point]
        
        train_df_indices = channel_df[channel_df['LOT_ID'].isin(train_lots)].index
        X_train_final, y_train_final = X.loc[train_df_indices], y.loc[train_df_indices]

        # 최종 모델 훈련
        final_pipeline.fit(X_train_final, y_train_final)

        # 예측 생성
        log_predict_all = final_pipeline.predict(X)
        final_predictions_df[f'{channel}_predicted'] = pd.Series(np.expm1(log_predict_all), index=X.index)

    # 최종 예측 결과에서 y_target이 없는 행(예측 대상)만 필터링
    submission_df = final_predictions_df[df[config.TARGET_SOURCE_COLS[0]].shift(-10).isna()]

    return submission_df.drop(columns=['LOT_ID', 'bar_in_lot_sequence']), pd.DataFrame() # Fold analysis can be added back if needed