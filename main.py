# main.py

import pandas as pd
from tqdm import tqdm

from src import config
from src import data_loader
from src import feature_engineering
from src import train

def main():
    """메인 실행 파이프라인"""
    
    # 출력 폴더 생성
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. LOT 정보 로드 및 BAR 파일 목록 정렬
    lot_info_df = data_loader.create_lot_info_df(config.ROOT_DATA_DIR)
    sorted_bar_meta = data_loader.get_sorted_bar_meta_df(config.ROOT_DATA_DIR, lot_info_df)

    # 2. BAR 파일별 미시 피처 추출
    print("\nStep 3: Extracting micro features from all BAR files...")
    all_features_list = []
    for _, row in tqdm(sorted_bar_meta.iterrows(), total=len(sorted_bar_meta), desc="Processing BAR files"):
        bar_file_path = row['full_path']
        df_raw = data_loader.load_bar_data(bar_file_path)
        
        if df_raw is not None:
            features = feature_engineering.extract_micro_features(df_raw, bar_file_path)
            features['LOT_ID'] = row['LOT_ID']
            features['BAR_FILE_NAME'] = row['BAR_FILE_NAME']
            features['bar_in_lot_sequence'] = row['bar_in_lot_sequence']
            all_features_list.append(features)

    # 3. 데이터프레임 병합 및 기본 전처리
    print("\nStep 4: Creating baseline features DataFrame...")
    baseline_df = pd.DataFrame(all_features_list)
    df = pd.merge(baseline_df, lot_info_df, on='LOT_ID', how='left')

    # 중복 제거 및 정렬
    df.drop_duplicates(subset=['LOT_ID', 'bar_in_lot_sequence'], keep='first', inplace=True)
    df.sort_values(by=['LOT_Start_Time', 'bar_in_lot_sequence'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 4. 외부 피처 로드 및 병합
    external_features_df = data_loader.load_external_features(config.EXTERNAL_FEATURE_FILE)
    df = pd.merge(df, external_features_df, on=['LOT_ID', 'bar_in_lot_sequence'], how='inner')
    
    # 5. 거시적 피처 생성
    df = feature_engineering.add_macro_features(df)

    # 중간 피처 데이터 저장 (선택 사항)
    print(f"\nSaving processed data with {df.shape[1]} features to {config.PROCESSED_DATA_PATH}...")
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    
    # 6. 최종 모델 훈련 및 예측
    final_predictions, _ = train.train_and_evaluate_pipeline(df)

    # 7. 최종 제출 파일 저장
    print(f"\nSaving final submission file to {config.SUBMISSION_PATH}...")
    final_predictions.to_csv(config.SUBMISSION_PATH, index=False)
    
    print("\nPipeline finished successfully")

if __name__ == "__main__":
    main()