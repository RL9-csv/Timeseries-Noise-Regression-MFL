# src/data_loader.py

import pandas as pd
import re
import os
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

def parse_lot_file(file_path: Path) -> Dict[str, Any]:
    """LOT.CSV 파일 하나를 파싱하여 메타데이터를 추출합니다."""
    metadata = {}
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lot_no_match = re.search(r"Lot No\.,(\S+)", content)
    steel_match = re.search(r"STEEL,(\S+)", content)
    size_match = re.search(r"SIZE,([\d\.]+)", content)
    start_time_match = re.search(r"LOT Start,(\d+)", content)
    line_speed_match = re.search(r"LINE SPEED,([\d\.]+)m/min", content)

    if lot_no_match: metadata['LOT_ID'] = lot_no_match.group(1).strip()
    if steel_match: metadata['STEEL_TYPE'] = steel_match.group(1).strip()
    if size_match: metadata['SIZE'] = float(size_match.group(1).strip())
    if start_time_match:
        metadata['LOT_Start_Time'] = pd.to_datetime(start_time_match.group(1), format='%Y%m%d%H%M%S')
    if line_speed_match: metadata['LINE_SPEED'] = float(line_speed_match.group(1).strip())
    
    return metadata

def create_lot_info_df(folder_path: Path) -> pd.DataFrame:
    """지정된 폴더 내 모든 LOT.CSV 파일을 찾아 LOT 정보 데이터프레임을 생성합니다."""
    print("Step 1: Creating LOT info DataFrame...")
    all_lot_files = list(folder_path.glob('**/LOT.CSV'))
    lot_data_list = [parse_lot_file(f) for f in tqdm(all_lot_files, desc="Parsing LOT files")]
    lot_info_df = pd.DataFrame(lot_data_list)
    lot_info_df.drop_duplicates(subset=['LOT_ID'], keep='first', inplace=True)
    print(f"-> Parsed {len(lot_info_df)} unique LOT files.")
    return lot_info_df

def get_sorted_bar_meta_df(root_folder: Path, lot_info_df: pd.DataFrame) -> pd.DataFrame:
    """모든 BAR 파일 경로를 찾아 시간 순서대로 정렬된 메타 데이터프레임을 반환합니다."""
    print("\nStep 2: Finding and pre-sorting all BAR file paths...")
    all_bar_files_metadata = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.startswith('BAR') and file.endswith('.CSV'):
                lot_id = Path(root).name
                bar_no_match = re.search(r'BAR(\d+)', file)
                if bar_no_match:
                    bar_no = int(bar_no_match.group(1))
                    all_bar_files_metadata.append({
                        'LOT_ID': lot_id,
                        'bar_in_lot_sequence': bar_no,
                        'BAR_FILE_NAME': file,
                        'full_path': Path(root) / file
                    })

    bar_meta_df = pd.DataFrame(all_bar_files_metadata)
    bar_meta_df = pd.merge(bar_meta_df, lot_info_df[['LOT_ID', 'LOT_Start_Time']], on='LOT_ID', how='left')
    sorted_bar_meta_df = bar_meta_df.sort_values(by=['LOT_Start_Time', 'bar_in_lot_sequence'])
    print(f"-> Success! Found and sorted {len(sorted_bar_meta_df)} BAR files.")
    return sorted_bar_meta_df

def load_bar_data(bar_file_path: Path) -> pd.DataFrame:
    """BAR.CSV 파일 하나를 읽어 데이터프레임으로 반환합니다."""
    try:
        with open(bar_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(6): next(f)
            header_line = next(f)
            column_names = [name.strip() for name in header_line.strip().split(',')]
        df_raw = pd.read_csv(bar_file_path, skiprows=7, header=None, names=column_names, low_memory=False)
        return df_raw
    except Exception as e:
        print(f"Warning: Could not read {bar_file_path.name}: {e}")
        return None

def load_external_features(file_path: Path) -> pd.DataFrame:
    """외부 피처가 포함된 CSV 파일을 불러옵니다."""
    print(f"\nStep 4: Loading external features from {file_path.name}...")
    ex_data = pd.read_csv(file_path)
    ex_data.rename(columns={'LOT': 'LOT_ID'}, inplace=True)
    ex_data['bar_in_lot_sequence'] = ex_data.groupby('LOT_ID').cumcount() + 1
    
    feature_cols = ['LOT_ID', 'bar_in_lot_sequence'] + [col for col in ex_data.columns if 'movmin30_mean' in col]
    print(f"-> Loaded {len(ex_data)} rows and selected {len(feature_cols)} feature columns.")
    return ex_data[feature_cols]