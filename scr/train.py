# src/train.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config


# -----------------------------------------------------------------------------
# 로깅
# -----------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logger = logging.getLogger("train")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class TrainingConfig:
    # 예측 수평선: y_target = shift(-horizon)
    horizon: int = 10
    # LOT 기준 학습 비율
    train_ratio: float = 0.8
    # 범주형 기본 컬럼
    categorical_cols: Tuple[str, ...] = ("STEEL_TYPE",)
    # 그룹핑 기본 키
    lot_id_col: str = "LOT_ID"
    # LOT 내 순번
    lot_seq_col: str = "bar_in_lot_sequence"
    # LOT 시간 컬럼(최초시간으로 LOT 정렬)
    bar_dt_col: str = "BAR_datetime"
    # 제출 대상을 만들 때 참조할 “대표 타깃 원천” 컬럼명
    # 예: config.TARGET_SOURCE_COLS[0]
    target_source_for_mask: Optional[str] = None
    # OneHotEncoder 옵션
    ohe_drop_first: bool = True
    # 수치 임퓨팅 기본값
    num_fill_value: float = 0.0


CFG = TrainingConfig(
    horizon=10,
    train_ratio=0.8,
    target_source_for_mask=(config.TARGET_SOURCE_COLS[0] if getattr(config, "TARGET_SOURCE_COLS", None) else None),
)


# -----------------------------------------------------------------------------
# 유틸
# -----------------------------------------------------------------------------
def _build_preprocessor(X: pd.DataFrame, categorical_cols: List[str], *, num_fill_value: float = 0.0) -> ColumnTransformer:
    """숫자/범주형 자동 분리 후 ColumnTransformer 구성."""
    # 실제로 존재하는 범주형만 사용
    cat_cols = [c for c in categorical_cols if c in X.columns]
    # 숫자형 컬럼
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # 중복/충돌 방지
    num_cols = [c for c in num_cols if c not in cat_cols]

    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=num_fill_value)),
        ]
    )
    # sklearn 버전에 따라 OneHotEncoder 인자가 다릅니다.
    # 하위호환 위해 sparse=False 사용 (경고가 나올 수 있지만 동작은 문제 없음)
    categorical_tf = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first" if CFG.ohe_drop_first else None, sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",  # 지정되지 않은 컬럼은 드롭(누수 방지)
    )
    return preprocessor


def _time_ordered_lot_split(df: pd.DataFrame, *, lot_id_col: str, bar_dt_col: str, train_ratio: float) -> Tuple[List[Any], List[Any]]:
    """
    LOT 단위로 시간 순 정렬 후 학습/검증 LOT 분리.
    BAR_datetime이 없거나 일부 LOT이 NaT이면 대체 전략(알파벳/정렬) 사용.
    """
    if lot_id_col not in df.columns:
        raise ValueError(f"'{lot_id_col}' 컬럼이 필요합니다.")

    # LOT 별 최소 시간
    if bar_dt_col in df.columns:
        lot_start_times = df.groupby(lot_id_col)[bar_dt_col].min()
        # 전부 NaT 인 경우 대비
        if lot_start_times.isna().all():
            logger.warning("모든 LOT의 BAR_datetime이 NaT입니다. LOT_ID 정렬로 대체합니다.")
            lot_order = sorted(lot_start_times.index.tolist())
        else:
            lot_order = lot_start_times.sort_values(kind="mergesort").index.tolist()
    else:
        logger.warning("BAR_datetime 컬럼이 없어 LOT_ID 정렬로 대체합니다.")
        lot_order = sorted(df[lot_id_col].unique().tolist())

    split_point = max(1, int(len(lot_order) * train_ratio))
    train_lots = lot_order[:split_point]
    valid_lots = lot_order[split_point:] if split_point < len(lot_order) else []
    return train_lots, valid_lots


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"r2": r2, "mae": mae, "rmse": rmse, "mse": mse}


def _safe_expm1(a: np.ndarray) -> np.ndarray:
    """overflow 방지용 expm1 래퍼."""
    a = np.clip(a, -50, 50)  # 매우 큰 값 클리핑
    return np.expm1(a)


# -----------------------------------------------------------------------------
# 데이터 준비
# -----------------------------------------------------------------------------
def prepare_channel_data(
    df: pd.DataFrame,
    channel_name: str,
    selected_features: List[str],
    *,
    horizon: int,
    lot_id_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    특정 채널에 대한 X(피처), y(로그변환 타깃), 가공된 채널 DF 반환.
    - y_target = 그룹(LOT)별 target_source를 shift(-horizon)
    - 누락치 제거 후 y_log = log1p(y_target)
    """
    if df is None or df.empty:
        raise ValueError("입력 df가 비어 있습니다.")

    channel_df = df.copy()
    target_source = f"{channel_name}_movmin30_mean"
    if target_source not in channel_df.columns:
        raise KeyError(f"타깃 원천 컬럼이 없습니다: {target_source}")

    # 그룹별 미래 시점 라벨
    channel_df["y_target"] = channel_df.groupby(lot_id_col)[target_source].shift(-horizon)
    # 학습 가능한 행만
    channel_df = channel_df.dropna(subset=["y_target"]).copy()
    channel_df["y_log"] = np.log1p(channel_df["y_target"])

    # 항상 필요한 범주형(예: STEEL_TYPE) 추가
    base_cats = list(CFG.categorical_cols)
    features_to_use = list(set(selected_features + base_cats))
    # 실제 존재하는 것만
    features_to_use = [f for f in features_to_use if f in channel_df.columns]

    X = channel_df[features_to_use].copy()
    y = channel_df["y_log"].copy()
    return X, y, channel_df


# -----------------------------------------------------------------------------
# 채널 학습/예측
# -----------------------------------------------------------------------------
def _train_one_channel(
    df: pd.DataFrame,
    channel: str,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    채널 하나에 대해:
    - 데이터 준비
    - LOT 단위 시간 순 학습/홀드아웃 분리
    - 파이프라인 구성/학습
    - 전체 X에 대한 예측 생성
    - 홀드아웃 평가 메트릭 기록
    반환:
      - pred_series: 원본 df 인덱스에 맞춘 예측(레벨 복원)
      - info: 메트릭/채널명/학습 LOT 수 등 요약
    """
    # 준비
    X, y, ch_df = prepare_channel_data(
        df=df,
        channel_name=channel,
        selected_features=config.SELECTED_FEATURES,
        horizon=CFG.horizon,
        lot_id_col=CFG.lot_id_col,
    )

    # 전처리/모델
    preprocessor = _build_preprocessor(X, list(CFG.categorical_cols), num_fill_value=CFG.num_fill_value)
    model = lgb.LGBMRegressor(**getattr(config, "BEST_LGBM_PARAMS", {}))
    pipe = Pipeline(steps=[("prep", preprocessor), ("reg", model)])

    # LOT 단위 학습/검증 분리
    train_lots, valid_lots = _time_ordered_lot_split(
        ch_df, lot_id_col=CFG.lot_id_col, bar_dt_col=CFG.bar_dt_col, train_ratio=CFG.train_ratio
    )
    train_idx = ch_df[ch_df[CFG.lot_id_col].isin(train_lots)].index
    valid_idx = ch_df[ch_df[CFG.lot_id_col].isin(valid_lots)].index

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_valid, y_valid = (X.loc[valid_idx], y.loc[valid_idx]) if len(valid_idx) else (None, None)

    # 학습
    pipe.fit(X_train, y_train)
    logger.info("Channel=%s | train_lots=%d valid_lots=%d | X_train=%d X_valid=%d",
                channel, len(train_lots), len(valid_lots), X_train.shape[0], 0 if X_valid is None else X_valid.shape[0])

    # 평가(홀드아웃 LOT이 있을 때만)
    metrics: Dict[str, float] = {}
    if X_valid is not None and len(X_valid):
        y_valid_hat_log = pipe.predict(X_valid)
        y_valid_hat = _safe_expm1(y_valid_hat_log)
        y_valid_true = np.expm1(y_valid.to_numpy())
        metrics = _compute_metrics(y_valid_true, y_valid_hat)
        logger.info("Channel=%s | Holdout metrics: %s", channel, metrics)

    # 전체 예측(로그 → 레벨 복원)
    y_all_hat_log = pipe.predict(X)
    y_all_hat = _safe_expm1(y_all_hat_log)

    # 예측 시리즈(원본 인덱스에 맵핑)
    pred_series = pd.Series(y_all_hat, index=X.index, name=f"{channel}_predicted")

    info: Dict[str, Any] = {
        "channel": channel,
        "train_lots": len(train_lots),
        "valid_lots": len(valid_lots),
        **{f"holdout_{k}": v for k, v in metrics.items()},
    }
    return pred_series, info


def _submission_mask(df: pd.DataFrame, *, horizon: int, lot_id_col: str, target_source: Optional[str]) -> pd.Series:
    """
    제출 대상 마스크:
    - 그룹별로 target_source를 shift(-horizon) 했을 때 NaN이 되는 위치(= 미래)만 True
    - target_source가 없으면 전체 False
    """
    if target_source is None or target_source not in df.columns:
        logger.warning("제출 마스크 기준 target_source가 없으므로 빈 제출을 반환합니다. target_source=%s", target_source)
        return pd.Series(False, index=df.index)

    mask = df.groupby(lot_id_col)[target_source].shift(-horizon).isna()
    # index 정렬/정합
    mask = mask.reindex(df.index, fill_value=False)
    return mask


# -----------------------------------------------------------------------------
# 공개 API
# -----------------------------------------------------------------------------
def train_and_evaluate_pipeline(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    모든 채널에 대해 모델을 학습하고 전체 예측을 생성.
    반환:
      - submission_df: 예측 대상(미래) 행만 추린 예측 결과(채널별 *_predicted 컬럼)
      - analysis_df:   채널별 홀드아웃 메트릭/요약
    """
    if df is None or df.empty:
        raise ValueError("입력 df가 비어 있습니다.")

    logger.info("===== Stage 6: Final Fold Analysis & Prediction Generation =====")

    # 제출 마스크 계산(미래행)
    submit_mask = _submission_mask(
        df,
        horizon=CFG.horizon,
        lot_id_col=CFG.lot_id_col,
        target_source=CFG.target_source_for_mask,
    )

    # 결과 테이블 준비
    preds_df = pd.DataFrame(index=df.index)
    preds_df[[CFG.lot_id_col, CFG.lot_seq_col]] = df[[CFG.lot_id_col, CFG.lot_seq_col]]

    analysis_rows: List[Dict[str, Any]] = []

    # 채널별 학습/예측
    for channel in getattr(config, "CHANNELS", []):
        logger.info("--- Processing Final Model for Channel: %s ---", channel)
        pred_series, info = _train_one_channel(df, channel)
        preds_df[pred_series.name] = pred_series
        analysis_rows.append(info)

    # 제출 객체: 미래만 선별
    submission_df = preds_df.loc[submit_mask].copy()
    # 제출에는 LOT/순번 제외(원 코드 호환)
    drop_cols = [c for c in (CFG.lot_id_col, CFG.lot_seq_col) if c in submission_df.columns]
    submission_df = submission_df.drop(columns=drop_cols)

    analysis_df = pd.DataFrame(analysis_rows)

    logger.info("-> Prediction generation complete. submission_rows=%d", submission_df.shape[0])
    return submission_df, analysis_df