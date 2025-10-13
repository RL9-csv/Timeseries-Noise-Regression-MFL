# src/feature_engineering.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import antropy as ant
import numpy as np
import pandas as pd
from tqdm import tqdm


__all__ = [
    "FeatureConfig",
    "FeatureEngineer",
]


# -----------------------------------------------------------------------------
# 로깅
# -----------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logger = logging.getLogger("feature_engineering")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# -----------------------------------------------------------------------------
# 설정(파라미터) 묶음
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class FeatureConfig:
    # 채널 컬럼 접두어
    ch_prefix: str = "CH"

    # BAR 헤더 파싱
    bar_header_lines: int = 7
    bar_datetime_regex: re.Pattern = re.compile(r"DATE,(\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2})")
    bar_datetime_fmt: str = "%Y.%m.%d %H:%M"

    # Micro Rolling/EWM
    micro_rolling_windows: Sequence[int] = (11, 33)
    micro_ewm_spans: Sequence[int] = (11, 33)

    # Macro Rolling/EWM/Lag
    macro_rolling_windows: Sequence[int] = (3, 11)
    macro_ewm_spans: Sequence[int] = (3, 11)
    lag_numbers: Sequence[int] = (1, 3, 5, 9)

    # 복잡 피처 최소 길이
    perm_entropy_min_len: int = 50
    katz_fd_min_len: int = 2

    # 대상 컬럼 필터링 규칙
    base_stat_suffixes: Sequence[str] = ("std", "mean", "median")
    exclude_keywords: Sequence[str] = ("rolling", "ewm", "entropy", "perm_entropy", "katz_fd", "micro", "movmin30")

    # LOT 문맥 피처 관련
    lot_id_col: str = "LOT_ID"
    lot_seq_col: str = "bar_in_lot_sequence"
    context_noise_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(r"^micro_.*_rolling_std_mean_11$")
    )

    # 수치 안정성
    eps: float = 1e-6


# -----------------------------------------------------------------------------
# 본체
# -----------------------------------------------------------------------------
class FeatureEngineer:
    def __init__(self, config: Optional[FeatureConfig] = None, *, show_progress: bool = True, log: Optional[logging.Logger] = None):
        self.cfg = config or FeatureConfig()
        self.show_progress = show_progress
        self.log = log or logger.getChild(self.__class__.__name__)

    # ---------------------------
    # 유틸
    # ---------------------------
    def _iter(self, iterable: Iterable, desc: str = ""):
        if self.show_progress:
            return tqdm(iterable, desc=desc)
        return iterable

    def _ch_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith(self.cfg.ch_prefix)]

    def _safe_last(self, s: pd.Series) -> Optional[float]:
        try:
            return float(s.iloc[-1])
        except Exception:
            return None

    # ---------------------------
    # Micro Feature Extraction
    # ---------------------------
    def extract_micro_features(self, df_raw: pd.DataFrame, bar_file_path: Path) -> Dict[str, Any]:
        """
        하나의 BAR DataFrame에서 미시적 피처들을 추출.
        반환: {feature_name: value}
        """
        features: Dict[str, Any] = {}
        if df_raw is None or df_raw.empty:
            self.log.warning("빈 DataFrame이므로 micro feature 추출을 건너뜀.")
            return features

        df = df_raw.copy()
        ch_cols = self._ch_cols(df)
        if not ch_cols:
            self.log.warning("채널 컬럼(CH*)이 없음. 기본 통계 피처는 건너뜀.")

        # BAR 시간 정보
        features["BAR_datetime"] = self._parse_bar_datetime(bar_file_path)

        # 기초 통계량
        for col in ch_cols:
            s = df[col]
            features[f"{col}_mean"] = float(s.mean())
            features[f"{col}_std"] = float(s.std())
            features[f"{col}_median"] = float(s.median())
            features[f"{col}_min"] = float(s.min())

        # 블록 4: 미시적 Rolling
        for col in ch_cols:
            signal = df_raw[col].dropna()
            if signal.empty:
                continue
            for w in self.cfg.micro_rolling_windows:
                roll = signal.rolling(window=w, min_periods=1)
                std_series = roll.std()
                mean_series = roll.mean()
                min_series = roll.min()

                features[f"micro_{col}_rolling_std_mean_{w}"] = float(std_series.mean())
                features[f"micro_{col}_rolling_mean_std_{w}"] = float(mean_series.std())
                features[f"micro_{col}_rolling_std_std_{w}"] = float(std_series.std())
                features[f"micro_{col}_rolling_min_mean_{w}"] = float(min_series.mean())
                features[f"micro_{col}_rolling_min_std_{w}"] = float(min_series.std())

        # 블록 5: 미시적 EWM
        for col in ch_cols:
            signal = df_raw[col].dropna()
            if signal.empty:
                continue
            for span in self.cfg.micro_ewm_spans:
                ewm = signal.ewm(span=span, min_periods=1, adjust=False)
                last_mean = self._safe_last(ewm.mean().fillna(0))
                last_std = self._safe_last(ewm.std().fillna(0))
                features[f"micro_{col}_ewm_{span}_mean"] = last_mean
                features[f"micro_{col}_ewm_{span}_std"] = last_std

        # 블록 6/7: 엔트로피 & Katz FD
        for col in ch_cols:
            signal = df_raw[col].dropna().to_numpy()
            try:
                if signal.size >= self.cfg.perm_entropy_min_len:
                    features[f"{col}_perm_entropy"] = float(ant.perm_entropy(signal, normalize=True))
                else:
                    features[f"{col}_perm_entropy"] = None
            except Exception as e:
                self.log.debug("perm_entropy 실패 (%s): %s", col, e)
                features[f"{col}_perm_entropy"] = None

            try:
                if signal.size >= self.cfg.katz_fd_min_len:
                    features[f"{col}_katz_fd"] = float(ant.katz_fd(signal))
                else:
                    features[f"{col}_katz_fd"] = None
            except Exception as e:
                self.log.debug("katz_fd 실패 (%s): %s", col, e)
                features[f"{col}_katz_fd"] = None

        return features

    def _parse_bar_datetime(self, bar_file_path: Path) -> Optional[pd.Timestamp]:
        try:
            with open(bar_file_path, "r", encoding="utf-8", errors="ignore") as f:
                header_content = "".join([next(f) for _ in range(self.cfg.bar_header_lines)])
            m = self.cfg.bar_datetime_regex.search(header_content)
            if not m:
                return None
            return pd.to_datetime(m.group(1), format=self.cfg.bar_datetime_fmt)
        except Exception as e:
            self.log.debug("BAR 헤더 시간 파싱 실패: %s", e)
            return None

    # ---------------------------
    # Macro Feature Engineering
    # ---------------------------
    def _macro_target_cols(self, df: pd.DataFrame) -> List[str]:
        suffixes = tuple(self.cfg.base_stat_suffixes)
        excludes = tuple(self.cfg.exclude_keywords)
        cols = [
            c for c in df.columns
            if (c.endswith(suffixes) and not any(k in c for k in excludes))
        ]
        return cols

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag 피처 생성: 그룹(LOT_ID)별로 지정 래그 시프트."""
        if self.cfg.lot_id_col not in df.columns:
            raise ValueError(f"'{self.cfg.lot_id_col}' 컬럼이 필요합니다.")
        out = df.copy()
        target_cols = self._macro_target_cols(out)

        for col in self._iter(target_cols, desc="  - Adding Lag Features"):
            g = out.groupby(self.cfg.lot_id_col, observed=True)[col]
            for num in self.cfg.lag_numbers:
                out[f"{col}_lag_{num}"] = g.shift(num)
        return out

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """거시적 Rolling 피처: 그룹별 rolling-mean/std"""
        if self.cfg.lot_id_col not in df.columns:
            raise ValueError(f"'{self.cfg.lot_id_col}' 컬럼이 필요합니다.")
        out = df.copy()
        target_cols = self._macro_target_cols(out)

        for col in self._iter(target_cols, desc="  - Adding Rolling Features"):
            g = out.groupby(self.cfg.lot_id_col, observed=True)[col]
            for w in self.cfg.macro_rolling_windows:
                # mean/std 각각 계산
                r = g.rolling(window=w, min_periods=1)
                out[f"macro_{col}_rolling_mean_{w}"] = r.mean().reset_index(level=0, drop=True)
                out[f"macro_{col}_rolling_std_{w}"] = r.std().reset_index(level=0, drop=True)
        return out

    def add_ewm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """거시적 EWM 피처: 그룹별 ewm-mean/std"""
        if self.cfg.lot_id_col not in df.columns:
            raise ValueError(f"'{self.cfg.lot_id_col}' 컬럼이 필요합니다.")
        out = df.copy()
        target_cols = self._macro_target_cols(out)

        for col in self._iter(target_cols, desc="  - Adding EWM Features"):
            g = out.groupby(self.cfg.lot_id_col, observed=True)[col]
            for span in self.cfg.macro_ewm_spans:
                e = g.ewm(span=span, adjust=False, min_periods=1)
                out[f"macro_{col}_ewm_mean_{span}"] = e.mean().reset_index(level=0, drop=True)
                out[f"macro_{col}_ewm_std_{span}"] = e.std().reset_index(level=0, drop=True)
        return out

    def add_lot_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        LOT 내 순서 정보를 활용한 컨텍스트 피처:
        - 시작 시점 대비 변화량/비율
        - 순번 * 변동성
        """
        req = [self.cfg.lot_id_col, self.cfg.lot_seq_col]
        for c in req:
            if c not in df.columns:
                raise ValueError(f"'{c}' 컬럼이 필요합니다.")
        out = df.copy()

        # 노이즈 기준 컬럼 선택 (예: micro_*_rolling_std_mean_11)
        noise_cols = [c for c in out.columns if self.cfg.context_noise_pattern.match(c)]
        if not noise_cols:
            self.log.info("컨텍스트 기준 노이즈 컬럼이 없어 lot context 피처를 건너뜁니다.")
            return out

        # 정규화 순번 (0~1)
        max_seq = out.groupby(self.cfg.lot_id_col, observed=True)[self.cfg.lot_seq_col].transform("max")
        denom = (max_seq - 1).replace(0, 1)  # 단일 바 LOT 보호
        out["normalized_sequence"] = (out[self.cfg.lot_seq_col] - 1) / denom
        out["normalized_sequence"] = out["normalized_sequence"].fillna(0)

        # 시작값 대비 변화량/비율, seq x volatility
        for noise_col in self._iter(noise_cols, desc="  - Adding LOT Context Features"):
            # channel_name 추출(패턴: micro_{CH-??}_rolling_std_mean_11)
            try:
                parts = noise_col.split("_")
                channel_name = parts[1] if len(parts) > 1 else "CH"
            except Exception:
                channel_name = "CH"

            start_noise = out.groupby(self.cfg.lot_id_col, observed=True)[noise_col].transform("first")
            out[f"noise_delta_from_start_{channel_name}"] = out[noise_col] - start_noise
            out[f"noise_ratio_from_start_{channel_name}"] = out[noise_col] / (start_noise + self.cfg.eps)
            out[f"seq_x_volatility_{channel_name}"] = out[self.cfg.lot_seq_col] * out[noise_col]

        return out

    def add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 거시적 피처 생성(래그→롤링→EWM→문맥)"""
        self.log.info("Step 5: Generating Macro Features...")
        out = df.copy()
        out = self.add_lag_features(out)
        out = self.add_rolling_features(out)
        out = self.add_ewm_features(out)
        out = self.add_lot_context_features(out)
        self.log.info("-> Macro feature generation complete.")
        return out