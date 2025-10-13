# src/data_loader.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any, Optional, Iterable, List

import pandas as pd

# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logger = logging.getLogger("lot")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    file_handler = logging.FileHandler("mylog.txt")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# 예외
# ---------------------------------------------------------------------------
class LotParseError(Exception):
    """LOT 파일 파싱 중 발생하는 예외"""
    pass


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class LotMetadata:
    lot_id: str
    steel_type: Optional[str] = None
    size: Optional[float] = None
    lot_start_time: Optional[datetime] = None
    line_speed: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 필드 사양 정의
#   - name: LotMetadata 필드명
#   - pattern: 캡처 그룹 1개를 가지는 정규식
#   - converter: str -> Any 변환 함수
#   - required: 필수 여부
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    name: str
    pattern: re.Pattern
    converter: Callable[[str], Any]
    required: bool = False


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


def _to_datetime_utc(s: str) -> Optional[datetime]:
    # 예: 20240101123045
    try:
        dt = pd.to_datetime(s, format="%Y%m%d%H%M%S")
        # pandas Timestamp → python datetime
        return dt.to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception:
        return None


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="lot_id",
        pattern=re.compile(r"Lot No\.,(\S+)"),
        converter=str,
        required=True,
    ),
    FieldSpec(
        name="steel_type",
        pattern=re.compile(r"STEEL,(\S+)"),
        converter=str,
    ),
    FieldSpec(
        name="size",
        pattern=re.compile(r"SIZE,([\d\.]+)"),
        converter=_to_float,
    ),
    FieldSpec(
        name="lot_start_time",
        pattern=re.compile(r"LOT Start,(\d+)"),
        converter=_to_datetime_utc,
    ),
    FieldSpec(
        name="line_speed",
        pattern=re.compile(r"LINE SPEED,([\d\.]+)m/min"),
        converter=_to_float,
    ),
)


# ---------------------------------------------------------------------------
# 파서
# ---------------------------------------------------------------------------
class LotParser:
    __slots__ = ("file_path", "encoding", "_log")

    def __init__(self, file_path: Path, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding
        self._log = logger.getChild(self.__class__.__name__)

    def parse(self) -> LotMetadata:
        try:
            content = self.file_path.read_text(encoding=self.encoding, errors="ignore")
        except Exception as e:
            raise LotParseError(f"파일 열기 실패: {self.file_path} ({e})") from e

        field_values: dict[str, Any] = {}
        for spec in FIELD_SPECS:
            m = spec.pattern.search(content)
            if not m:
                if spec.required:
                    raise LotParseError(f"필수 필드 누락: {spec.name} (파일: {self.file_path})")
                self._log.debug("패턴 매칭 실패 (optional): %s", spec.name)
                field_values[spec.name] = None
                continue

            raw = m.group(1).strip()
            value = spec.converter(raw)
            if value is None and spec.required:
                raise LotParseError(f"필수 필드 변환 실패: {spec.name}='{raw}'")
            if value is None:
                self._log.debug("변환 실패 (optional) %s='%s'", spec.name, raw)
            field_values[spec.name] = value

        return LotMetadata(**field_values)  # type: ignore[arg-type]

    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        return float(m.group(1)) if (m := re.search(r"([\d\.]+)", text)) else None

    # 기존 parse_multiple 대체
    @classmethod
    def parse_many(
        cls,
        file_paths: Iterable[Path],
        ignore_errors: bool = True,
        encoding: str = "utf-8",
    ) -> List[LotMetadata]:
        results: List[LotMetadata] = []
        for fp in file_paths:
            parser = cls(fp, encoding=encoding)
            try:
                results.append(parser.parse())
            except LotParseError as e:
                if ignore_errors:
                    logger.warning("파싱 실패(무시): %s (%s)", fp, e)
                else:
                    raise
        return results


# ---------------------------------------------------------------------------
# CLI / 테스트 실행
# ---------------------------------------------------------------------------
def _main():
    import argparse
    import json

    p = argparse.ArgumentParser(description="LOT 파일 메타데이터 파서")
    p.add_argument("paths", nargs="+", help="LOT 파일(들)의 경로")
    p.add_argument("--no-ignore", action="store_true", help="에러 발생 시 즉시 중단")
    p.add_argument("--encoding", default="utf-8")
    args = p.parse_args()

    files = [Path(x) for x in args.paths]
    metas = LotParser.parse_many(files, ignore_errors=not args.no_ignore, encoding=args.encoding)
    print(json.dumps([m.as_dict() for m in metas], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()