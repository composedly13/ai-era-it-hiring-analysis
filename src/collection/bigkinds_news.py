"""
빅카인즈(BIGKinds) 뉴스 데이터 로더
빅카인즈 다운로드 CSV/XLSX를 읽어 전처리 파이프라인 입력 형태로 변환한다.
본문은 최대 200자(저작권 제한), 다운로드 최대 20,000건/쿼리.
수집 키워드·기간을 분할해 복수 파일로 관리할 것.
"""

import pandas as pd
from pathlib import Path


NEWS_KEYWORDS = [
    "개발자 채용", "IT 채용", "개발자 취업",
    "주니어 개발자", "신입 개발자",
    "AI 개발자", "AI 인재", "생성형 AI 채용",
    "ChatGPT 개발자", "개발자 감축", "개발자 연봉",
    "소프트웨어 인재", "디지털 인재", "클라우드 인재",
]

# 분석 기간
DATE_BEFORE_AI = ("2021-01-01", "2022-10-31")   # ChatGPT 출시 이전
DATE_AFTER_AI  = ("2022-11-01", "2026-05-29")   # ChatGPT 출시 이후


def load_bigkinds_csv(path: str) -> pd.DataFrame:
    """빅카인즈 내보내기 파일(CSV/XLSX) 로드 및 컬럼 표준화."""
    # TODO: 구현
    raise NotImplementedError


def merge_all(data_dir: str = "data/raw/news") -> pd.DataFrame:
    """복수 분할 파일을 병합해 중복 제거 후 반환."""
    # TODO: 구현
    raise NotImplementedError
