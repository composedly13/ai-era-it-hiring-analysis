"""
워크넷 채용정보 API 수집 모듈
공공데이터포털 #3038225 (한국고용정보원_워크넷 채용정보 채용목록 및 상세정보)
출처표시 필수 (이용허락 제4유형: 출처표시·비상업적·변경금지)
"""

import os
import time
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

# 수집 대상 IT 키워드
IT_KEYWORDS = [
    "개발자", "소프트웨어", "웹 개발", "앱 개발",
    "프론트엔드", "백엔드", "풀스택",
    "Java", "Python", "JavaScript", "React", "Spring",
    "데이터 분석", "인공지능", "AI", "머신러닝",
    "클라우드", "정보보안", "DevOps",
]

BASE_URL = "https://www.work.go.kr/opi/opi/opia/wantedApiList.do"
DETAIL_URL = "https://www.work.go.kr/opi/opi/opia/wantedApiDetail.do"


def fetch_job_list(api_key: str, keyword: str, start: int = 1, display: int = 100) -> list[dict]:
    """채용목록 API 호출 → 파싱된 공고 리스트 반환."""
    # TODO: 구현
    raise NotImplementedError


def fetch_job_detail(api_key: str, wanted_auth_no: str) -> dict:
    """채용상세 API 호출 (callTp=D) → 회사 상세정보 포함 dict 반환."""
    # TODO: 구현
    raise NotImplementedError


def collect_all(api_key: str, save_path: str = "data/raw/worknet") -> pd.DataFrame:
    """전체 IT 키워드에 대해 목록 + 상세 수집 후 DataFrame 반환."""
    # TODO: 구현
    raise NotImplementedError
