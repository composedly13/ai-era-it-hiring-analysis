"""
워크넷 채용공고 URL 크롤러
API가 직무내용 자유서술 텍스트를 제공하지 않으므로 채용정보URL 페이지를 직접 파싱한다.
국내 공고 특성상 이미지 게시 비율이 높으므로 텍스트 파싱 성공률을 반드시 검증할 것.
"""

import time
import requests
from bs4 import BeautifulSoup
import pandas as pd


def crawl_job_detail(url: str, delay: float = 1.0) -> dict:
    """단일 채용공고 URL에서 직무내용·자격요건·우대사항 텍스트 추출."""
    # TODO: 구현 — HTML 파싱, 이미지 감지 시 None 반환
    raise NotImplementedError


def crawl_batch(df: pd.DataFrame, url_col: str = "채용정보URL") -> pd.DataFrame:
    """DataFrame의 URL 컬럼을 순회하며 본문 텍스트 추가."""
    # TODO: 구현
    raise NotImplementedError
