"""
LinkedIn Job Postings 2023-2024 (Kaggle: arshkon/linkedin-job-postings) 로더
postings.csv + skills.csv + companies.csv + industries.csv 병합 후 IT 직무군 필터링.
ChatGPT 출시 이후 데이터만 포함되므로 "AI 이전/이후" 비교에 사용 불가.
LinkedIn 데이터는 글로벌 단면으로만 해석할 것.
"""

import pandas as pd
from pathlib import Path

# 분석 대상 IT 직무 타이틀 패턴
IT_TITLE_PATTERNS = [
    r"software engineer", r"frontend engineer", r"backend engineer",
    r"fullstack engineer", r"full.stack engineer",
    r"data scientist", r"data analyst", r"data engineer",
    r"machine learning engineer", r"ml engineer", r"ai engineer",
    r"devops engineer", r"cloud engineer", r"security engineer",
    r"web developer", r"mobile developer",
]


def load_postings(data_dir: str = "data/raw/linkedin") -> pd.DataFrame:
    """postings.csv 로드 및 epoch → datetime 변환."""
    # TODO: 구현
    raise NotImplementedError


def merge_skills(postings: pd.DataFrame, data_dir: str = "data/raw/linkedin") -> pd.DataFrame:
    """skills.csv를 postings와 조인."""
    # TODO: 구현
    raise NotImplementedError


def filter_it_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """IT 직무 타이틀 패턴으로 필터링."""
    # TODO: 구현
    raise NotImplementedError
