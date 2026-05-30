"""
02_linkedin_collection_preprocessing.py
담당: C → A 통합

Kaggle LinkedIn Job Postings 데이터 로드 → IT 직무 필터링 → 본문에서 기술스택 추출 →
직무군 분류 → AI 4-tier 태깅 → 저장.

출력: data/processed/linkedin/linkedin_processed.csv

핵심 설계 결정:
  - listed_time이 2024-04 단면이라 시계열 분석 불가 (단면-단면 비교만)
  - skill_abr (Engineering/IT/Marketing 등 직무 분야 라벨)은 분석 보조용이고
    실제 기술스택은 description 본문에서 tech_dictionary로 추출 → 국내와 동일 단위
  - IT 직무 필터는 title 정규식 광베이스 (linkedin_loader.IT_TITLE_PATTERN)

Kaggle 다운로드 (참고):
  kaggle datasets download arshkon/linkedin-job-postings -p data/raw/linkedin --unzip
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from collections import Counter

from src.collection.linkedin_loader import (
    load_postings, merge_skills, merge_companies, filter_it_jobs,
)
from src.preprocessing.tech_dictionary import extract_skills
from src.preprocessing.role_classifier import classify_from_title

RAW_DIR  = "data/raw/linkedin"
OUT_PATH = "data/processed/linkedin/linkedin_processed.csv"


# ---------------------------------------------------------------------------
# Step 1. postings 로드 + epoch → datetime
# ---------------------------------------------------------------------------
def step1_load() -> pd.DataFrame:
    df = load_postings(RAW_DIR)
    print(f"[Step1] 전체 공고: {len(df):,}건 / 컬럼 {len(df.columns)}개")
    if "listed_time_dt" in df.columns:
        valid = df["listed_time_dt"].dropna()
        print(f"        listed_time     {valid.min()} ~ {valid.max()}")
    if "original_listed_time_dt" in df.columns:
        valid = df["original_listed_time_dt"].dropna()
        print(f"        original_listed {valid.min()} ~ {valid.max()}")
    return df


# ---------------------------------------------------------------------------
# Step 2. job_skills (skill_abr) + 회사 산업 병합
# ---------------------------------------------------------------------------
def step2_merge(df: pd.DataFrame) -> pd.DataFrame:
    df = merge_skills(df, RAW_DIR)
    df = merge_companies(df, RAW_DIR)
    has_skill_abr = df["skill_abrs"].notna().sum()
    print(f"[Step2] skill_abr 보유 {has_skill_abr:,}건 / 회사 산업 조인 완료")
    return df


# ---------------------------------------------------------------------------
# Step 3. IT 직무 필터 (title 광베이스)
# ---------------------------------------------------------------------------
def step3_filter(df: pd.DataFrame) -> pd.DataFrame:
    df_it = filter_it_jobs(df)
    print(f"[Step3] IT 서브셋: {len(df_it):,}건 ({len(df_it)/len(df)*100:.1f}%)")
    print(df_it["title"].value_counts().head(10).to_string())
    return df_it


# ---------------------------------------------------------------------------
# Step 4. description에서 기술스택 추출 (우리 사전 적용)
# ---------------------------------------------------------------------------
def step4_extract_tech(df: pd.DataFrame) -> pd.DataFrame:
    """국내와 동일 단위(canonical 토큰)로 통일.

    - skills      = description 본문에서 tech_dictionary로 추출한 canonical 토큰만
                    (= 국내와 직접 비교 가능한 단위)
    - skill_abrs  / skill_names = LinkedIn의 분야 라벨 (Engineering·Marketing·IT 등),
                    분석 보조용으로 보존하되 stack 비교에는 사용 안 함
    """
    desc = df["description"].fillna("")
    df["skills"] = desc.apply(lambda t: sorted(extract_skills(t)))
    df["n_skills"] = df["skills"].apply(len)
    covered = (df["n_skills"] > 0).sum()
    print(f"[Step4] description에서 기술스택 추출 (canonical 토큰 only): "
          f"{covered:,}건 ({covered/len(df)*100:.1f}%) 1개 이상")
    return df


# ---------------------------------------------------------------------------
# Step 5. 직무군 분류 (영어 title 패턴)
# ---------------------------------------------------------------------------
def step5_classify_role(df: pd.DataFrame) -> pd.DataFrame:
    df["role"] = df["title"].fillna("").apply(classify_from_title)
    dist = df["role"].value_counts()
    print(f"[Step5] 직무 분포: {dict(dist)}")
    return df


# ---------------------------------------------------------------------------
# Step 6. 저장 (국내와 동일 스키마)
# ---------------------------------------------------------------------------
def step6_save(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_cols = ["job_id", "title", "company_name", "industry", "role",
                "formatted_work_type", "n_skills", "skills",
                "listed_time_dt", "original_listed_time_dt",
                "skill_abrs", "skill_names", "description"]
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols].copy()
    df_out["skills"] = df_out["skills"].apply(lambda x: "|".join(x) if isinstance(x, list) else x)
    df_out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[Step6] 저장: {OUT_PATH} ({len(df_out):,}건)")

    # 상위 기술 스택
    flat = [s for lst in df["skills"] for s in (lst if isinstance(lst, list) else [])]
    top = Counter(flat).most_common(15)
    print(f"  상위 기술스택: {[f'{s}({c})' for s, c in top]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = step1_load()
    df = step2_merge(df)
    df = step3_filter(df)
    df = step4_extract_tech(df)
    df = step5_classify_role(df)
    step6_save(df)
