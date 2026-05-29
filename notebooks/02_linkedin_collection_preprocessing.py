"""
02_linkedin_collection_preprocessing.py
담당: D

Kaggle LinkedIn Job Postings 2023-24 로드 → IT 직무 필터링 → 영어 전처리 → 기술스택 추출
출력: data/processed/linkedin/linkedin_processed.csv

주의 (설계 원칙):
  - 데이터셋이 ChatGPT 출시(2022.11) 이후만 포함 → 'AI 이전/이후' 비교 불가
  - LinkedIn은 글로벌 단면으로만 해석 (국내 시장과 직접 동일시 불가)
  - 한국어 데이터와 토픽 결과를 직접 비교하지 않음

Kaggle 다운로드:
  kaggle datasets download arshkon/linkedin-job-postings -p data/raw/linkedin --unzip
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.collection.linkedin_loader import load_postings, merge_skills, filter_it_jobs
from src.preprocessing.english_preprocessor import preprocess as en_preprocess
from src.preprocessing.tech_dictionary import normalize, TECH_DICT

RAW_DIR = "data/raw/linkedin"
OUT_PATH = "data/processed/linkedin/linkedin_processed.csv"

TECH_SET = {token for tokens in TECH_DICT.values() for token in tokens}


# ---------------------------------------------------------------------------
# Step 1. 파일 로드 + epoch → datetime 변환
# ---------------------------------------------------------------------------
def step1_load() -> pd.DataFrame:
    df = load_postings(RAW_DIR)
    print(f"[Step1] 전체 공고: {len(df):,}건 / 컬럼: {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# Step 2. skills, companies, industries 병합
# ---------------------------------------------------------------------------
def step2_merge(df: pd.DataFrame) -> pd.DataFrame:
    df = merge_skills(df, RAW_DIR)
    print(f"[Step2] skills 병합 후: {len(df):,}건")
    return df


# ---------------------------------------------------------------------------
# Step 3. IT 직무군 필터링
# ---------------------------------------------------------------------------
def step3_filter(df: pd.DataFrame) -> pd.DataFrame:
    df_it = filter_it_jobs(df)
    print(f"[Step3] IT 서브셋: {len(df_it):,}건")
    print(df_it["title"].value_counts().head(10).to_string())
    return df_it


# ---------------------------------------------------------------------------
# Step 4. 영어 전처리 (lemmatization + stopwords)
# ---------------------------------------------------------------------------
def step4_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["tokens"] = df["description"].fillna("").apply(en_preprocess)
    print(f"[Step4] 전처리 완료. 평균 토큰 수: {df['tokens'].apply(len).mean():.1f}")
    return df


# ---------------------------------------------------------------------------
# Step 5. 기술스택 사전 매칭
# ---------------------------------------------------------------------------
def step5_extract_tech(df: pd.DataFrame) -> pd.DataFrame:
    def extract(tokens: list[str]) -> list[str]:
        return [normalize(t) for t in tokens if normalize(t) in TECH_SET]

    # skills 컬럼(구조화)도 함께 활용
    def extract_from_skills(skills_str) -> list[str]:
        if not isinstance(skills_str, str):
            return []
        skills = [s.strip() for s in skills_str.split(",")]
        return [normalize(s) for s in skills if normalize(s) in TECH_SET]

    df["tech_from_desc"]   = df["tokens"].apply(extract)
    df["tech_from_skills"] = df.get("skills", pd.Series([""] * len(df))).apply(extract_from_skills)
    df["tech_tokens"] = (df["tech_from_desc"] + df["tech_from_skills"]).apply(
        lambda x: list(set(x))
    )
    covered = (df["tech_tokens"].apply(len) > 0).sum()
    print(f"[Step5] 기술스택 추출: {covered:,}건 ({covered/len(df)*100:.1f}%) 공고에서 1개 이상 매칭")
    return df


# ---------------------------------------------------------------------------
# Step 6. 저장
# ---------------------------------------------------------------------------
def step6_save(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[Step6] 저장 완료: {OUT_PATH} ({len(df):,}건)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = step1_load()
    df = step2_merge(df)
    df = step3_filter(df)
    df = step4_preprocess(df)
    df = step5_extract_tech(df)
    step6_save(df)
