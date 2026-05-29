"""
00_worknet_collection_preprocessing.py
담당: B

워크넷 채용정보 API 수집 → 직무내용 크롤링 → 한국어 전처리 → 기술스택 추출
출력: data/processed/worknet/worknet_processed.csv

출처: 한국고용정보원_워크넷 채용정보, 공공데이터포털 #3038225
라이선스: 이용허락 제4유형 (출처표시·비상업적·변경금지) — 논문에 출처 명시 필수
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.collection.worknet_api import collect_all
from src.collection.worknet_crawler import crawl_batch
from src.preprocessing.korean_preprocessor import preprocess as ko_preprocess
from src.preprocessing.tech_dictionary import normalize, TECH_DICT

API_KEY = os.environ.get("WORKNET_API_KEY", "")
RAW_PATH = "data/raw/worknet"
OUT_PATH = "data/processed/worknet/worknet_processed.csv"

# 사전에 등록된 기술 토큰 전체 집합 (동의어 표준화 후 매칭용)
TECH_SET = {token for tokens in TECH_DICT.values() for token in tokens}


# ---------------------------------------------------------------------------
# Step 1. API 수집
# ---------------------------------------------------------------------------
def step1_collect() -> pd.DataFrame:
    assert API_KEY, "환경변수 WORKNET_API_KEY를 설정하세요."
    df = collect_all(API_KEY, save_path=RAW_PATH)
    print(f"[Step1] 수집 완료: {len(df):,}건 / 컬럼: {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# Step 2. 직무내용 크롤링 (1일차 검증 필수)
# ---------------------------------------------------------------------------
def step2_crawl(df: pd.DataFrame) -> pd.DataFrame:
    # 1건 URL 열어 텍스트/이미지 여부 직접 확인 후 진행
    sample_url = df["채용정보URL"].dropna().iloc[0]
    print(f"[Step2] 샘플 URL (텍스트 여부 육안 확인): {sample_url}")

    df = crawl_batch(df)
    text_ok = df["직무내용"].notna().sum()
    print(f"[Step2] 본문 확보: {text_ok:,}건 / 이미지 등 누락: {len(df) - text_ok:,}건")

    # 본문 누락 공고는 채용제목으로 대체 (안전망)
    df["text"] = df["직무내용"].fillna(df["채용제목"])
    return df


# ---------------------------------------------------------------------------
# Step 3. 한국어 전처리 (KoNLPy)
# ---------------------------------------------------------------------------
def step3_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    from konlpy.tag import Okt
    okt = Okt()

    df["tokens"] = df["text"].apply(lambda x: ko_preprocess(str(x), tagger=okt))
    print(f"[Step3] 전처리 완료. 평균 토큰 수: {df['tokens'].apply(len).mean():.1f}")
    return df


# ---------------------------------------------------------------------------
# Step 4. 기술스택 사전 매칭 + 동의어 표준화
# ---------------------------------------------------------------------------
def step4_extract_tech(df: pd.DataFrame) -> pd.DataFrame:
    def extract(tokens: list[str]) -> list[str]:
        normalized = [normalize(t) for t in tokens]
        return [t for t in normalized if t in TECH_SET]

    df["tech_tokens"] = df["tokens"].apply(extract)
    covered = (df["tech_tokens"].apply(len) > 0).sum()
    print(f"[Step4] 기술스택 추출: {covered:,}건 ({covered/len(df)*100:.1f}%) 공고에서 1개 이상 매칭")
    return df


# ---------------------------------------------------------------------------
# Step 5. 저장
# ---------------------------------------------------------------------------
def step5_save(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[Step5] 저장 완료: {OUT_PATH} ({len(df):,}건)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = step1_collect()
    df = step2_crawl(df)
    df = step3_preprocess(df)
    df = step4_extract_tech(df)
    step5_save(df)
