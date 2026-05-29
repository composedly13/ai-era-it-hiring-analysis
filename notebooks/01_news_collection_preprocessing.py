"""
01_news_collection_preprocessing.py
담당: C

빅카인즈 내보내기 파일 병합 → 한국어 전처리 → before/after 라벨 부착 → 기술스택 추출
출력: data/processed/news/news_processed.csv

제약:
  - 본문 최대 200자 (저작권 제한) — 분석 단위: 제목 + 200자 리드 + 제공 키워드
  - 다운로드 최대 20,000건/쿼리 → 키워드·기간 분할 수집 필수
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.collection.bigkinds_news import merge_all, NEWS_KEYWORDS
from src.preprocessing.korean_preprocessor import preprocess as ko_preprocess
from src.preprocessing.tech_dictionary import normalize, TECH_DICT
from src.analysis.time_series import before_after_comparison, CHATGPT_LAUNCH

RAW_DIR = "data/raw/news"
OUT_PATH = "data/processed/news/news_processed.csv"

TECH_SET = {token for tokens in TECH_DICT.values() for token in tokens}


# ---------------------------------------------------------------------------
# Step 1. 빅카인즈 분할 수집 계획 출력 (UI 작업 가이드)
# ---------------------------------------------------------------------------
def step1_print_query_plan() -> None:
    print("=== 빅카인즈 수집 쿼리 계획 ===")
    print("아래 키워드·기간 조합으로 빅카인즈(bigkinds.or.kr)에서 CSV/XLSX 내보내기:")
    print()

    periods = [
        ("2021-01-01", "2022-10-31", "before_ai"),
        ("2022-11-01", "2024-12-31", "after_ai_1"),
        ("2025-01-01", "2026-05-29", "after_ai_2"),
    ]
    for start, end, label in periods:
        print(f"  기간: {start} ~ {end}  →  파일명: news_{label}.xlsx")
    print()
    print("  키워드 그룹 A:", " OR ".join(NEWS_KEYWORDS[:5]))
    print("  키워드 그룹 B:", " OR ".join(NEWS_KEYWORDS[5:10]))
    print("  키워드 그룹 C:", " OR ".join(NEWS_KEYWORDS[10:]))
    print()
    print("  각 파일 20,000건 이하 확인 → data/raw/news/ 에 저장")


# ---------------------------------------------------------------------------
# Step 2. 파일 병합 + 중복 제거
# ---------------------------------------------------------------------------
def step2_merge() -> pd.DataFrame:
    df = merge_all(RAW_DIR)
    print(f"[Step2] 병합 후: {len(df):,}건 / 컬럼: {df.columns.tolist()}")
    return df


# ---------------------------------------------------------------------------
# Step 3. before/after 라벨 부착
# ---------------------------------------------------------------------------
def step3_label(df: pd.DataFrame, date_col: str = "발행일") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    df["period"] = df[date_col].apply(
        lambda d: "after" if str(d.date()) >= CHATGPT_LAUNCH else "before"
    )
    before = (df["period"] == "before").sum()
    after  = (df["period"] == "after").sum()
    print(f"[Step3] before(~2022.10): {before:,}건 / after(2022.11~): {after:,}건")
    return df


# ---------------------------------------------------------------------------
# Step 4. 한국어 전처리
# ---------------------------------------------------------------------------
def step4_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    from konlpy.tag import Okt
    okt = Okt()

    # 분석 단위: 제목 + 본문(200자) 합산
    df["text"] = df["제목"].fillna("") + " " + df["본문"].fillna("")
    df["tokens"] = df["text"].apply(lambda x: ko_preprocess(str(x), tagger=okt))
    print(f"[Step4] 전처리 완료. 평균 토큰 수: {df['tokens'].apply(len).mean():.1f}")
    return df


# ---------------------------------------------------------------------------
# Step 5. 기술스택 사전 매칭
# ---------------------------------------------------------------------------
def step5_extract_tech(df: pd.DataFrame) -> pd.DataFrame:
    def extract(tokens: list[str]) -> list[str]:
        return [normalize(t) for t in tokens if normalize(t) in TECH_SET]

    df["tech_tokens"] = df["tokens"].apply(extract)
    return df


# ---------------------------------------------------------------------------
# Step 6. 저장
# ---------------------------------------------------------------------------
def step6_save(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[Step6] 저장 완료: {OUT_PATH} ({len(df):,}건)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    step1_print_query_plan()

    df = step2_merge()
    df = step3_label(df)
    df = step4_preprocess(df)
    df = step5_extract_tech(df)
    step6_save(df)
