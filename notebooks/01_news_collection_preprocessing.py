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
from src.preprocessing.tech_dictionary import extract_skills
from src.analysis.time_series import CHATGPT_LAUNCH

RAW_DIR = "data/raw/news"
OUT_PATH = "data/processed/news/news_processed.csv"


# ---------------------------------------------------------------------------
# Step 1. 빅카인즈 분할 수집 계획 출력 (UI 작업 가이드)
# ---------------------------------------------------------------------------
def step1_print_query_plan() -> None:
    """전수 수집 가이드 — 빅카인즈 UI에서 키워드 그룹 × 연 단위로 분할 내보내기."""
    print("=== 빅카인즈 수집 쿼리 계획 (전수 수집) ===")
    print("bigkinds.or.kr → 뉴스검색 → 아래 그룹을 OR로 검색, 기간별 분할 export")
    print("저작권: 본문 200자 / 한도: 쿼리당 최대 20,000건\n")

    # 키워드를 5개씩 그룹으로 묶어 한 쿼리에서 OR 검색
    GROUP_SIZE = 5
    groups = [NEWS_KEYWORDS[i:i + GROUP_SIZE]
              for i in range(0, len(NEWS_KEYWORDS), GROUP_SIZE)]

    # 전수 수집: 연 단위 분할 (2026은 5/29까지). 한 파일이 2만 초과면 반기로 더 쪼갤 것.
    years = [
        ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-05-29"),
    ]
    for gi, group in enumerate(groups, 1):
        print(f"[그룹 G{gi}]  " + " OR ".join(group))
        for start, end in years:
            print(f"   {start} ~ {end}  → Group{gi}_{start[:4]}.xlsx")
        print()

    print("주의:")
    print("  - export 건수 20,000 초과 시 해당 연도를 반기로 분할 (예: Group1_2023_H1.xlsx / _H2)")
    print("  - 파일명 형식만 유지하면 됨 — merge_all이 전부 병합·중복제거")
    print("  - 키워드 세트는 전 기간 고정 (분모 일관성 — RQ3 월별 언급률)")
    print("  - 저장 위치: data/raw/news/")


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
# Step 5. 기술스택 추출 (국내공고·LinkedIn과 동일 단위 — extract_skills)
# ---------------------------------------------------------------------------
def step5_extract_tech(df: pd.DataFrame) -> pd.DataFrame:
    """제목+본문에서 canonical 기술토큰 추출. 국내/글로벌과 동일 비교 단위."""
    df["tech_tokens"] = df["text"].apply(lambda t: sorted(extract_skills(str(t))))
    covered = (df["tech_tokens"].apply(len) > 0).sum()
    print(f"[Step5] 기술토큰 추출: {covered:,}건 ({covered/len(df)*100:.1f}%) 1개 이상")
    return df


# ---------------------------------------------------------------------------
# Step 6. 저장 (list 컬럼은 파이프 구분 — 국내/LinkedIn과 동일 포맷)
# ---------------------------------------------------------------------------
def step6_save(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_cols = ["뉴스 식별자", "발행일", "period", "언론사", "제목", "본문",
                "text", "tokens", "tech_tokens"]
    out = df[[c for c in out_cols if c in df.columns]].copy()
    for col in ("tokens", "tech_tokens"):
        if col in out.columns:
            out[col] = out[col].apply(lambda x: "|".join(x) if isinstance(x, list) else x)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[Step6] 저장 완료: {OUT_PATH} ({len(out):,}건)")
    flat = [s for lst in df["tech_tokens"] for s in (lst if isinstance(lst, list) else [])]
    from collections import Counter
    print(f"  뉴스 상위 기술토큰: {[f'{s}({c})' for s, c in Counter(flat).most_common(12)]}")


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
