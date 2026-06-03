"""
01_news_collection_preprocessing.py

빅카인즈 통합본 (data/raw/news/all_news_with_body.xlsx) 전처리 → 분석용 csv.

흐름:
  Step 1. load_news — 56,963건 (본문 크롤링 ok 53,479건 / 93.9%)
  Step 2. 분석 텍스트 — 본문 ok면 제목+본문, 실패면 제목만
  Step 3. 기술스택 추출 (정규식, 빠름) — extract_skills 사전 매칭
  Step 4. AI 4-tier 라벨 부착 (tier_a/b/c/d 0/1 컬럼)
  Step 5. 제목·본문 KoNLPy 토큰화 (워드클라우드·빈도용, ~15분)
  Step 6. 저장: data/processed/news/news_processed.csv

출력 컬럼:
  news_id · date · year · month · media · title · period · url
  body_status · body_len ·
  skills (str, ';' 구분) · n_skills (int) ·
  tier_a_coding_tool · tier_b_model_platform · tier_c_ml_skill · tier_d_generic ·
  title_tokens (str, ' ' 구분) · body_tokens (str, ' ' 구분; body_status!=ok면 빈 문자열)
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from tqdm import tqdm

from src.collection.bigkinds_news import load_news, analysis_text, CHATGPT_LAUNCH, NEWS_KEYWORDS
from src.preprocessing.korean_preprocessor import preprocess as ko_preprocess, _get_okt
from src.preprocessing.tech_dictionary import extract_skills, get_ai_tier, AI_TIERS

INPUT_PATH = "data/raw/news/all_news_with_body.xlsx"
OUT_PATH = "data/processed/news/news_processed.csv"

TIER_KEYS = list(AI_TIERS.keys())  # tier_a_coding_tool, ..., tier_d_generic


# ---------------------------------------------------------------------------
# Step 1. 로드
# ---------------------------------------------------------------------------
def step1_load() -> pd.DataFrame:
    print(f"[Step1] {INPUT_PATH} 로드")
    df = load_news(path=INPUT_PATH, ok_only=False)
    n_ok = (df["body_status"] == "ok").sum()
    n_before = (df["period"] == "before").sum()
    n_after = (df["period"] == "after").sum()
    print(f"  전체 {len(df):,}건 (본문 ok {n_ok:,} · {n_ok/len(df)*100:.1f}%)")
    print(f"  before {n_before:,} · after {n_after:,} (cutoff = {CHATGPT_LAUNCH.date()})")
    print(f"  기간 {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# Step 2-3. 기술스택 추출 + AI tier 라벨
# ---------------------------------------------------------------------------
def step2_extract_skills(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[Step2] 분석 텍스트 + 기술스택 추출 ({len(df):,}건)")
    t0 = time.time()
    df["analysis_text"] = df.apply(analysis_text, axis=1)
    skills_sets = [extract_skills(t) for t in tqdm(df["analysis_text"], desc="extract_skills")]
    df["skills_set"] = skills_sets
    df["skills"] = df["skills_set"].apply(lambda s: ";".join(sorted(s)))
    df["n_skills"] = df["skills_set"].apply(len)

    # AI 4-tier 라벨
    for tier in TIER_KEYS:
        tier_skills = set(AI_TIERS[tier])
        df[tier] = df["skills_set"].apply(lambda s: int(bool(s & tier_skills)))

    n_any_skill = (df["n_skills"] > 0).sum()
    print(f"  추출 완료 ({time.time()-t0:.1f}s) · 평균 {df['n_skills'].mean():.2f}개/건 · "
          f"하나 이상 매칭 {n_any_skill:,}건 ({n_any_skill/len(df)*100:.1f}%)")
    for tier in TIER_KEYS:
        cnt = df[tier].sum()
        print(f"    {tier:30s} {cnt:>6,}건 ({cnt/len(df)*100:5.2f}%)")
    return df


# ---------------------------------------------------------------------------
# Step 3. KoNLPy 토큰화 (제목·본문)
# ---------------------------------------------------------------------------
def step3_tokenize(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[Step3] KoNLPy(Okt) 토큰화 — 약 15분 예상")
    okt = _get_okt()

    print("  제목 토큰화...")
    t0 = time.time()
    title_tokens = [
        " ".join(ko_preprocess(t, tagger=okt))
        for t in tqdm(df["title"].fillna(""), desc="title")
    ]
    df["title_tokens"] = title_tokens
    print(f"  제목 완료 ({time.time()-t0:.0f}s)")

    print("  본문 토큰화 (ok 행만)...")
    t0 = time.time()
    body_tokens = []
    bodies = df["body"].fillna("").tolist()
    statuses = df["body_status"].tolist()
    for status, body in zip(tqdm(statuses, desc="body"), bodies):
        if status == "ok" and body:
            body_tokens.append(" ".join(ko_preprocess(body, tagger=okt)))
        else:
            body_tokens.append("")
    df["body_tokens"] = body_tokens
    print(f"  본문 완료 ({time.time()-t0:.0f}s · {(time.time()-t0)/60:.1f}분)")
    return df


# ---------------------------------------------------------------------------
# Step 4. 저장
# ---------------------------------------------------------------------------
def step4_save(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_cols = [
        "news_id", "date", "year", "month", "media", "title", "url", "period",
        "body_status", "body_len",
        "skills", "n_skills",
        *TIER_KEYS,
        "title_tokens", "body_tokens",
    ]
    out = df[out_cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"[Step4] 저장: {OUT_PATH} · {len(out):,}건 · {size_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = step1_load()
    df = step2_extract_skills(df)
    df = step3_tokenize(df)
    step4_save(df)
