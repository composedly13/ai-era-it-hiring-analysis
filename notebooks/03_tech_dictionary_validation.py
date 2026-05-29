"""
03_tech_dictionary_validation.py
담당: 미정 (선행 조건: 00·01·02 PR 머지 완료)

3개 데이터소스에서 실제 등장한 어휘로 기술스택 사전을 검증·보완한다.
이후 04~07 분석 스크립트의 공통 입력으로 사용된다.

의의 (Validity Guardrail #2):
  기술스택 사전은 한↔영 비교의 유일한 공통 단위다.
  사전 품질이 RQ4(담론 vs 현실) 비교의 타당성을 직접 결정한다.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from collections import Counter
import pandas as pd
from src.preprocessing.tech_dictionary import TECH_DICT, SYNONYMS, normalize

KR_PATH  = "data/processed/domestic/kr_jobs_clean.csv"
NEWS_PATH     = "data/processed/news/news_processed.csv"
LINKEDIN_PATH = "data/processed/linkedin/linkedin_processed.csv"
DICT_PATH     = "dict/tech_stack_dictionary.json"
SYN_PATH      = "dict/synonyms.json"


# ---------------------------------------------------------------------------
# Step 1. 3개 processed 파일 로드
# ---------------------------------------------------------------------------
def step1_load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_w = pd.read_csv(KR_PATH)
    df_n = pd.read_csv(NEWS_PATH)
    df_l = pd.read_csv(LINKEDIN_PATH)
    print(f"[Step1] 국내공고: {len(df_w):,} / 뉴스: {len(df_n):,} / LinkedIn: {len(df_l):,}")
    return df_w, df_n, df_l


# ---------------------------------------------------------------------------
# Step 2. 전체 어휘 빈도 집계 — 미등록 기술 후보 발굴
# ---------------------------------------------------------------------------
def step2_unknown_vocab(
    df_w: pd.DataFrame,
    df_n: pd.DataFrame,
    df_l: pd.DataFrame,
    top_n: int = 50,
) -> None:
    tech_set = {tok for toks in TECH_DICT.values() for tok in toks}

    all_tokens: list[str] = []
    for df, col in [(df_w, "tokens"), (df_n, "tokens"), (df_l, "tokens")]:
        for row in df[col].dropna():
            try:
                tokens = eval(row) if isinstance(row, str) else row
                all_tokens.extend(tokens)
            except Exception:
                pass

    freq = Counter(all_tokens)
    unknown = {tok: cnt for tok, cnt in freq.most_common(5000)
               if normalize(tok) not in tech_set and len(tok) >= 2}

    print(f"\n[Step2] 미등록 고빈도 어휘 (상위 {top_n}개) — 사전 추가 후보:")
    for tok, cnt in list(unknown.items())[:top_n]:
        print(f"  {tok}: {cnt}")


# ---------------------------------------------------------------------------
# Step 3. 수동 검토 후 dict/ 파일 업데이트
# ---------------------------------------------------------------------------
def step3_update_dict(new_entries: dict[str, list[str]], new_synonyms: dict[str, str]) -> None:
    """
    새 기술스택·동의어를 dict/ 파일에 추가한다.
    변경 후 반드시 git commit으로 기록할 것.

    Args:
        new_entries:  { "category": ["NewTech", ...] }
        new_synonyms: { "원문": "표준표현" }
    """
    with open(DICT_PATH, encoding="utf-8") as f:
        tech_dict = json.load(f)
    with open(SYN_PATH, encoding="utf-8") as f:
        synonyms = json.load(f)

    for cat, tokens in new_entries.items():
        tech_dict.setdefault(cat, [])
        tech_dict[cat] = sorted(set(tech_dict[cat] + tokens))

    synonyms.update(new_synonyms)

    with open(DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(tech_dict, f, ensure_ascii=False, indent=2)
    with open(SYN_PATH, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)

    print("[Step3] dict/ 업데이트 완료 — git add dict/ && git commit -m 'dict: add new tech tokens'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df_w, df_n, df_l = step1_load()
    step2_unknown_vocab(df_w, df_n, df_l)

    # Step3는 수동 검토 후 직접 호출
    # step3_update_dict(
    #     new_entries={"cloud_infra": ["Terraform", "Ansible"]},
    #     new_synonyms={"테라폼": "Terraform"},
    # )
