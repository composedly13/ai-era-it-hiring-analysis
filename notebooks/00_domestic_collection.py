"""
00_domestic_collection.py
담당: A

국내 IT 채용공고 수집·통합 (점핏 + 원티드) → 통합 코퍼스
  점핏(techStacks 구조화) + 원티드(skill_tags + 본문) 공개 JSON 수집
  → 공통 스키마 병합 → 스킬 정규화/추출 → 교차중복 제거
출력: data/processed/kr_jobs_clean.csv / .jsonl

수집 방식: 공개 JSON 엔드포인트 호출 + 직접 크롤링 (가산점). 학술용·요청 간 지연.
(국내 소스로 점핏·원티드 채택)
"""

import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from collections import Counter
from src.collection.jumpit_crawler import collect_jumpit
from src.collection.wanted_crawler import collect_wanted
from src.preprocessing.tech_dictionary import normalize_tag, extract_skills

RAW_DIR = "data/raw/domestic"
OUT_CSV = "data/processed/domestic/kr_jobs_clean.csv"
OUT_JSONL = "data/processed/domestic/kr_jobs_clean.jsonl"
SKIP_COLLECT = os.environ.get("SKIP_COLLECT") == "1"   # 1이면 raw 재사용(재크롤 안 함)


def _save_raw(name, rows):
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(f"{RAW_DIR}/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_raw(name):
    p = f"{RAW_DIR}/{name}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else None


def step1_collect():
    if SKIP_COLLECT and _load_raw("jumpit") and _load_raw("wanted"):
        print("[Step1] raw 재사용(SKIP_COLLECT=1)")
        return _load_raw("jumpit"), _load_raw("wanted")
    print("[Step1] 점핏 수집...")
    jp = collect_jumpit()
    _save_raw("jumpit", jp)
    print("[Step1] 원티드 수집...")
    wt = collect_wanted()
    _save_raw("wanted", wt)
    print(f"[Step1] 점핏 {len(jp)} / 원티드 {len(wt)}")
    return jp, wt


def _unify_jumpit(d):
    body = "\n".join(x for x in [d.get("responsibility", ""), d.get("qualifications", ""),
                                 d.get("preferredRequirements", "")] if x)
    return {"source": "jumpit", "id": d.get("id"), "company": d.get("company", ""),
            "title": d.get("title", ""), "career_min": d.get("minCareer"),
            "career_max": d.get("maxCareer"),
            "tags_raw": [s for s in (d.get("techStacks", "") or "").split("|") if s],
            "body": body, "url": d.get("url", "")}


def _unify_wanted(d):
    body = "\n".join(x for x in [d.get("main_tasks", ""), d.get("requirements", ""),
                                 d.get("preferred_points", "")] if x)
    return {"source": "wanted", "id": d.get("id"), "company": d.get("company", ""),
            "title": d.get("title", ""), "career_min": d.get("annualFrom"),
            "career_max": d.get("annualTo"),
            "tags_raw": [s for s in (d.get("skillTags", "") or "").split("|") if s],
            "body": body, "url": d.get("url", "")}


def _norm_key(c, t):
    return ("".join(str(c).split()).lower(), "".join(str(t).split()).lower())


def step2_merge(jp, wt):
    recs = [_unify_jumpit(d) for d in jp] + [_unify_wanted(d) for d in wt]
    rescued = 0
    for r in recs:
        tagged = set()
        for t in r["tags_raw"]:
            tagged |= normalize_tag(t)
        body_sk = extract_skills(r["body"])
        if not tagged and body_sk:
            rescued += 1
        r["skills"] = sorted(tagged | body_sk)
        r["n_skills"] = len(r["skills"])
    # 교차중복(점핏에도 원티드에도 있는 동일 공고)만 제거
    jset = {_norm_key(r["company"], r["title"]) for r in recs if r["source"] == "jumpit"}
    dedup = [r for r in recs
             if not (r["source"] == "wanted" and _norm_key(r["company"], r["title"]) in jset)]
    print(f"[Step2] 통합 {len(dedup)}건 (본문에서 스킬 복원 {rescued}건)")
    return dedup


def step3_save(dedup):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    rows = [{"source": r["source"], "id": r["id"], "company": r["company"], "title": r["title"],
             "career_min": r["career_min"], "career_max": r["career_max"],
             "n_skills": r["n_skills"], "skills": "|".join(r["skills"]),
             "body": r["body"], "url": r["url"]} for r in dedup]
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in dedup:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(dedup)
    by_src = Counter(r["source"] for r in dedup)
    with_skill = sum(1 for r in dedup if r["n_skills"] > 0)
    top = Counter(s for r in dedup for s in r["skills"]).most_common(15)
    print(f"[Step3] 저장: {OUT_CSV} / {OUT_JSONL}")
    print(f"  통합 {n}건 {dict(by_src)} | 스킬 보유 {with_skill/n*100:.0f}%")
    print(f"  상위 기술스택: {[f'{s}({c})' for s, c in top]}")


if __name__ == "__main__":
    jp, wt = step1_collect()
    dedup = step2_merge(jp, wt)
    step3_save(dedup)
