"""
News_data_final 통합 스크립트
=============================
4개 part(jsonl + csv)를 각각 하나의 파일로 통합.

출력 (News_data_final/):
  - all_news.body.jsonl   : 4개 jsonl 통합 + news_id 중복제거(성공·긴 본문 우선)
  - all_news_with_body.xlsx: csv 메타 + 본문 병합, 통일 스키마, 중복제거

통일 스키마(xlsx):
  뉴스 식별자 · 발행일 · 언론사 · 제목 · URL · 본문_원문 · 본문_원문_길이 · 본문_원문_상태
"""
import glob
import json
import os

import pandas as pd

BASE = "/Users/eric1889/Desktop/MacBook/TM/News_data_final"
CSV_DIR = "/Users/eric1889/Desktop/MacBook/TM/scrape_parts"

OUT_JSONL = os.path.join(BASE, "all_news.body.jsonl")
OUT_XLSX = os.path.join(BASE, "all_news_with_body.xlsx")


def best(a: dict, b: dict) -> dict:
    """같은 news_id 두 레코드 중 더 좋은 것: ok 우선 → 본문 긴 것 우선."""
    a_ok = a.get("status") == "ok"
    b_ok = b.get("status") == "ok"
    if a_ok != b_ok:
        return a if a_ok else b
    return a if a.get("len", 0) >= b.get("len", 0) else b


# ── 1) jsonl 통합 + 중복제거 ────────────────────────────────
best_by_id: dict = {}
jsonl_files = sorted(glob.glob(os.path.join(BASE, "*.body.jsonl")))
jsonl_files = [f for f in jsonl_files if os.path.basename(f) != os.path.basename(OUT_JSONL)]
total_lines = 0
for f in jsonl_files:
    n = 0
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if not isinstance(d, dict) or not d.get("news_id"):
            continue
        n += 1
        nid = str(d["news_id"])
        d["news_id"] = nid
        if nid in best_by_id:
            best_by_id[nid] = best(best_by_id[nid], d)
        else:
            best_by_id[nid] = d
    total_lines += n
    print(f"[jsonl] {os.path.basename(f):45} {n:>6,}줄")

with open(OUT_JSONL, "w", encoding="utf-8") as out:
    for d in best_by_id.values():
        out.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"[통합 jsonl] 입력 {total_lines:,}줄 → 고유 {len(best_by_id):,}건 (중복 {total_lines-len(best_by_id):,} 제거)")
print(f"  저장: {OUT_JSONL}\n")


# ── 2) csv 통합 + 본문 병합 + 중복제거 ──────────────────────
csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "part*.csv")))
dfs = []
for f in csv_files:
    df = pd.read_csv(f, encoding="utf-8-sig")
    dfs.append(df)
    print(f"[csv] {os.path.basename(f):40} {len(df):>6,}행")
meta = pd.concat(dfs, ignore_index=True)
meta["뉴스 식별자"] = meta["뉴스 식별자"].astype(str)
meta = meta.drop_duplicates(subset="뉴스 식별자", keep="first")
print(f"[통합 csv] 고유 메타 {len(meta):,}건")

# 본문 DataFrame
bodies = pd.DataFrame(best_by_id.values())
bodies = bodies.rename(columns={"body": "본문_원문", "len": "본문_원문_길이", "status": "본문_원문_상태"})
bodies = bodies[["news_id", "본문_원문", "본문_원문_길이", "본문_원문_상태"]]
bodies["news_id"] = bodies["news_id"].astype(str)

merged = meta.merge(bodies, left_on="뉴스 식별자", right_on="news_id", how="left").drop(columns=["news_id"])

ok = (merged["본문_원문_상태"] == "ok").sum()
avg_len = merged.loc[merged["본문_원문_상태"] == "ok", "본문_원문_길이"].mean()
print(f"[병합] {len(merged):,}건 · 본문성공 {ok:,}건 ({ok/len(merged)*100:.1f}%) · 평균 {avg_len:.0f}자")

merged.to_excel(OUT_XLSX, index=False, engine="openpyxl")
print(f"  저장: {OUT_XLSX}")
