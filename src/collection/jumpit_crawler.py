"""
점핏(Jumpit) IT 채용공고 수집 모듈 — 국내 '현실' 데이터
========================================================
dev 전용 잡보드. techStacks 구조화 + 본문(주요업무/자격요건/우대사항) 텍스트 제공.
공개 JSON 엔드포인트 호출(= API 활용 + 직접 수집, 가산점). 학술용·요청 간 지연.

엔드포인트:
  - 목록: https://api.jumpit.co.kr/api/positions?page=N
  - 상세: https://api.jumpit.co.kr/api/position/{id}
"""

import time
import requests

LIST_URL = "https://api.jumpit.co.kr/api/positions"
DETAIL_URL = "https://api.jumpit.co.kr/api/position/{id}"
PUBLIC_URL = "https://www.jumpit.co.kr/position/{id}"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research)", "Accept": "application/json"}
DELAY = 0.35


def _join_stacks(stacks):
    out = []
    for s in stacks or []:
        out.append(s["stack"] if isinstance(s, dict) else str(s))
    return "|".join(out)


def _join_names(items):
    out = []
    for it in items or []:
        if isinstance(it, dict):
            out.append(str(it.get("name") or it.get("jobCategory")
                           or it.get("title") or next(iter(it.values()), "")))
        else:
            out.append(str(it))
    return ",".join(out)


def get_all_position_ids(limit=None):
    """목록 API 페이지네이션으로 전체 포지션 id 수집."""
    ids, page = [], 1
    while True:
        r = requests.get(LIST_URL, params={"sort": "relation", "highlight": "false", "page": page},
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        positions = r.json().get("result", {}).get("positions", [])
        if not positions:
            break
        ids.extend(p["id"] for p in positions)
        if limit and len(ids) >= limit:
            return ids[:limit]
        page += 1
        time.sleep(DELAY)
    return ids


def fetch_detail(pid):
    """상세 API → 분석용 레코드 dict."""
    r = requests.get(DETAIL_URL.format(id=pid), headers=HEADERS, timeout=15)
    r.raise_for_status()
    d = r.json().get("result", {})
    return {
        "source": "jumpit", "id": pid,
        "title": d.get("title", ""), "company": d.get("companyName", ""),
        "jobCategories": _join_names(d.get("jobCategories")),
        "techStacks": _join_stacks(d.get("techStacks")),
        "minCareer": d.get("minCareer"), "maxCareer": d.get("maxCareer"),
        "education": d.get("educationName") or d.get("education"),
        "location": d.get("location", ""),
        "publishedAt": d.get("publishedAt", ""), "closedAt": d.get("closedAt", ""),
        "responsibility": (d.get("responsibility") or "").strip(),
        "qualifications": (d.get("qualifications") or "").strip(),
        "preferredRequirements": (d.get("preferredRequirements") or "").strip(),
        "url": PUBLIC_URL.format(id=pid),
    }


def collect_jumpit(limit=None, verbose=True):
    """점핏 전체 수집 → list[dict]."""
    ids = get_all_position_ids(limit=limit)
    rows = []
    for i, pid in enumerate(ids, 1):
        try:
            rows.append(fetch_detail(pid))
        except Exception as e:
            if verbose:
                print(f"  [jumpit] id={pid} 실패: {str(e)[:60]}")
        if verbose and i % 100 == 0:
            print(f"  [jumpit] {i}/{len(ids)}")
        time.sleep(DELAY)
    return rows
