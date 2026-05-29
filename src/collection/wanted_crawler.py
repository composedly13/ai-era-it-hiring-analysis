"""
원티드(Wanted) 개발 직군 채용공고 수집 모듈 — 국내 '현실' 데이터 (볼륨)
========================================================================
개발 부모 카테고리(518)는 offset ~700에서 막히므로 세부 직무 17개 카테고리별로
수집 후 id 중복제거. skill_tags 구조화(~50%) + 본문 100% 제공.

엔드포인트:
  - 목록: https://www.wanted.co.kr/api/v4/jobs?tag_type_ids={dev_subcat}&...
  - 상세: https://www.wanted.co.kr/api/v4/jobs/{id}
"""

import time
import requests

LIST_URL = "https://www.wanted.co.kr/api/v4/jobs"
DETAIL_URL = "https://www.wanted.co.kr/api/v4/jobs/{id}"
PUBLIC_URL = "https://www.wanted.co.kr/wd/{id}"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research)", "Accept": "application/json"}
DELAY = 0.35
PAGE_LIMIT = 100
OFFSET_CAP = 900

# 검증된 '개발' 세부 직무 tag_type_ids
DEV_CATEGORY_TAGS = {
    873: "소프트웨어엔지니어", 872: "웹개발자", 669: "서버개발자", 660: "프론트엔드",
    900: "자바개발자", 899: "C/C++", 671: "파이썬개발자", 672: "node.js개발자",
    1634: "머신러닝엔지니어", 665: "데이터엔지니어", 655: "데이터사이언티스트",
    674: "DevOps/시스템관리", 1024: "DevOps엔지니어", 895: "안드로이드", 677: "iOS",
    676: "게임클라이언트", 1027: "QA엔지니어",
}


def _join_tags(tags, key="title"):
    return "|".join(str(t.get(key, "")) for t in (tags or []) if isinstance(t, dict))


def get_all_ids(limit=None, verbose=True):
    """세부 카테고리별 페이지네이션 → unique id(중복제거)."""
    seen = {}
    for tid, name in DEV_CATEGORY_TAGS.items():
        offset, before = 0, len(seen)
        while True:
            params = {"country": "kr", "tag_type_ids": tid, "job_sort": "job.latest_order",
                      "years": -1, "locations": "all", "limit": PAGE_LIMIT, "offset": offset}
            r = requests.get(LIST_URL, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            j = r.json()
            data = j.get("data", [])
            if not data:
                break
            for d in data:
                seen.setdefault(d["id"], None)
            if not j.get("links", {}).get("next") or offset >= OFFSET_CAP:
                break
            offset += PAGE_LIMIT
            time.sleep(DELAY)
        if verbose:
            print(f"  [wanted:{name}] +{len(seen) - before} (누적 {len(seen)})")
        time.sleep(DELAY)
        if limit and len(seen) >= limit:
            break
    ids = list(seen.keys())
    return ids[:limit] if limit else ids


def fetch_detail(jid):
    r = requests.get(DETAIL_URL.format(id=jid), headers=HEADERS, timeout=15)
    r.raise_for_status()
    j = r.json().get("job", r.json())
    det = j.get("detail", {}) or {}
    company = j.get("company", {}) or {}
    return {
        "source": "wanted", "id": jid,
        "title": j.get("position", ""), "company": company.get("name", ""),
        "skillTags": _join_tags(j.get("skill_tags")),
        "companyTags": _join_tags(j.get("company_tags")),
        "categoryTags": _join_tags(j.get("category_tags")),
        "address": (j.get("address", {}) or {}).get("location", ""),
        "annualFrom": j.get("annual_from"), "annualTo": j.get("annual_to"),
        "dueTime": j.get("due_time", ""),
        "intro": (det.get("intro") or "").strip(),
        "main_tasks": (det.get("main_tasks") or "").strip(),
        "requirements": (det.get("requirements") or "").strip(),
        "preferred_points": (det.get("preferred_points") or "").strip(),
        "url": PUBLIC_URL.format(id=jid),
    }


def collect_wanted(limit=None, verbose=True):
    """원티드 개발 직군 전체 수집 → list[dict]."""
    ids = list(dict.fromkeys(get_all_ids(limit=limit, verbose=verbose)))
    rows = []
    for i, jid in enumerate(ids, 1):
        try:
            rows.append(fetch_detail(jid))
        except Exception as e:
            if verbose:
                print(f"  [wanted] id={jid} 실패: {str(e)[:60]}")
        if verbose and i % 100 == 0:
            print(f"  [wanted] {i}/{len(ids)}")
        time.sleep(DELAY)
    return rows
