"""
빅카인즈 뉴스 본문 크롤러 — 파일 단위 처리
============================================
빅카인즈 export(xlsx) 1개 파일을 받아 URL 컬럼으로 원문 본문 크롤 → JSONL 저장.

사용법:
  1) 아래 INPUT_PATH만 바꾸고 실행:
        python3 crawl_news_body.py
  2) 또는 명령행 인자:
        python3 crawl_news_body.py /path/to/NewsResult_20210101-20211231.xlsx

특징:
  - 비동기 + 도메인별 rate limit (같은 언론사 0.6초 간격)
  - 점진 JSONL 저장 (중단 후 같은 명령 재실행 시 처리된 건은 자동 스킵)
  - trafilatura 본문 추출 (한국 주요 언론사 대부분 지원)
  - 실패 사유 status에 기록 (no_url / HTTP_xxx / no_extract / err_*)

결과 JSONL 스키마:
  {news_id, status, body, len, url}
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd
import trafilatura

# ============== 설정 — 파일명만 여기서 바꾸면 됨 ==============
INPUT_PATH = "/Users/eric1889/Desktop/MacBook/TM/scrape_parts/part2_group1_2024-2026.csv"
# ============================================================

CONCURRENCY        = 16     # 동시 요청 수 (전체)
PER_DOMAIN_GAP_SEC = 0.6    # 같은 도메인 요청 사이 최소 간격
TIMEOUT_SEC        = 15
RETRIES            = 2
PROGRESS_EVERY     = 100    # N건마다 진행률 출력

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}


def output_path_for(input_path: Path) -> Path:
    """입력 xlsx와 같은 폴더에 동일 이름의 .body.jsonl로 저장."""
    return input_path.with_suffix(".body.jsonl")


def load_done_ids(output_path: Path) -> set:
    """이미 처리된 뉴스 식별자 set (중단 재개용)."""
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                done.add(d["news_id"])
            except Exception:
                pass
    return done


async def fetch_one(client: httpx.AsyncClient, domain_next_allowed: dict, row: dict) -> dict:
    """단일 URL 본문 추출. (worker가 호출 — 동시성 제어는 worker 수로)"""
    nid = str(row["뉴스 식별자"])
    url = row.get("URL")
    if pd.isna(url) or not url:
        return {"news_id": nid, "status": "no_url", "body": "", "len": 0, "url": None}

    domain = urlparse(url).netloc
    # 도메인별 rate limit
    now = time.monotonic()
    wait = max(0.0, domain_next_allowed.get(domain, 0.0) - now)
    if wait > 0:
        await asyncio.sleep(wait)
    domain_next_allowed[domain] = time.monotonic() + PER_DOMAIN_GAP_SEC

    for attempt in range(RETRIES + 1):
        try:
            r = await client.get(url)
            if r.status_code == 200:
                body = trafilatura.extract(
                    r.text, include_comments=False, include_tables=False
                ) or ""
                return {
                    "news_id": nid,
                    "status": "ok" if body else "no_extract",
                    "body": body,
                    "len": len(body),
                    "url": url,
                }
            elif r.status_code in (403, 429, 503):
                if attempt == RETRIES:
                    return {"news_id": nid, "status": f"HTTP_{r.status_code}",
                            "body": "", "len": 0, "url": url}
                await asyncio.sleep(2)
                continue
            else:
                return {"news_id": nid, "status": f"HTTP_{r.status_code}",
                        "body": "", "len": 0, "url": url}
        except Exception as e:
            if attempt == RETRIES:
                return {"news_id": nid, "status": f"err_{type(e).__name__}",
                        "body": "", "len": 0, "url": url}
            await asyncio.sleep(1)

    # 모든 재시도가 rate-limit(403/429/503)으로 소진된 경우의 안전망
    return {"news_id": nid, "status": "retry_exhausted", "body": "", "len": 0, "url": url}


async def worker(queue: asyncio.Queue, client: httpx.AsyncClient,
                 domain_next_allowed: dict, out_f, write_lock: asyncio.Lock,
                 stats: dict, total: int, t0: float):
    """큐에서 row를 하나씩 꺼내 처리. CONCURRENCY 개수만큼 worker 동시 실행."""
    while True:
        try:
            row = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            result = await fetch_one(client, domain_next_allowed, row)
            async with write_lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                stats["completed"] += 1
                if result["status"] == "ok":
                    stats["ok"] += 1
                if stats["completed"] % PROGRESS_EVERY == 0:
                    out_f.flush()
                    elapsed = time.time() - t0
                    rate = stats["completed"] / elapsed
                    eta_min = (total - stats["completed"]) / rate / 60 if rate > 0 else 0
                    print(f"  {stats['completed']:>6,}/{total:,} "
                          f"({stats['completed']/total*100:5.1f}%) · "
                          f"성공률 {stats['ok']/stats['completed']*100:5.1f}% · "
                          f"{rate:4.1f} req/s · ETA {eta_min:5.1f}분")
        finally:
            queue.task_done()


async def run(input_path: Path):
    output_path = output_path_for(input_path)
    print(f"[입력 ] {input_path}")
    print(f"[출력 ] {output_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    print(f"[로드 ] 전체 {len(df):,}건")

    done = load_done_ids(output_path)
    if done:
        print(f"[재개 ] 기존 완료 {len(done):,}건 스킵")

    rows = df[~df["뉴스 식별자"].astype(str).isin(done)].to_dict("records")
    total = len(rows)
    print(f"[작업 ] 남은 {total:,}건 · worker {CONCURRENCY}개\n")
    if total == 0:
        print("이미 모두 처리됨.")
        return

    # worker queue 패턴 — 메모리 일정, 큰 파일도 안전
    queue: asyncio.Queue = asyncio.Queue()
    for r in rows:
        queue.put_nowait(r)

    domain_next_allowed: dict = {}
    write_lock = asyncio.Lock()
    stats = {"completed": 0, "ok": 0}
    t0 = time.time()

    timeout = httpx.Timeout(TIMEOUT_SEC, connect=5)
    async with httpx.AsyncClient(headers=HEADERS, timeout=timeout, follow_redirects=True) as client:
        with open(output_path, "a", encoding="utf-8") as out_f:
            workers = [
                asyncio.create_task(
                    worker(queue, client, domain_next_allowed, out_f, write_lock, stats, total, t0)
                )
                for _ in range(CONCURRENCY)
            ]
            await asyncio.gather(*workers)

    elapsed = time.time() - t0
    print(f"\n[완료 ] {stats['completed']:,}건 · 성공률 {stats['ok']/max(stats['completed'],1)*100:.1f}% · "
          f"소요 {elapsed/60:.1f}분")
    print(f"[저장 ] {output_path}")

    # xlsx에 본문 컬럼 추가하여 새 파일로 저장
    merge_to_xlsx(input_path, output_path)


def merge_to_xlsx(input_path: Path, jsonl_path: Path) -> None:
    """크롤링 결과 JSONL을 xlsx에 병합하여 새 xlsx로 저장.

    원본 보존, 새 파일 생성: NewsResult_YYYY_with_body.xlsx
    추가 컬럼:
      - 본문_원문: 크롤링 본문 (실패 시 빈 문자열)
      - 본문_원문_길이: 자 수
      - 본문_원문_상태: ok / no_url / HTTP_xxx / no_extract / err_*
    """
    out_xlsx = input_path.with_name(input_path.stem + "_with_body.xlsx")
    print(f"\n[병합 ] {jsonl_path.name} → {out_xlsx.name}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if not jsonl_path.exists():
        print(f"  jsonl 없음 — 병합 스킵")
        return

    bodies = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if isinstance(d, dict) and d.get("news_id"):
                    bodies.append(d)
            except Exception:
                pass
    if not bodies:
        print(f"  jsonl 비어있음 — 병합 스킵")
        return

    b = pd.DataFrame(bodies)
    b = b.rename(columns={"body": "본문_원문", "len": "본문_원문_길이", "status": "본문_원문_상태"})
    b = b[["news_id", "본문_원문", "본문_원문_길이", "본문_원문_상태"]]
    b["news_id"] = b["news_id"].astype(str)

    # 재개·재시도로 같은 news_id가 여러 번 기록될 수 있음 → 중복 제거(성공·긴 본문 우선)
    b["_ok"] = (b["본문_원문_상태"] == "ok").astype(int)
    b = (b.sort_values(["_ok", "본문_원문_길이"], ascending=False)
           .drop_duplicates(subset="news_id", keep="first")
           .drop(columns="_ok"))

    df["뉴스 식별자"] = df["뉴스 식별자"].astype(str)
    merged = df.merge(b, left_on="뉴스 식별자", right_on="news_id", how="left").drop(columns=["news_id"])

    # 통계
    ok = (merged["본문_원문_상태"] == "ok").sum()
    avg_len = merged.loc[merged["본문_원문_상태"] == "ok", "본문_원문_길이"].mean()
    print(f"  성공 {ok:,}/{len(merged):,}건 ({ok/len(merged)*100:.1f}%) · 평균 본문 {avg_len:.0f}자")

    merged.to_excel(out_xlsx, index=False, engine="openpyxl")
    print(f"  저장: {out_xlsx}")


def main():
    """사용법:
        python3 crawl_news_body.py /path/to/file.xlsx              # 크롤 + 자동 머지
        python3 crawl_news_body.py /path/to/file.xlsx --merge-only # jsonl만 있을 때 머지만
        python3 crawl_news_body.py                                  # 상단 INPUT_PATH 사용
    """
    args = sys.argv[1:]
    merge_only = "--merge-only" in args
    args = [a for a in args if not a.startswith("--")]

    input_path = Path(args[0]) if args else Path(INPUT_PATH)
    if not input_path.exists():
        sys.exit(f"파일 없음: {input_path}")

    if merge_only:
        merge_to_xlsx(input_path, output_path_for(input_path))
    else:
        asyncio.run(run(input_path))


if __name__ == "__main__":
    main()
