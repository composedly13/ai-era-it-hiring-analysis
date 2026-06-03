# 빅카인즈 뉴스 본문 크롤러 — 사용 설명서

`crawl_news_body.py` — 빅카인즈 export(.xlsx) 1개 파일을 받아 URL 컬럼에서 원문 본문을 크롤링하여 JSONL로 저장. **파일 단위 처리**라 group1·group2·기간별 13개 파일을 따로 돌릴 수 있음.

## 1. 사전 준비 (1회만)

```bash
/opt/homebrew/anaconda3/bin/pip install trafilatura httpx pandas openpyxl
```

(이미 시범 크롤 시 설치됨)

## 2. 실행 방법

### 방법 A. 파일 경로를 변수로 지정
파일 상단의 `INPUT_PATH`만 바꾸고 실행:

```python
INPUT_PATH = "/Users/eric1889/Desktop/MacBook/TM/news/group1/NewsResult_20210101-20211231.xlsx"
```

```bash
/opt/homebrew/anaconda3/bin/python3 /Users/eric1889/Desktop/MacBook/TM/crawl_news_body.py
```

### 방법 B. 명령행 인자로 (코드 수정 없이)

```bash
/opt/homebrew/anaconda3/bin/python3 /Users/eric1889/Desktop/MacBook/TM/crawl_news_body.py \
  /Users/eric1889/Desktop/MacBook/TM/news/group1/NewsResult_20210101-20211231.xlsx
```

### 방법 C. 13개 파일 한 번에 (쉘 루프)

```bash
for f in /Users/eric1889/Desktop/MacBook/TM/news/group{1,2}/NewsResult_*.xlsx; do
  echo "=== $f ==="
  /opt/homebrew/anaconda3/bin/python3 /Users/eric1889/Desktop/MacBook/TM/crawl_news_body.py "$f"
done
```

> 순차 실행 권장. 병렬로 13개 동시에 돌리면 같은 언론사 서버에 과부하 줄 수 있음.

## 3. 결과 파일

입력 xlsx와 **같은 폴더**에 두 가지 결과 자동 생성:

```
news/group1/NewsResult_20210101-20211231.xlsx              ← 원본 (보존)
       ├── NewsResult_20210101-20211231.body.jsonl         ← 점진 저장 (중간 안전판)
       └── NewsResult_20210101-20211231_with_body.xlsx     ← 본문 컬럼 추가된 새 xlsx
```

### `..._with_body.xlsx` — 사용자가 주로 쓸 결과
원본 xlsx 모든 컬럼 + 신규 3개 컬럼:

| 추가 컬럼 | 설명 |
|---|---|
| `본문_원문` | 크롤링한 원문 본문 (실패 시 빈 문자열) |
| `본문_원문_길이` | 본문 자 수 |
| `본문_원문_상태` | `ok` / `no_url` / `HTTP_403` / `no_extract` / `err_*` |

> 기존 `본문` (빅카인즈 200자)은 그대로 보존됨 — 크롤 실패 시 fallback 활용 가능.

### `.body.jsonl` — 중간 점진 저장 파일
중단·재개용. 한 줄 스키마:
```json
{"news_id": "...", "status": "ok", "body": "...", "len": 1842, "url": "https://..."}
```

크롤링이 끝나면 자동으로 위 jsonl을 xlsx에 머지. 따로 머지 단계 실행 불필요.

### 머지만 따로 돌리기 (jsonl만 있고 xlsx 머지를 다시 하고 싶을 때)
```bash
python3 crawl_news_body.py /path/to/NewsResult_20210101-20211231.xlsx --merge-only
```

## 4. 중단·재개

처리 중 Ctrl+C 또는 머신 다운 시:
- 이미 저장된 `news_id`는 자동 스킵
- 같은 명령 다시 실행하면 남은 건만 처리

테스트 시 일부만 돌려보고 싶다면, 미리 출력 파일 일부분만 작성하거나 `INPUT_PATH`를 작은 파일로 바꾸면 됨.

## 5. 동작 원리

```
xlsx 로드 → 처리 완료된 ID 스킵 → 비동기 큐
  ├ 동시 16건까지 (CONCURRENCY)
  ├ 같은 도메인은 0.6초 간격 보장 (PER_DOMAIN_GAP_SEC) — robots 친화
  ├ trafilatura가 HTML에서 본문만 추출 (광고·메뉴·댓글 제거)
  └ 결과를 한 줄씩 JSONL에 즉시 flush (중단 안전)
```

**핵심 라이브러리**
- `httpx.AsyncClient` — 비동기 HTTP 클라이언트
- `trafilatura` — 다국어 본문 추출 (한국 주요 언론사 잘 지원)
- `asyncio.Semaphore` — 전역 동시성 제한
- 도메인별 `next_allowed` 타임스탬프 — 같은 도메인 연속 요청 방지

**진행률 출력 예시**
```
   100/10,651 (  0.9%) · 성공률  88.0% · 14.2 req/s · ETA  12.4분
   200/10,651 (  1.9%) · 성공률  85.5% · 15.1 req/s · ETA  11.5분
```

## 6. 예상 시간 (파일별)

| 파일 | 건수 | ETA |
|---|---:|---:|
| group1 2021 | 10,651 | ~12분 |
| group1 2022 | 10,853 | ~12분 |
| group1 2023 | 8,755 | ~10분 |
| group1 2024 | 7,768 | ~9분 |
| group1 2025 | 8,068 | ~9분 |
| group1 2026 | 3,257 | ~4분 |
| group2 2021 | 4,967 | ~6분 |
| group2 2022 | 4,619 | ~5분 |
| group2 2023 | 7,697 | ~9분 |
| group2 2024 | 13,837 | ~16분 |
| group2 2025상 | 10,822 | ~13분 |
| group2 2025하 | 19,770 | ~23분 |
| group2 2026 | 18,859 | ~22분 |
| **합계** | **129,923** | **~2.5시간** |

> 도메인 rate limit에 종속. 같은 언론사가 많이 몰리면 더 느림.
> 동시 16건 × 빠른 도메인 ~50ms 응답 = 이론상 200 req/s지만 실제 10-20 req/s 예상.

## 7. 성공률 (200건 시범 결과)

상위 20개 언론사 = 전체 76.5% 차지

**성공 17개사** (평균 본문 850~5,300자):
전자신문 · 머니투데이 · 이데일리 · 매일경제 · 한국경제 · 서울경제 · 아시아경제 ·
디지털타임스 · 동아일보 · 헤럴드경제 · 중앙일보 · 뉴스핌 · 파이낸셜뉴스 · 아주경제 ·
이투데이 · 아시아투데이 · 브레이크뉴스

**실패 3개사**:
- **EBN**: URL 컬럼 자체가 비어있음 → 어차피 불가
- **데일리안 · 조선일보**: trafilatura가 본문 추출 못 함 (페이지 구조 다름 / paywall)

전체 예상 성공률: **약 70-80%** (성공 못한 행은 status 컬럼에 사유 기록되어 추후 분석 가능)

## 8. 분석 단계에서의 fallback

크롤 실패 행도 분석 가능하게:

```python
# 본문 = (크롤 성공 시 원문 / 실패 시 빅카인즈 200자)
full["text"] = full["body"].fillna("").where(full["status"]=="ok", full["본문"])
```

즉 13만 건 모두 분석 가능 — 풀 본문이 안 되는 행은 200자만 활용.

## 9. 설정 튜닝 (필요 시)

코드 상단 상수:

```python
CONCURRENCY        = 16    # 동시 16건 (도메인 부하 무관)
PER_DOMAIN_GAP_SEC = 0.6   # 같은 도메인 0.6초 간격 (보수적)
TIMEOUT_SEC        = 15    # 요청 타임아웃
RETRIES            = 2     # 실패 시 재시도 횟수
```

- **빠르게**: `PER_DOMAIN_GAP_SEC` 0.3, `CONCURRENCY` 32 → 성공률 약간 떨어질 수 있음
- **느리게·안전하게**: `PER_DOMAIN_GAP_SEC` 1.0 → robots 더 친화적, 더 느림

## 10. 주의 사항

- 학술 연구용·비상업 활용 전제
- 언론사 robots.txt와 약관 준수
- 본문은 분석용으로만 사용, **본문 전재·재배포 금지**
- 보고서·논문에는 발췌·인용만 (저작권법 제28조 공정 이용 범위 내)
- 빅카인즈 약관도 동일 — 200자 본문 분석은 허용되지만 외부 공개는 제한
