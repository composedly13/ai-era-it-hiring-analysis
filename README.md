# 생성형 AI 시대 IT 채용시장의 요구역량과 채용 담론 변화 분석

> **한 줄 요약:** 언론은 "AI가 개발자를 대체한다"고 말하지만, 실제 채용공고는 무엇을 요구하는가?
> 뉴스 담론(국내 장기 시계열) · 점핏·원티드(국내 현재 현실) · LinkedIn(글로벌 현실) 세 데이터를 결합해 **담론과 현실의 간극**과 **국내·글로벌 격차**를 데이터로 검증한다.

| 항목 | 내용 |
|---|---|
| 과목 | 텍스트마이닝 |
| 팀 구성 | 4명 |
| 제출 마감 | 2026-06-15 23:59 |
| 발표 | 2026-06-16 11:00 |
| 데이터 규모 | 국내 3,274 + LinkedIn(IT 수만) + 뉴스(수천) → 전체 수만 건+ |

---

## 디렉터리 구조

```
ai-era-it-hiring-analysis/
├── notebooks/                        # 파이프라인 실행 스크립트 (순서대로)
│   ├── README.md
│   ├── 00_domestic_collection.py     점핏+원티드 수집·통합 (담당: A)
│   ├── 01_news_collection_preprocessing.py         담당: B
│   ├── 02_linkedin_collection_preprocessing.py     담당: C
│   ├── 03_tech_dictionary_validation.py            담당: 전원
│   ├── 04_eda_keyword_analysis.py                  담당: 전원
│   ├── 05_topic_modeling_bertopic.py               담당: 전원
│   ├── 06_classification_model.py                  RQ6 (보조)
│   ├── 07_rq4_discourse_vs_reality.py              ← RQ4 핵심 (전원)
│   └── 08_visualization_summary.py                 담당: 전원
│
├── src/
│   ├── collection/
│   │   ├── jumpit_crawler.py         점핏 공개 JSON 수집 (techStacks 구조화)
│   │   ├── wanted_crawler.py         원티드 공개 JSON 수집 (개발 17개 카테고리)
│   │   ├── bigkinds_news.py          빅카인즈 CSV 로더
│   │   └── linkedin_loader.py        Kaggle 데이터셋 로더 + IT 필터
│   ├── preprocessing/
│   │   ├── korean_preprocessor.py   KoNLPy 형태소 분석
│   │   ├── english_preprocessor.py  lemmatization · stopwords
│   │   └── tech_dictionary.py       기술스택 사전 + 정규식 추출 + 동의어 표준화
│   ├── analysis/
│   │   ├── frequency_analysis.py
│   │   ├── tfidf_analysis.py
│   │   ├── cooccurrence_network.py  Co-occurrence Network (수업 외)
│   │   ├── word2vec_analysis.py     Word2Vec 공기어 (수업 외·선택)
│   │   ├── time_series.py           시계열 (before/after ChatGPT)
│   │   └── gap_index.py             담론-현실 간극지수 (Spearman·순위차)
│   ├── modeling/
│   │   ├── bertopic_model.py        BERTopic (수업 외)
│   │   ├── job_classifier.py        직무군 분류 (보조·라벨누수 차단)
│   │   └── evaluation.py            Accuracy · F1 · Confusion Matrix
│   └── visualization/
│       ├── wordcloud_gen.py
│       ├── network_plot.py
│       ├── umap_plot.py             UMAP 2D 시각화 (수업 외)
│       └── comparison_chart.py      담론 vs 현실 / 국내 vs 글로벌 차트
│
├── dict/
│   ├── tech_stack_dictionary.json   범주별 기술스택 표준 토큰
│   └── synonyms.json                동의어 → 표준 표현 매핑
│
├── data/                            (git 미추적)
│   ├── raw/                         점핏·원티드·LinkedIn·뉴스 원본
│   └── processed/                   전처리 결과 (kr_jobs_clean.* 등)
│
├── outputs/
│   ├── figures/                     시각화 PNG
│   └── models/                      학습된 모델 파일
│
├── docs/project_plan.md
├── requirements.txt
└── .gitignore
```

---

## 연구 질문 (개정판)

| # | 연구 질문 | 데이터 | 담당 노트북 |
|---|---|---|---|
| **RQ1** | 현재 국내 IT 공고가 가장 많이 요구하는 역량은? | 점핏·원티드 | 04 |
| **RQ2** | 글로벌 IT 공고의 요구역량·AI 역량은 어떻게 나타나는가? | LinkedIn | 04·05 |
| **RQ3** | 국내 뉴스의 개발자 채용 담론은 2021–2026 어떻게 변했나? | 뉴스 | 04 |
| **RQ4** | **언론 담론과 실제 요구역량은 일치하는가? (핵심)** | 뉴스 × 국내공고 | **07** |
| └ RQ4a | (기술스택 간극) 뉴스·공고의 기술/역량 용어 비교 시 불일치는? | 뉴스 × 국내공고 | 07 |
| └ RQ4b | (담론 프레임) 뉴스 내 위기/대체/재편 프레임 강도는? | 뉴스 | 07 |
| **RQ5** | **국내 vs 글로벌 요구역량 격차(lag)는 얼마나 되는가?** | 국내공고 × LinkedIn | 07 |
| RQ6 | (보조) 공고 텍스트로 직무군을 분류할 수 있는가? | 국내공고/LinkedIn | 06 |

> **RQ4 2트랙 분리:** 프레임어(AI대체·일자리위기)와 기술토큰(SQL·Git)을 한 지표에 섞지 않는다. 기술스택 간극(RQ4a)과 담론 프레임 강도(RQ4b)를 따로 측정.

---

## 데이터 역할 분리 (설계 핵심)

```
뉴스(빅카인즈)   = 2021~2026 국내 담론 시계열  → before/after ChatGPT 비교 가능
점핏·원티드      = 2026 현재 국내 실제 공고     → 국내 현실 스냅샷 (시계열·종료공고 불가)
LinkedIn        = 2023~2024 글로벌 채용 단면   → "AI 이후" 글로벌 스냅샷 (과거 포함)

핵심 비교 ① [담론 vs 현실]   국내 간극   → RQ4 (기술스택 간극 + 담론 프레임)
핵심 비교 ② [국내 vs 글로벌]  수용 격차   → RQ5 (lag)
```

> **주의:** 한국어(국내공고/뉴스)↔영어(LinkedIn) 토픽 결과는 직접 비교하지 않는다. 비교 단위는 기술스택 사전 토큰(React, Python, AWS 등 언어 무관)의 **빈도/비율**뿐이다.
> **국내 공고는 현재 활성 스냅샷**이라 종료/과거 공고·국내 시계열은 불가 → "변화" 서사는 뉴스·LinkedIn이 담당.

---

## 방법론 (가산점 ② — 수업 외 포함)

| 방법 | 목적 | 수업 범위 |
|---|---|---|
| 빈도 분석 / TF-IDF | 분포·직무군 변별 키워드 | 수업 내 |
| WordCloud | 발표용 직관 시각화 | 보조 |
| **Co-occurrence Network** | 기술스택 동시출현 관계 | **수업 외** |
| **Word2Vec** | 공기어 분석 | **수업 외·선택** |
| **BERTopic** | Sentence-BERT 임베딩 토픽 군집 | **수업 외** |
| **UMAP** | 임베딩 2D 축소 → 직무군 분리 | **수업 외** |
| **간극지수(Spearman·순위차)** | 담론 vs 현실 정량 비교 (RQ4a) | **수업 외** |

---

## 데이터 출처 & 수집 현황

| 데이터 | 출처 | 수집 현황 |
|---|---|---|
| 점핏 채용공고 | `https://api.jumpit.co.kr/api/positions` (공개 JSON) | **777건** (techStacks 100%, 본문 99%) |
| 원티드 채용공고 | `https://www.wanted.co.kr/api/v4/jobs` (개발 17개 카테고리) | **2,534건** (skill_tags 50%·본문 100%) |
| → 국내 통합 | (교차중복 제거) | **3,274건**, 스킬 보유 97% |
| LinkedIn 2023–24 | [Kaggle: arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | 123,849 → IT 필터 예정 |
| 빅카인즈 뉴스 | [bigkinds.or.kr](https://www.bigkinds.or.kr/) | 2021–26, 본문 200자·2만건/쿼리 제한 |

> **수집 방식:** 점핏·원티드의 공개 JSON 엔드포인트 호출 + 직접 크롤링 → "API 활용 + 직접 수집"(가산점). 학술용·비상업·요청 간 지연.

---

## 설치 및 실행

```bash
git clone https://github.com/composedly13/ai-era-it-hiring-analysis.git
cd ai-era-it-hiring-analysis

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

---

## 데이터 배치 방법

데이터 파일은 git에서 관리하지 않습니다 (`.gitignore`). 팀 공유본을 아래 경로에 직접 배치하세요.

### 국내 공고 (점핏·원티드)
팀 카톡 공유본 zip 압축 해제 후:
```
data/raw/domestic/jumpit.jsonl
data/raw/domestic/wanted.jsonl
data/processed/domestic/kr_jobs_clean.csv
data/processed/domestic/kr_jobs_clean.jsonl
```
> 직접 재수집하려면: `python notebooks/00_domestic_collection.py`
> raw 재사용(재크롤 없이 재처리)만 하려면: `SKIP_COLLECT=1 python notebooks/00_domestic_collection.py`

### 뉴스 (빅카인즈)
빅카인즈(bigkinds.or.kr)에서 키워드·기간별 분할 내보내기 후:
```
data/raw/news/news_<기간>.xlsx   (예: news_2021_2022.xlsx)
```
> 수집 쿼리 계획: `python notebooks/01_news_collection_preprocessing.py` 실행 시 출력

### LinkedIn
Kaggle에서 다운로드 후:
```bash
kaggle datasets download arshkon/linkedin-job-postings -p data/raw/linkedin --unzip
```

### 전처리 결과 경로 요약
| 파일 | 경로 | 생성 스크립트 |
|---|---|---|
| 국내 공고 | `data/processed/domestic/kr_jobs_clean.csv` / `.jsonl` | 00 또는 카톡 공유본 |
| 뉴스 | `data/processed/news/news_processed.csv` | 01 |
| LinkedIn | `data/processed/linkedin/linkedin_processed.csv` | 02 |

---

## 역할 분담

| 팀원 | 주 담당 | 핵심 스크립트 |
|---|---|---|
| **A** | 국내공고(점핏·원티드) 수집·통합 + 기술스택 사전·간극지수(RQ4a) | 00 · 07 |
| **B** | 빅카인즈 뉴스 수집 + 담론 프레임 지수(RQ4b) | 01 |
| **C** | LinkedIn 수집·IT필터 + BERTopic/UMAP + 격차(RQ5) | 02 · 05 |
| **D** | 통합·직무군 분류(보조) · 시각화 · 논문/발표 총괄 | 06 · 08 |

> RQ4(담론 vs 현실)는 A·B·D 공동. 기술스택 사전은 전원 기여.
> 국내 소스는 **점핏·원티드 공개 JSON 크롤링**으로 수집한다.

---

## 일정 (2~3주)

| 단계 | 작업 |
|---|---|
| 0. 셋업·검증 | 점핏·원티드 수집 검증(완료), Kaggle 다운로드, 빅카인즈 export |
| 1. 수집 | 국내공고(완료)·뉴스·LinkedIn 수집 |
| 2. 전처리·사전 | 통합·정규화(완료), 사전 precision/recall 검증 |
| 3. 탐색·시계열 | 빈도/TF-IDF/Co-occurrence, 뉴스 시계열 |
| 4. 고급분석·모델링 | BERTopic/UMAP, 직무군 분류(보조) |
| 5. 비교·인사이트 | RQ4 간극지수 + RQ5 격차 |
| 6. 산출물 | 논문·발표·zip 제출 |

---

## 핵심 메시지

> 생성형 AI 시대의 IT 채용시장은 단순히 AI가 개발자를 대체하는 방향으로만 변하지 않는다.
> 뉴스 담론은 위기와 일자리 재편을 강조하지만, 실제 채용공고는 여전히 기본 개발역량과 실무 기술스택을 핵심으로 요구한다.
> 다만 AI·데이터·클라우드 역량이 기존 개발역량과 **결합**되는 흐름이 나타나며 — 이 흐름은 글로벌이 국내보다 앞서 있다 — 앞으로의 개발자는 단일 기술 전문가가 아니라 **AI 활용 능력과 실무 개발역량을 함께 갖춘 문제 해결형 인재**로 성장할 필요가 있다.
