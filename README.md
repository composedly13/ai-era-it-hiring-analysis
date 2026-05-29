# 생성형 AI 시대 IT 채용시장의 요구역량과 채용 담론 변화 분석

> **한 줄 요약:** 언론은 "AI가 개발자를 대체한다"고 말하지만, 실제 채용공고는 무엇을 요구하는가?  
> 뉴스 담론(국내 장기 시계열) · LinkedIn(글로벌 단면) · 워크넷(국내 현재 현실) 세 데이터를 결합해 **담론과 현실의 간극**을 데이터로 검증한다.

| 항목 | 내용 |
|---|---|
| 과목 | 텍스트마이닝 |
| 팀 구성 | 4명 |
| 제출 마감 | 2026-06-15 23:59 |
| 발표 | 2026-06-16 11:00 |

---

## 디렉터리 구조

```
ai-era-it-hiring-analysis/
├── notebooks/                        # 파이프라인 실행 스크립트 (순서대로)
│   ├── README.md                                   ← 각 스크립트 상세 명세
│   ├── 00_worknet_collection_preprocessing.py      담당: B
│   ├── 01_news_collection_preprocessing.py         담당: C
│   ├── 02_linkedin_collection_preprocessing.py     담당: D
│   ├── 03_tech_dictionary_validation.py            담당: 전원
│   ├── 04_eda_keyword_analysis.py                  담당: 전원
│   ├── 05_topic_modeling_bertopic.py               담당: 전원
│   ├── 06_classification_model.py                  담당: 전원
│   ├── 07_rq4_discourse_vs_reality.py              ← RQ4 핵심 (전원)
│   └── 08_visualization_summary.py                 담당: 전원
│
├── src/                              # 재사용 가능한 Python 모듈
│   ├── collection/
│   │   ├── worknet_api.py            워크넷 채용정보 API 수집
│   │   ├── worknet_crawler.py        채용정보URL 크롤링 (직무내용 본문)
│   │   ├── bigkinds_news.py          빅카인즈 CSV 로더
│   │   └── linkedin_loader.py        Kaggle 데이터셋 로더 + IT 필터
│   ├── preprocessing/
│   │   ├── korean_preprocessor.py   KoNLPy 형태소 분석
│   │   ├── english_preprocessor.py  lemmatization · stopwords
│   │   └── tech_dictionary.py       기술스택 사전 + 동의어 표준화
│   ├── analysis/
│   │   ├── frequency_analysis.py
│   │   ├── tfidf_analysis.py
│   │   ├── cooccurrence_network.py  Co-occurrence Network (수업 외)
│   │   ├── word2vec_analysis.py     Word2Vec 공기어 (수업 외)
│   │   └── time_series.py           시계열 (before/after ChatGPT)
│   ├── modeling/
│   │   ├── bertopic_model.py        BERTopic (수업 외)
│   │   ├── job_classifier.py        직무군 분류 (주 모델)
│   │   └── evaluation.py            Accuracy · F1 · Confusion Matrix
│   └── visualization/
│       ├── wordcloud_gen.py
│       ├── network_plot.py
│       ├── umap_plot.py             UMAP 2D 시각화 (수업 외)
│       └── comparison_chart.py      담론 vs 현실 비교 차트 (RQ4)
│
├── dict/
│   ├── tech_stack_dictionary.json   범주별 기술스택 표준 토큰
│   └── synonyms.json                동의어 → 표준 표현 매핑
│
├── data/
│   ├── raw/                         원본 데이터 (git 미추적)
│   │   ├── worknet/
│   │   ├── linkedin/
│   │   └── news/
│   └── processed/                   전처리 결과 (git 미추적)
│
├── outputs/
│   ├── figures/                     시각화 PNG
│   └── models/                      학습된 모델 파일
│
├── docs/
│   └── project_plan.md
│
├── requirements.txt
└── .gitignore
```

---

## 연구 질문

| # | 연구 질문 | 데이터 | 담당 노트북 |
|---|---|---|---|
| **RQ1** | 현재 국내 IT 채용공고에서 가장 많이 요구되는 역량은? | 워크넷 | 04 |
| **RQ2** | 글로벌 IT 채용공고에서 AI 관련 역량 요구는 어떻게 나타나는가? | LinkedIn | 04 |
| **RQ3** | 국내 뉴스에서 IT 채용·개발자 담론은 2021–2026 어떻게 변했는가? | 뉴스 | 04 |
| **RQ4** | **언론 담론과 실제 채용공고의 요구역량은 일치하는가?** | 뉴스 × 워크넷 | **07** |
| **RQ5** | 채용공고 텍스트로 직무군을 분류할 수 있는가? | 워크넷/LinkedIn | 06 |

---

## 데이터 역할 분리 (설계 핵심)

```
뉴스(빅카인즈) = 2021~2026 국내 담론 시계열  → before/after ChatGPT 비교 가능
LinkedIn      = 2023~2024 글로벌 채용 단면   → "AI 이후" 글로벌 스냅샷
워크넷 API    = 2026 현재 국내 실제 공고      → 국내 현실 스냅샷
최종 비교      = 담론(뉴스) vs 글로벌(LinkedIn) vs 국내현실(워크넷)
```

> **주의:** 한국어(워크넷/뉴스)↔영어(LinkedIn) 토픽 결과는 직접 비교하지 않는다.  
> 비교 단위는 기술스택 사전 토큰(React, Python, AWS 등 언어 무관)의 **빈도/비율**뿐이다.

---

## 방법론 (가산점 ② — 수업 외 포함)

| 방법 | 목적 | 수업 범위 |
|---|---|---|
| 빈도 분석 | 전체 분포 파악 | 수업 내 |
| TF-IDF | 직무군·연도·데이터셋별 변별 키워드 | 수업 내 |
| WordCloud | 발표용 직관 시각화 | 보조 |
| **Co-occurrence Network** | 기술스택 동시출현 관계 | **수업 외** |
| **Word2Vec** | AI/LLM/신입 등 공기어 분석 | **수업 외** |
| **BERTopic** | Sentence-BERT 임베딩 기반 토픽 군집 | **수업 외** |
| **UMAP** | 임베딩 2D 축소 → 직무군 분리 시각화 | **수업 외** |

---

## 데이터 출처

| 데이터 | 출처 | 비고 |
|---|---|---|
| 워크넷 채용정보 API | [공공데이터포털 #3038225](https://www.data.go.kr/data/3038225/openapi.do) | 이용허락 제4유형 (출처표시·비상업적·변경금지) |
| LinkedIn Job Postings 2023–24 | [Kaggle: arshkon/linkedin-job-postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) | 무료 |
| 빅카인즈 뉴스 | [bigkinds.or.kr](https://www.bigkinds.or.kr/) | 본문 200자 제한, 다운로드 20,000건/쿼리 |

---

## 설치 및 실행

```bash
# 1. 저장소 클론
git clone https://github.com/composedly13/ai-era-it-hiring-analysis.git
cd ai-era-it-hiring-analysis

# 2. 가상환경 생성 및 의존성 설치
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# 3. API 키 설정
# .env 파일 생성 (git 미추적)
# WORKNET_API_KEY=your_key_here

# 4. 스크립트 순서대로 실행 (프로젝트 루트에서)
python notebooks/00_worknet_collection_preprocessing.py
python notebooks/01_news_collection_preprocessing.py
python notebooks/02_linkedin_collection_preprocessing.py
# ... (notebooks/README.md 참조)
```

> **데이터 파일은 git에서 관리하지 않습니다.**  
> `data/raw/` 하위 디렉터리에 직접 배치하세요.

---

## 역할 분담

| 팀원 | 주 담당 | 브랜치 | 핵심 스크립트 |
|---|---|---|---|
| **A** | 기획서 재작성 · 논문/보고서 총괄 | `develop` | — |
| **B** | 워크넷 수집 + 전처리 | `feature/worknet` | 00 |
| **C** | 빅카인즈 뉴스 수집 + 전처리 | `feature/news` | 01 |
| **D** | LinkedIn 수집 + 전처리 | `feature/linkedin` | 02 |

> B·C·D는 각자 브랜치에서 작업 후 `develop`으로 PR → 머지 완료 시점에 03~08 담당 분배 예정  
> **03~08: 담당 미정**

---

## 일정

| 단계 | 기간 | 작업 |
|---|---|---|
| 0. 셋업·검증 | 1주차 초 | API 신청, 워크넷 텍스트 유무 1일차 검증, Kaggle 다운로드 |
| 1. 수집 | 1주차 | 워크넷·뉴스·LinkedIn 수집 |
| 2. 전처리·사전 | 1주차 말~2주차 초 | 언어별 전처리, 기술스택 사전 |
| 3. 탐색·시계열 | 2주차 | 빈도/TF-IDF/WordCloud, Co-occurrence |
| 4. 고급분석·모델링 | 2주차 말~3주차 초 | BERTopic/Word2Vec/UMAP, 직무군 분류 |
| 5. 비교·인사이트 | 3주차 | RQ4 담론 vs 현실 비교 |
| 6. 산출물 | 3주차 말 | 논문 초안, 발표자료, zip 제출 |

---

## 핵심 메시지

> 생성형 AI 시대의 IT 채용시장은 단순히 AI가 개발자를 대체하는 방향으로만 변하지 않는다.
> 뉴스 담론은 위기와 일자리 재편을 강조하지만, 실제 채용공고는 여전히 기본 개발역량과 실무 기술스택을 핵심으로 요구한다.
> 앞으로의 개발자는 단일 기술 전문가가 아니라 **AI 활용 능력과 실무 개발역량을 함께 갖춘 문제 해결형 인재**로 성장할 필요가 있다.
