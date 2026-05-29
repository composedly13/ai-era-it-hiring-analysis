# notebooks/

파이프라인 실행 스크립트 디렉터리.  
**00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 순서로 실행한다.**  
각 스크립트는 `python notebooks/<파일명>.py`로 단독 실행 가능하다.

---

## 브랜치 전략

```
develop          ← 통합 브랜치 (기본 브랜치)
├── feature/worknet    B 담당 — 00 작업
├── feature/news       C 담당 — 01 작업
└── feature/linkedin   D 담당 — 02 작업
```

**작업 흐름:**
1. 각자 `feature/<데이터소스>` 브랜치를 `develop`에서 분기
2. 수집 + 전처리 완료 후 `develop`으로 PR 생성
3. 팀원 1인 이상 리뷰 후 머지
4. 03번부터는 `develop`에 세 브랜치가 모두 머지된 시점부터 진행

```bash
# 각자 브랜치 생성 예시
git checkout develop
git checkout -b feature/worknet   # B
git checkout -b feature/news      # C
git checkout -b feature/linkedin  # D
```

---

## 실행 순서 및 명세

### 00_worknet_collection_preprocessing.py
**담당: B** | 브랜치: `feature/worknet`  
입력: 워크넷 API | 출력: `data/processed/worknet/worknet_processed.csv`

워크넷 채용정보 API 수집 → 채용정보URL 크롤링(직무내용 본문) → KoNLPy 전처리 → 기술스택 추출

> **1일차 필수:** `채용정보URL` 1건을 직접 열어 직무내용이 텍스트인지 이미지인지 확인할 것.  
> 이미지 게시라면 본문 크롤링을 포기하고 채용제목 기반 분석으로 전환한다.

```
환경변수: WORKNET_API_KEY=<공공데이터포털 발급 키>
```

---

### 01_news_collection_preprocessing.py
**담당: C** | 브랜치: `feature/news`  
입력: 빅카인즈 내보내기 파일 | 출력: `data/processed/news/news_processed.csv`

빅카인즈 분할 CSV 병합 → before/after 라벨 부착(ChatGPT 출시 2022-11-01 기준) → KoNLPy 전처리 → 기술스택 추출

> 본문 200자 제한·다운로드 20,000건/쿼리 제한으로 인해 빅카인즈 UI 수집이 선행되어야 한다.  
> 스크립트 실행 전 `data/raw/news/` 에 내보내기 파일을 배치할 것.  
> 스크립트 상단 `step1_print_query_plan()`이 키워드·기간별 분할 쿼리 계획을 출력한다.

---

### 02_linkedin_collection_preprocessing.py
**담당: D** | 브랜치: `feature/linkedin`  
입력: Kaggle LinkedIn 데이터셋 | 출력: `data/processed/linkedin/linkedin_processed.csv`

postings + skills + companies + industries 병합 → IT 직무 필터링 → 영어 전처리(lemmatization) → 기술스택 추출

> Kaggle CLI로 `data/raw/linkedin/`에 먼저 다운로드해야 한다.
> ```bash
> kaggle datasets download arshkon/linkedin-job-postings -p data/raw/linkedin --unzip
> ```
> LinkedIn 데이터는 2022.11 이후만 포함 — **AI 이전/이후 비교 불가, 글로벌 단면으로만 해석.**

---

### 03_tech_dictionary_validation.py
**담당: 미정** | 선행 조건: 00·01·02 PR 머지 완료  
입력: 세 processed 파일 | 출력: `dict/` 파일 업데이트

3개 데이터소스의 실제 어휘에서 기술스택 사전 누락 토큰을 발굴하고 보완한다.

> 사전 품질이 RQ4(담론 vs 현실) 비교의 타당성을 직접 결정한다 (Validity Guardrail #2).  
> 변경 후 반드시 `git commit`으로 이력을 남긴다.

---

### 04_eda_keyword_analysis.py
**담당: 미정** | 선행 조건: 03 완료 | 출력: `outputs/figures/`

| 생성 파일 | 내용 | 연결 RQ |
|---|---|---|
| `worknet_top_skills.png` | 국내 상위 기술스택 막대 | RQ1 |
| `linkedin_top_skills.png` | 글로벌 상위 기술스택 막대 | RQ2 |
| `news_ai_trend.png` | AI 키워드 월별 언급률 시계열 | RQ3 |
| `wc_worknet.png` | 워크넷 WordCloud | RQ1 |
| `wc_news_before/after.png` | 뉴스 before/after WordCloud | RQ3 |

---

### 05_topic_modeling_bertopic.py
**담당: 미정** | 선행 조건: 03 완료 | 수업 외 방법론 (가산점 ②)  
출력: `outputs/figures/`, `outputs/models/`

| 생성 파일 | 내용 | 방법론 |
|---|---|---|
| `bertopic_topics.html` | 토픽 분포 인터랙티브 차트 | BERTopic |
| `word2vec.model` | 학습된 Word2Vec 모델 | Word2Vec |
| `cooccurrence_network.png` | 기술스택 동시출현 그래프 | Co-occurrence Network |
| `umap_clusters.png` | 직무군 임베딩 2D 산점도 | UMAP + Sentence-BERT |

---

### 06_classification_model.py
**담당: 미정** | 선행 조건: 03 완료  
입력: processed 파일 | 출력: `outputs/figures/confusion_matrix.png`

직무군 분류 (주 모델, RQ5): TF-IDF + LR / SVM 학습 및 Accuracy·F1·Confusion Matrix 평가

> **순환논리 회피 원칙:** 라벨은 워크넷 직종코드 / LinkedIn job title 기반.  
> AI 관련 공고 분류는 키워드 약지도(보조)로만 사용하며 논문에 한계를 명시한다.

---

### 07_rq4_discourse_vs_reality.py
**담당: 미정** | 선행 조건: 03 완료 | **본 연구의 핵심**  
출력: `outputs/figures/rq4_*.png`

| 생성 파일 | 내용 |
|---|---|
| `rq4_discourse_vs_reality.png` | 뉴스 담론 vs 워크넷 요구역량 나란히 비교 |
| `rq4_domestic_vs_global.png` | 국내(워크넷) vs 글로벌(LinkedIn) 기술스택 |

> 비교 단위: 기술스택 사전 토큰 (언어 무관).  
> 동일 시점(2026년 뉴스 vs 2026년 워크넷)으로 제한한다.

---

### 08_visualization_summary.py
**담당: 미정** | 모든 `outputs/figures/` 파일 생성 여부를 체크리스트로 출력한다.

```bash
python notebooks/08_visualization_summary.py
```

---

## 스크립트 간 데이터 흐름

```
feature/worknet  ──► 00 (B) ──┐
feature/news     ──► 01 (C) ──┼──► develop 머지 ──► 03 ──► 04, 05, 06, 07 ──► 08
feature/linkedin ──► 02 (D) ──┘                        ↑
                                                   dict/ 업데이트
```

---

## 공통 실행 방법

```bash
# 프로젝트 루트에서 실행 (sys.path가 루트를 참조)
cd ai-era-it-hiring-analysis
python notebooks/00_worknet_collection_preprocessing.py
python notebooks/01_news_collection_preprocessing.py
python notebooks/02_linkedin_collection_preprocessing.py
python notebooks/03_tech_dictionary_validation.py
python notebooks/04_eda_keyword_analysis.py
python notebooks/05_topic_modeling_bertopic.py
python notebooks/06_classification_model.py
python notebooks/07_rq4_discourse_vs_reality.py
python notebooks/08_visualization_summary.py
```
