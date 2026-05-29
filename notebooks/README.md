# notebooks/

파이프라인 실행 스크립트. **00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08** 순서.
각 스크립트는 프로젝트 루트에서 `python notebooks/<파일명>.py`로 실행.

---

## 브랜치 전략

```
develop                  ← 통합 브랜치
├── feature/domestic     A 담당 — 00 (점핏·원티드)
├── feature/news         B 담당 — 01 (빅카인즈)
└── feature/linkedin     C 담당 — 02 (Kaggle)
```

1. 각자 `feature/*` 브랜치를 `develop`에서 분기
2. 수집·전처리 완료 후 `develop`으로 PR → 리뷰 → 머지
3. 03번부터는 세 소스가 모두 머지된 시점에 진행

---

## 실행 순서 및 명세

### 00_domestic_collection.py — 담당 A · `feature/domestic`
입력: 점핏·원티드 공개 JSON | 출력: `data/processed/kr_jobs_clean.csv` / `.jsonl`

점핏(`techStacks` 구조화) + 원티드(`skill_tags` + 본문) 수집 → 공통 스키마 병합 → 스킬 정규화/본문 추출 → 교차중복 제거.
- **API 키 불필요** (공개 JSON 호출 + 직접 크롤링 = 가산점)
- 재크롤 생략: `SKIP_COLLECT=1 python notebooks/00_domestic_collection.py` (raw 재사용)
- 현재 수집: 점핏 777 + 원티드 2,534 → 통합 3,274건

### 01_news_collection_preprocessing.py — 담당 B · `feature/news`
입력: 빅카인즈 내보내기 CSV | 출력: `data/processed/news_processed.csv`

분할 CSV 병합 → before/after 라벨(ChatGPT 2022-11-01 기준) → KoNLPy 전처리 → 기술스택/프레임 추출.
> 본문 200자·다운로드 2만건/쿼리 제한 → UI 수집 선행, `data/raw/news/`에 배치.

### 02_linkedin_collection_preprocessing.py — 담당 C · `feature/linkedin`
입력: Kaggle LinkedIn 데이터셋 | 출력: `data/processed/linkedin_processed.csv`

postings+skills+companies+industries 병합 → IT 직무 필터 → 영어 전처리 → 기술스택 추출.
> ```bash
> kaggle datasets download arshkon/linkedin-job-postings -p data/raw/linkedin --unzip
> ```
> 2022.11 이후만 포함 — AI 이전/이후 비교 불가, 글로벌 단면으로만 해석.

### 03_tech_dictionary_validation.py — 전원 · 선행: 00·01·02 머지
세 소스 실제 어휘에서 사전 누락 토큰 발굴 + **100건 수동 라벨 precision/recall ≥0.9 검증** (Validity Guardrail #5).

### 04_eda_keyword_analysis.py — 전원 · 출력 `outputs/figures/`
| 파일 | 내용 | RQ |
|---|---|---|
| `kr_top_skills.png` | 국내 상위 기술스택 | RQ1 |
| `linkedin_top_skills.png` | 글로벌 상위 기술스택 | RQ2 |
| `news_ai_trend.png` | AI 키워드 월별 언급률 | RQ3 |
| `cooccurrence_network.png` | 기술스택 동시출현(수업 외) | RQ1 |

### 05_topic_modeling_bertopic.py — 전원 · 수업 외(가산점 ②)
BERTopic(LinkedIn description) + UMAP 2D + (선택)Word2Vec → `outputs/figures/`, `outputs/models/`.

### 06_classification_model.py — 담당 D · RQ6(보조)
직무군 분류: TF-IDF/SBERT + LR·SVM → Accuracy·F1·Confusion Matrix.
> **라벨 누수 차단:** 라벨 출처(점핏 `jobCategories`/원티드 `categoryTags`)는 **피처에서 제외**, 본문만 입력. AI 분류는 키워드 약지도(한계 명시).

### 07_rq4_discourse_vs_reality.py — A·B·D 공동 · **핵심**
| 파일 | 내용 |
|---|---|
| `rq4_gap_index.png` + `outputs/rq4_gap_table.csv` | RQ4a 담론 vs 현실 간극지수(Spearman·순위차 사분면) |
| `rq5_domestic_vs_global.png` | RQ5 국내(점핏·원티드) vs 글로벌(LinkedIn) |
> 비교 단위 = 기술스택 사전 토큰(언어 무관). 동일 시점(2026 뉴스 vs 2026 국내공고). 프레임 분석(RQ4b)은 time_series/08.

### 08_visualization_summary.py — 담당 D
`outputs/figures/` 생성 체크리스트 + 발표용 종합 차트.

---

## 데이터 흐름

```
feature/domestic ──► 00 (A) ──┐
feature/news     ──► 01 (B) ──┼──► develop 머지 ──► 03 ──► 04·05·06·07 ──► 08
feature/linkedin ──► 02 (C) ──┘                        ↑ dict/ 검증·업데이트
```

## 공통 실행

```bash
cd ai-era-it-hiring-analysis
python notebooks/00_domestic_collection.py
python notebooks/01_news_collection_preprocessing.py
python notebooks/02_linkedin_collection_preprocessing.py
python notebooks/03_tech_dictionary_validation.py
python notebooks/04_eda_keyword_analysis.py
python notebooks/05_topic_modeling_bertopic.py
python notebooks/06_classification_model.py
python notebooks/07_rq4_discourse_vs_reality.py
python notebooks/08_visualization_summary.py
```
