"""
LinkedIn Job Postings 2023-2024 (Kaggle: arshkon/linkedin-job-postings) 로더.

postings.csv (123,849건) + job_skills.csv + companies/company_industries.csv +
mappings/{skills,industries}.csv 병합.

주의:
- 데이터셋명은 '2023-2024'지만 listed_time은 거의 전부 2024-04 단면.
  original_listed_time이 2023-12 ~ 2024-04 범위라 시계열 분석은 4개월에 한정.
- skill_abr 매핑(skills.csv)은 'Engineering/IT/Marketing' 같은 직무 분야 라벨 →
  실제 기술스택(Python·React·Cursor 등)은 description 본문에서 별도 추출 필요.
- IT 직무 필터는 title 광베이스 정규식(코어 IT + QA·Solutions Eng·Tech PM·Architect 등).
"""

import pandas as pd
from pathlib import Path

# IT 직무 광베이스 title 패턴 (소문자 매칭, 정규식)
IT_TITLE_PATTERN = "|".join([
    # core SW engineering
    "software engineer", "software developer", "software architect",
    "staff engineer", "principal engineer", "lead engineer",
    "frontend", "front.?end", "backend", "back.?end",
    "fullstack", "full.?stack",
    # data
    "data scientist", "data analyst", "data engineer", "analytics engineer",
    "data infrastructure", "data platform", "data architect",
    "business intelligence", "bi engineer",
    # ML/AI
    "machine learning", "ml engineer", "ml ops", "mlops",
    "ai engineer", "ai researcher", "ai scientist",
    "computer vision", "nlp engineer", "research scientist",
    "applied scientist", "prompt engineer",
    # DevOps / SRE / Infra
    "devops", "sre", "site reliability", "cloud engineer", "platform engineer",
    "infrastructure engineer", "systems engineer", "network engineer",
    "kubernetes engineer", "linux engineer",
    # Security
    "security engineer", "application security", "infosec", "appsec",
    "security architect", "cybersecurity",
    # Mobile / Web / Embedded
    "web developer", "mobile developer", "ios developer", "android developer",
    "react native", "embedded", "firmware",
    # DBA
    "database engineer", "database administrator", r"\bdba\b",
    # QA
    "qa engineer", "qe engineer", r"\bsdet\b", "test engineer",
    "quality engineer", "automation engineer",
    # 보조 IT
    "solutions engineer", "sales engineer", "technical program manager",
    "developer relations", "developer advocate",
])


def load_postings(data_dir: str = "data/raw/linkedin") -> pd.DataFrame:
    """postings.csv 로드 및 epoch → datetime 변환."""
    p = Path(data_dir) / "postings.csv"
    df = pd.read_csv(p, low_memory=False)
    for col in ("listed_time", "original_listed_time"):
        if col in df.columns:
            df[col + "_dt"] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    return df


def merge_skills(postings: pd.DataFrame, data_dir: str = "data/raw/linkedin") -> pd.DataFrame:
    """job_skills (skill_abr) + mappings/skills.csv 조인.

    skill_abr (3-letter)와 skill_name 모두 보존. **분야 라벨**이지 기술스택 아님.
    실제 기술스택은 description에서 별도 추출.
    """
    js = pd.read_csv(Path(data_dir) / "jobs" / "job_skills.csv")
    sm = pd.read_csv(Path(data_dir) / "mappings" / "skills.csv")
    js_named = js.merge(sm, on="skill_abr", how="left")
    agg = js_named.groupby("job_id").agg(
        skill_abrs=("skill_abr", lambda s: "|".join(sorted(set(s.dropna())))),
        skill_names=("skill_name", lambda s: "|".join(sorted(set(s.dropna())))),
    ).reset_index()
    return postings.merge(agg, on="job_id", how="left")


def merge_companies(df: pd.DataFrame, data_dir: str = "data/raw/linkedin") -> pd.DataFrame:
    """company_industries.csv를 조인. 회사 산업명을 industry 컬럼으로 추가."""
    ci = pd.read_csv(Path(data_dir) / "companies" / "company_industries.csv")
    # 한 회사가 복수 산업 가능 → 첫번째만 사용 (간소화). 추후 필요 시 list로 변경.
    ci_first = ci.drop_duplicates(subset="company_id", keep="first")
    return df.merge(ci_first, on="company_id", how="left")


def filter_it_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """title 정규식 광베이스로 IT 직무 필터. 회사 산업·skill_abr는 분석용으로만 보존."""
    title_low = df["title"].fillna("").str.lower()
    mask = title_low.str.contains(IT_TITLE_PATTERN, regex=True, na=False)
    return df[mask].copy()
