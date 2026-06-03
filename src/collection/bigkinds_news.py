"""
빅카인즈 뉴스 데이터 로더 (본문 크롤링 통합본 기반)

입력: data/raw/news/all_news_with_body.xlsx
  - 56,963건 (본문 크롤링 성공 53,479건 / 93.9%)
  - 2021-01-01 ~ 2026-05-29
  - 빅카인즈 export(메타) + URL 원문 크롤링(본문) 결합

수집 키워드는 빅카인즈 UI에서 쓴 것을 참고용으로 보존.
"""

import pandas as pd
from pathlib import Path


# 빅카인즈 수집 시 사용한 키워드 (참고)
NEWS_KEYWORDS = [
    "개발자 채용", "IT 채용", "개발자 취업",
    "주니어 개발자", "신입 개발자",
    "AI 개발자", "AI 인재", "생성형 AI 채용",
    "ChatGPT 개발자", "개발자 감축", "개발자 연봉",
    "소프트웨어 인재", "디지털 인재", "클라우드 인재",
]

# ChatGPT 출시일 (담론 단절점)
CHATGPT_LAUNCH = pd.Timestamp("2022-11-30")

DEFAULT_PATH = "data/raw/news/all_news_with_body.xlsx"


def load_news(path: str = DEFAULT_PATH, ok_only: bool = False) -> pd.DataFrame:
    """본문 크롤링 통합본 xlsx를 로드해 표준 스키마로 반환.

    Parameters
    ----------
    path : str
        all_news_with_body.xlsx 경로 (repo 루트 기준 상대경로)
    ok_only : bool
        True면 본문 크롤링 성공(`body_status == "ok"`) 행만 반환.
        False면 전체 — 메타 기반 분석(기사 수·언론사·발행일)에 사용.

    Returns
    -------
    pd.DataFrame
        표준 컬럼:
          - news_id (str): 빅카인즈 뉴스 식별자
          - date (datetime64): 발행일
          - media (str): 언론사
          - title (str): 제목
          - url (str): 원문 URL
          - body (str): 크롤링 본문 (실패 시 "")
          - body_len (int): 본문 자수
          - body_status (str): "ok" / "no_url" / "HTTP_xxx" / "no_extract" / "err_*"
          - period (str): "before" / "after" (ChatGPT 2022-11-30 기준)
          - year (int), month (int): 시계열 집계용
    """
    df = pd.read_excel(path, engine="openpyxl")
    df = df.rename(columns={
        "뉴스 식별자": "news_id",
        "발행일": "date",
        "언론사": "media",
        "제목": "title",
        "URL": "url",
        "본문_원문": "body",
        "본문_원문_길이": "body_len",
        "본문_원문_상태": "body_status",
    })
    df["news_id"] = df["news_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df["body_len"] = df["body_len"].fillna(0).astype(int)
    df["body_status"] = df["body_status"].fillna("missing")
    df["period"] = (df["date"] >= CHATGPT_LAUNCH).map({True: "after", False: "before"})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)

    if ok_only:
        df = df[df["body_status"] == "ok"].reset_index(drop=True)
    return df


def analysis_text(row: pd.Series) -> str:
    """분석용 텍스트 — 본문 성공 시 제목+본문, 실패 시 제목만.

    크롤 실패 행도 메타 분석에서는 빠뜨리지 않도록 제목은 항상 포함.
    """
    title = row.get("title") or ""
    body = row.get("body") or ""
    if row.get("body_status") == "ok" and body:
        return f"{title}\n{body}"
    return title
