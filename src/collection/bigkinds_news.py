"""
빅카인즈(BIGKinds) 뉴스 데이터 로더
빅카인즈 다운로드 CSV/XLSX를 읽어 전처리 파이프라인 입력 형태로 변환한다.
본문은 최대 200자(저작권 제한), 다운로드 최대 20,000건/쿼리.
수집 키워드·기간을 분할해 복수 파일로 관리할 것.
"""

import glob
import pandas as pd
from pathlib import Path


# 검색 키워드 — IT·개발자 채용 담론에 직결되는 것만.
# 광범위 '인재' 계열("AI 인재"·"소프트웨어/디지털/클라우드 인재")은 정부 인재양성·교육·
# 비IT 기사를 과다 유입시켜 분모를 오염시키므로 의도적으로 제외.
# ⚠ 이 세트는 전 기간 고정 — 중간에 바꾸면 RQ3 월별 언급률의 분모 일관성이 깨짐.
NEWS_KEYWORDS = [
    # 개발자·IT 채용/취업
    "개발자 채용", "IT 채용", "개발자 취업",
    "주니어 개발자", "신입 개발자",
    # 채용 시장 담론 (감축·연봉)
    "개발자 감축", "개발자 연봉",
    # AI × 개발자 채용 (에이전틱 AI 담론 포함)
    "AI 개발자", "생성형 AI 채용", "AI 에이전트",
]
# 참고: "에이전트" 단독은 보험·부동산·스포츠 등 노이즈가 커서 "AI 에이전트"로 한정.
#       "ChatGPT 개발자"(2023 중심·협소)는 "AI 에이전트"로 교체 — 에이전틱 AI 채용 담론 포착.

# 분석 기간
DATE_BEFORE_AI = ("2021-01-01", "2022-10-31")   # ChatGPT 출시 이전
DATE_AFTER_AI  = ("2022-11-01", "2026-05-29")   # ChatGPT 출시 이후


# 빅카인즈 표준 export 컬럼 → 우리 스키마
_KEEP = ["뉴스 식별자", "발행일", "언론사", "제목", "본문", "키워드", "URL"]


def load_bigkinds_csv(path: str) -> pd.DataFrame:
    """빅카인즈 내보내기 파일(CSV/XLSX) 로드 및 컬럼 표준화.

    표준화: '일자'(YYYYMMDD) → '발행일'(datetime), 분석에 쓰는 컬럼만 보존.
    """
    p = Path(path)
    df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    if "일자" in df.columns:
        df = df.rename(columns={"일자": "발행일"})
    if "발행일" in df.columns:
        df["발행일"] = pd.to_datetime(df["발행일"].astype(str).str.slice(0, 8),
                                    format="%Y%m%d", errors="coerce")
    keep = [c for c in _KEEP if c in df.columns]
    return df[keep]


def merge_all(data_dir: str = "data/raw/news") -> pd.DataFrame:
    """하위폴더 포함 모든 빅카인즈 파일 병합 → '뉴스 식별자' 기준 중복 제거.

    그룹 키워드가 겹쳐 같은 기사가 여러 파일에 잡히므로 중복 제거 필수.
    """
    files = sorted(glob.glob(str(Path(data_dir) / "**" / "*.xlsx"), recursive=True)) + \
            sorted(glob.glob(str(Path(data_dir) / "**" / "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"{data_dir} 에 빅카인즈 파일(.xlsx/.csv)이 없습니다")

    frames = [load_bigkinds_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    n_raw = len(df)

    subset = "뉴스 식별자" if "뉴스 식별자" in df.columns else None
    df = df.drop_duplicates(subset=subset) if subset else df.drop_duplicates(subset=["발행일", "제목"])
    df = df.dropna(subset=["발행일"]).reset_index(drop=True)
    print(f"  [merge_all] {len(files)}파일 · {n_raw:,}행 → 중복제거 후 {len(df):,}행")
    return df
