"""
직무군 분류기 — title + (점핏) jobCategories 기반 규칙 매칭.

점핏: jobCategories 우선 사용 (예: "프론트엔드 개발자,devops/시스템 엔지니어").
원티드: categoryTags가 비어있어 title 패턴 매칭.

분류군 (10개):
  backend / frontend / fullstack / ai_ml / data /
  devops / mobile / qa / security / embedded / game / other
"""

import re

ROLE_LABELS = {
    "backend":   "백엔드",
    "frontend":  "프론트엔드",
    "fullstack": "풀스택",
    "ai_ml":     "AI/ML",
    "data":      "데이터",
    "devops":    "DevOps/인프라",
    "mobile":    "모바일",
    "qa":        "QA",
    "security":  "보안",
    "embedded":  "임베디드/HW",
    "game":      "게임",
    "other":     "기타",
}


# 점핏 jobCategories 토큰 → 우리 분류군. 모호한 카테고리(SW/솔루션·기술지원·블록체인·VR/AR 등)는
# 의도적으로 매핑에서 제외 → 해당 공고는 title 기반 분류로 fall-through.
JUMPIT_CATEGORY_MAP = {
    "서버/백엔드 개발자": "backend",
    "프론트엔드 개발자": "frontend",
    "웹 풀스택 개발자": "fullstack",
    "인공지능/머신러닝": "ai_ml",
    "빅데이터 엔지니어": "data",
    "데이터 사이언티스트": "data",
    "데이터 엔지니어": "data",
    "devops/시스템 엔지니어": "devops",
    "DBA": "devops",
    "안드로이드 개발자": "mobile",
    "iOS 개발자": "mobile",
    "크로스플랫폼 앱개발자": "mobile",
    "QA 엔지니어": "qa",
    "정보보안 담당자": "security",
    "HW/임베디드": "embedded",
    "게임 클라이언트 개발자": "game",
    "게임 서버 개발자": "game",
    "웹퍼블리셔": "frontend",
}


# title 매칭 규칙 — 위에서 아래로, 먼저 맞는 게 이김
TITLE_RULES = [
    ("mobile",   r"\b(ios|android|안드로이드|아이폰|flutter|react.?native|크로스플랫폼)\b|모바일\s*(개발|앱|어플)"),
    ("game",     r"(게임|game\s*(client|server|engineer|developer|개발))"),
    ("ai_ml",    r"(ai\s*(엔지니어|engineer|researcher|scientist|개발|native|agent|에이전트|service|연구|research|모델|model|application|compiler|tool|operation|오퍼레이션)|머신러닝|machine\s*learning|딥러닝|deep\s*learning|ml\s*(engineer|ops|developer|researcher)|mlops|\bllm\b|\bnlp\b|natural\s*language|computer\s*vision|컴퓨터\s*비전|vision\s*(ai|research)|world\s*model|인공지능|생성형\s*ai|reinforcement\s*learning|prompt\s*engineer|llmops|gen\s*ai|generative\s*ai|robot\s*learning|physical\s*ai|\brag\b)"),
    ("data",     r"(데이터\s*(엔지니어|플랫폼|분석가|사이언티스트|통합|시스템|관리)|data\s*(engineer|scientist|analyst|platform|system)|빅데이터|big\s*data|bi\s*엔지니어|analytics|airflow\s*엔지니어)"),
    ("embedded", r"(임베디드|embedded|펌웨어|firmware|fpga|커널|kernel|디바이스\s*드라이버|device\s*driver|wi.?fi\s*s/?w|이동통신\s*시스템|자동화장비|pc제어|hw\b|hardware|vision\s*sw|비젼\s*sw|vision\s*소프트웨어|로봇\s*(소프트웨어|sw|제어|매니퓰레이터)|robotics|robot\s*engineer|제어\s*(프로그램|sw|소프트웨어)|3d\s*프린터|pbx|edge\s*sw|전장\s*설계|센서\s*(캘리브레이션|calibration)|\bcamera\b|\blidar\b|반도체|pcb)"),
    ("devops",   r"(devops|devsecops|sre|reliability|infrastructure|\binfra\b|인프라|cloud\s*engineer|platform\s*engineer|플랫폼\s*엔지니어|시스템\s*엔지니어|네트워크\s*엔지니어|dba\b|dbms|쿠버네티스|kubernetes)"),
    ("qa",       r"(\bqa\b|\bqe\b|품질|테스트\s*엔지니어|test\s*engineer|sdet|automation\s*tester)"),
    ("security", r"(보안|security|취약점|침해대응|소프트웨어\s*보안|application\s*security|보안칩|정보보호|개인정보보호)"),
    ("fullstack",r"(풀스택|full.?stack)"),
    ("frontend", r"(프론트|front.?end|ui\s*개발|ux\s*개발|web\s*publisher|웹\s*퍼블리셔|react|vue|angular|next\.?js)"),
    ("backend",  r"(백엔드|back.?end|서버\s*(엔지니어|개발|engineer)|server\s*engineer|api\s*개발|spring|node\.?js|django|java\s*개발자|kotlin\s*개발|python\s*개발|go\s*개발|golang\s*개발|\.net\s*개발|닷넷\s*개발|asp\.net|c#\s*개발|software\s*(engineer|developer)|principal\s*engineer|staff\s*engineer|lead\s*engineer\b|senior\s*engineer\b|software\s*architect|developer\s*relations|developer\s*advocate)"),
]


_COMPILED = [(role, re.compile(p, re.IGNORECASE)) for role, p in TITLE_RULES]


def classify_from_title(title: str) -> str:
    if not isinstance(title, str) or not title:
        return "other"
    for role, pat in _COMPILED:
        if pat.search(title):
            return role
    return "other"


def classify_jumpit_categories(categories: str) -> str | None:
    """점핏 jobCategories(comma 구분) 중 우선순위 높은 분류군 반환.

    여러 카테고리 매칭 시 다음 우선순위:
      ai_ml > data > mobile > embedded > security > qa > devops > fullstack > frontend > backend > game > other
    """
    if not isinstance(categories, str) or not categories:
        return None
    PRIORITY = ["ai_ml", "data", "mobile", "embedded", "security", "qa",
                "devops", "fullstack", "frontend", "backend", "game", "other"]
    found = set()
    for c in categories.split(","):
        c = c.strip()
        mapped = JUMPIT_CATEGORY_MAP.get(c)
        if mapped:
            found.add(mapped)
    if not found:
        return None
    for p in PRIORITY:
        if p in found:
            return p
    return "other"


def classify_record(source: str, title: str, jumpit_categories: str | None = None) -> str:
    """단일 레코드 분류 — 점핏은 카테고리 우선, 미매칭/원티드는 title 규칙."""
    if source == "jumpit" and jumpit_categories:
        role = classify_jumpit_categories(jumpit_categories)
        if role:
            return role
    return classify_from_title(title)


# ─── 뉴스용 멀티라벨 ──────────────────────────────────────────────
def match_all_roles(text: str) -> set[str]:
    """텍스트에서 매칭되는 모든 직무 라벨 집합 (뉴스 멀티라벨용).

    채용공고와 달리 한 기사가 여러 직무를 동시에 언급할 수 있으므로
    매칭 우선순위를 두지 않고 모든 매칭 라벨을 반환.
    'other'는 반환에서 제외 (의미 있는 매칭만 캐치).
    """
    if not isinstance(text, str) or not text:
        return set()
    return {role for role, pat in _COMPILED if pat.search(text)}
