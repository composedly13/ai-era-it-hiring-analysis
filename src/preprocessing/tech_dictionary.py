"""
기술스택·역량 사전 (측정 도구 · Validity Guardrail #2·#5)
=========================================================
한↔영 비교의 유일한 공통 단위. 사이트 제공 태그(점핏 techStacks / 원티드 skill_tags /
LinkedIn skills)는 normalize_tag로 표준화하고, 자유서술 본문은 extract_skills로 추출한다.

AI 관련 토큰은 4-tier로 분리:
  Tier A — 생성형 AI 코딩 도구 (Copilot · Cursor · Claude Code 등)
  Tier B — AI 모델 · 플랫폼 (GPT · Claude · Gemini · OpenAI 등)
  Tier C — AI/ML 직무 역량 (LLM · RAG · PyTorch · MLOps 등)
  Tier D — 모호한 "AI" 단일 언급 (맥락 의존, 보조 지표)

주의: 이 사전이 곧 측정 기구다. 무작위 100건 수동 라벨로 precision/recall(목표 ≥0.9)을
검증·보완할 것(notebooks/03).
"""

import re

# canonical -> 별칭(소문자/한글). 등록 순서는 의미 없음(길이 내림차순으로 매칭).
SKILL_ALIASES = {
    # Languages
    "Java": ["java", "자바"],
    "Python": ["python", "파이썬"],
    "JavaScript": ["javascript", "자바스크립트"],
    "TypeScript": ["typescript", "타입스크립트"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "csharp", "c sharp"],
    "Kotlin": ["kotlin", "코틀린"],
    "Swift": ["swift", "스위프트"],
    "Go": ["golang", "go"],
    "Rust": ["rust"],
    "PHP": ["php"],
    "Ruby": ["ruby", "루비"],
    "Scala": ["scala"],
    "SQL": ["sql"],
    # Frontend
    "React": ["react", "react.js", "reactjs", "리액트"],
    "Vue": ["vue", "vue.js", "vuejs"],
    "Angular": ["angular", "앵귤러"],
    "Next.js": ["next.js", "nextjs"],
    "Svelte": ["svelte"],
    "Redux": ["redux"],
    "HTML": ["html"],
    "CSS": ["css"],
    # Backend
    "Spring": ["spring", "spring boot", "springboot", "스프링", "스프링부트"],
    "Node.js": ["node.js", "nodejs", "node"],
    "Express": ["express", "expressjs"],
    "Django": ["django", "장고"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "NestJS": ["nestjs", "nest.js"],
    # Mobile
    "Android": ["android", "안드로이드"],
    "iOS": ["ios"],
    "Flutter": ["flutter", "플러터"],
    "React Native": ["react native", "react-native"],
    # Database
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo"],
    "Oracle": ["oracle", "오라클"],
    "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch", "elastic search"],
    "MariaDB": ["mariadb"],
    "Kafka": ["kafka", "카프카"],
    # Cloud / Infra
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud"],
    "Docker": ["docker", "도커"],
    "Kubernetes": ["kubernetes", "k8s", "쿠버네티스"],
    "Linux": ["linux", "리눅스"],
    "Nginx": ["nginx"],
    "Terraform": ["terraform"],
    "Jenkins": ["jenkins", "젠킨스"],
    "Airflow": ["airflow"],
    "Spark": ["spark"],
    "Hadoop": ["hadoop"],
    # Collaboration
    "Git": ["git", "깃"],
    "GitHub": ["github", "깃허브"],
    "GitLab": ["gitlab"],
    "Jira": ["jira", "지라"],
    "REST API": ["rest api", "restful", "rest"],
    "GraphQL": ["graphql"],
    # ---- Tier A: 생성형 AI 코딩 도구 ----
    "GitHub Copilot": ["github copilot", "깃허브 코파일럿"],
    "Copilot": ["copilot", "코파일럿"],
    "Cursor": ["cursor"],
    "Claude Code": ["claude code", "claude-code"],
    "Windsurf": ["windsurf"],
    "Devin": ["devin"],
    "Aider": ["aider"],
    "Codex": ["codex"],
    "Antigravity": ["antigravity"],
    # ---- Tier B: AI 모델 · 플랫폼 ----
    "ChatGPT": ["chatgpt", "chat-gpt", "chat gpt"],
    "GPT-4": ["gpt-4", "gpt4", "gpt-4o", "gpt 4", "gpt-4 turbo"],
    "GPT": ["gpt"],
    "Claude": ["claude"],
    "Gemini": ["gemini"],
    "OpenAI": ["openai", "open ai"],
    "Anthropic": ["anthropic"],
    "Llama": ["llama"],
    "Mistral": ["mistral"],
    # ---- Tier C: AI/ML 직무 역량 ----
    "AI": ["ai", "인공지능", "a.i."],
    "Machine Learning": ["machine learning", "머신러닝", "머신 러닝", "기계학습"],
    "Deep Learning": ["deep learning", "딥러닝", "딥 러닝"],
    "LLM": ["llm", "대규모 언어 모델"],
    "NLP": ["nlp", "자연어처리", "자연어 처리"],
    "Computer Vision": ["computer vision", "컴퓨터 비전"],
    "PyTorch": ["pytorch", "파이토치"],
    "TensorFlow": ["tensorflow", "텐서플로", "텐서플로우"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "사이킷런"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "Fine-tuning": ["fine-tuning", "finetuning", "fine tuning", "파인튜닝", "파인 튜닝"],
    "Embedding": ["embedding", "embeddings", "임베딩"],
    "Vector DB": ["vector db", "vector database", "vector store", "벡터 db", "벡터 데이터베이스"],
    "Hugging Face": ["hugging face", "huggingface"],
    "Transformers": ["transformers"],
    "MLOps": ["mlops", "ml ops", "엠엘옵스"],
}


# AI 4-tier 분류 — RQ1·RQ4·RQ5 분석 시 분해용
AI_TIERS = {
    "tier_a_coding_tool": [
        "GitHub Copilot", "Copilot", "Cursor", "Claude Code",
        "Windsurf", "Devin", "Aider", "Codex", "Antigravity",
    ],
    "tier_b_model_platform": [
        "ChatGPT", "GPT-4", "GPT", "Claude", "Gemini",
        "OpenAI", "Anthropic", "Llama", "Mistral",
    ],
    "tier_c_ml_skill": [
        "LLM", "RAG", "Fine-tuning", "Embedding", "Vector DB",
        "Hugging Face", "Transformers", "MLOps",
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "PyTorch", "TensorFlow",
    ],
    "tier_d_generic": ["AI"],
}

# canonical -> tier 역매핑
_SKILL2TIER = {s: tier for tier, skills in AI_TIERS.items() for s in skills}


def get_ai_tier(skill: str) -> str | None:
    """canonical 스킬명 → AI tier (A/B/C/D) 또는 None(비-AI)."""
    return _SKILL2TIER.get(skill)


# 본문 추출 시 제외할 모호 토큰(사이트 태그로만 신뢰)
#   "Go": "go to ~" 일반 동사 충돌
#   짧고 모호한 "GPT" 단독은 본문에서 자주 잡히지만 의미 보존을 위해 유지
BODY_EXCLUDE = {"Go"}

# 범주별 표준 토큰 (분석·시각화 그룹핑용) — AI는 4-tier로 분리
TECH_DICT = {
    "language": ["Java", "Python", "JavaScript", "TypeScript", "C++", "C#", "Kotlin", "Swift", "Go", "Rust", "PHP", "Ruby", "Scala", "SQL"],
    "frontend": ["React", "Vue", "Angular", "Next.js", "Svelte", "Redux", "HTML", "CSS"],
    "backend": ["Spring", "Node.js", "Express", "Django", "Flask", "FastAPI", "NestJS"],
    "mobile": ["Android", "iOS", "Flutter", "React Native"],
    "database": ["MySQL", "PostgreSQL", "MongoDB", "Oracle", "Redis", "Elasticsearch", "MariaDB"],
    "cloud_infra": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Linux", "Nginx", "Terraform", "Jenkins", "Kafka", "Airflow"],
    "data_eng": ["Pandas", "NumPy", "scikit-learn", "Spark", "Hadoop"],
    "ai_coding_tool": AI_TIERS["tier_a_coding_tool"],
    "ai_model_platform": AI_TIERS["tier_b_model_platform"],
    "ai_ml_skill": AI_TIERS["tier_c_ml_skill"],
    "ai_generic": AI_TIERS["tier_d_generic"],
    "collaboration": ["Git", "GitHub", "GitLab", "Jira", "REST API", "GraphQL"],
}

# 별칭 -> canonical 역매핑 (길이 내림차순으로 정렬 → 긴 패턴 우선 매칭)
_ALIAS2CANON = {a.lower(): canon for canon, aliases in SKILL_ALIASES.items() for a in aliases}
_ALIASES_SORTED = sorted(_ALIAS2CANON.keys(), key=len, reverse=True)
_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(a) for a in _ALIASES_SORTED) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def extract_skills(text, exclude_ambiguous=True):
    """본문 텍스트에서 사전 기반 스킬 추출 → canonical 집합."""
    if not text:
        return set()
    found = set()
    for m in _PATTERN.finditer(text):
        canon = _ALIAS2CANON.get(m.group().lower())
        if not canon or (exclude_ambiguous and canon in BODY_EXCLUDE):
            continue
        found.add(canon)
    return found


def normalize_tag(tag):
    """사이트 제공 태그 1개 → canonical 집합(실패 시 원본 보존)."""
    if not tag or not tag.strip():
        return set()
    t = tag.strip()
    direct = _ALIAS2CANON.get(t.lower())
    if direct:
        return {direct}
    found = extract_skills(t, exclude_ambiguous=False)
    return found if found else {t}


def normalize(token):
    """(하위호환) 단일 토큰 표준화 → canonical 문자열 또는 원본."""
    return _ALIAS2CANON.get(str(token).lower(), token)
