"""
기술스택·역량 사전 (Validity Guardrail #2)
한↔영 비교의 유일한 공통 단위. 동의어를 표준 표현으로 정규화한다.
"""

# 범주별 표준 토큰 목록
TECH_DICT: dict[str, list[str]] = {
    "programming_language": [
        "Java", "Python", "JavaScript", "TypeScript",
        "C", "C++", "C#", "Kotlin", "Swift", "Go", "Rust",
    ],
    "frontend": [
        "React", "Vue", "Angular", "Next.js",
        "HTML", "CSS", "Tailwind", "Redux",
    ],
    "backend": [
        "Spring", "Node.js", "Express", "Django", "Flask", "FastAPI",
    ],
    "database": [
        "MySQL", "PostgreSQL", "MongoDB", "Oracle",
        "Redis", "Elasticsearch",
    ],
    "cloud_infra": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes",
        "Linux", "Nginx", "CI/CD",
    ],
    "ai_data": [
        "AI", "Machine Learning", "Deep Learning", "LLM",
        "Generative AI", "NLP", "Computer Vision",
        "PyTorch", "TensorFlow", "Pandas",
    ],
    "collaboration": [
        "Git", "GitHub", "Jira", "Slack", "Notion", "Agile", "Scrum",
    ],
    "soft_skill": [
        "커뮤니케이션", "협업", "문제해결", "문서화", "자기주도", "프로젝트 관리",
    ],
}

# 동의어 → 표준 표현 매핑
SYNONYMS: dict[str, str] = {
    # Frontend
    "리액트": "React",
    "React.js": "React",
    "ReactJS": "React",
    "Vue.js": "Vue",
    "VueJS": "Vue",
    # Backend
    "스프링": "Spring",
    "Spring Boot": "Spring",
    "Node": "Node.js",
    "NodeJS": "Node.js",
    # Language
    "자바": "Java",
    "자바스크립트": "JavaScript",
    "JS": "JavaScript",
    "타입스크립트": "TypeScript",
    "TS": "TypeScript",
    "파이썬": "Python",
    # AI/Data
    "인공지능": "AI",
    "Artificial Intelligence": "AI",
    "머신러닝": "Machine Learning",
    "Machine Learning": "Machine Learning",
    "ML": "Machine Learning",
    "딥러닝": "Deep Learning",
    "Deep Learning": "Deep Learning",
    "DL": "Deep Learning",
    "생성형 AI": "Generative AI",
    "GenAI": "Generative AI",
    # Cloud
    "아마존 웹 서비스": "AWS",
    "쿠버네티스": "Kubernetes",
    "도커": "Docker",
}


def normalize(token: str) -> str:
    """동의어 사전으로 토큰 표준화."""
    return SYNONYMS.get(token, token)
