"""
05_topic_modeling_bertopic.py
담당: 미정

BERTopic 토픽 모델링 + Word2Vec 공기어 + Co-occurrence Network + UMAP 시각화
수업 외 방법론 (가산점 ②)
출력: outputs/figures/, outputs/models/

[보조 분석 4-4-2/4-4-3 재현 파라미터]
- 임베딩(공유): SBERT "paraphrase-multilingual-MiniLM-L12-v2" 를 한 번 계산해
  BERTopic 군집과 UMAP 시각화가 같은 임베딩 배열을 공유.
- BERTopic: vectorizer=CountVectorizer(stop_words=한·영 불용어, min_df=5, ngram=(1,2)),
  min_topic_size=30(쏠림 시 15로 1회 재시도), nr_topics="auto"(토픽 과다 시 10으로 reduce),
  calculate_probabilities=False.
- 한국어 불용어(KO_STOPWORDS): 경험·개발·운영·설계 등 일반 업무어 추가 제거(아래 리스트).
- UMAP: 색상=JOB_LABELS(7종, role 12→7 매핑 공유). n_neighbors=15, min_dist=0.1, random_state=42
  (분리 약하면 n_neighbors=30, min_dist=0.0 으로 1회 재시도해 더 나은 쪽 채택).
- 모든 random_state=42.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ast
import pandas as pd
import matplotlib.pyplot as plt

from src.modeling.bertopic_model import train_bertopic, get_topic_info
from src.modeling.job_classifier import map_roles_to_labels, JOB_LABELS
from src.visualization._fonts import setup_korean_font
from src.analysis.word2vec_analysis import train_word2vec, most_similar
from src.analysis.cooccurrence_network import build_cooccurrence_matrix, build_graph
from src.visualization.network_plot import draw_network
from src.visualization.umap_plot import reduce_and_plot, reduce_embeddings, plot_2d

KR_PATH  = "data/processed/domestic/kr_jobs_clean.csv"
LINKEDIN_PATH = "data/processed/linkedin/linkedin_processed.csv"
FIG_DIR       = "outputs/figures"
MDL_DIR       = "outputs/models"
EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)

# 한국어 일반 업무어 불용어 — 임베딩 토픽이 이 어휘로 쏠리는 것을 방지(c-TF-IDF 단어 추출에서 제거).
KO_STOPWORDS = [
    "경험", "경험이", "개발", "운영", "설계", "업무", "담당", "관련", "우대", "자격",
    "능력", "이해", "활용", "사용", "가능", "지원", "채용", "부문", "모집", "및",
    "등", "위해", "통해",
    # 결과 확인 후 추가한 일반 업무어/문법 어미 (T0 쏠림 키워드 정리)
    "있으신", "있는", "있으며", "이상", "보유", "수행",
]


def build_stopwords() -> list[str]:
    """한국어 일반 업무어 + 영문 NLTK 불용어(없으면 sklearn 기본)를 합쳐 반환."""
    en = []
    try:
        from nltk.corpus import stopwords
        en = stopwords.words("english")
    except Exception:
        try:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            en = stopwords.words("english")
        except Exception:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            en = list(ENGLISH_STOP_WORDS)
    return sorted(set(KO_STOPWORDS) | set(en))


def encode_docs(docs: list[str]):
    """SBERT 임베딩 1회 계산 (BERTopic·UMAP 공유용). (encoder, embeddings) 반환."""
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(EMBED_MODEL)
    embeddings = encoder.encode(docs, show_progress_bar=True)
    return encoder, embeddings


def load_docs_and_tokens(path: str, text_col: str = "text", token_col: str = "tokens"):
    df = pd.read_csv(path)
    docs = df[text_col].fillna("").tolist()
    token_lists = [ast.literal_eval(r) if isinstance(r, str) else r
                   for r in df[token_col].fillna("[]")]
    return df, docs, token_lists


def load_domestic_docs(path: str = KR_PATH):
    """국내공고 문서 로더. kr_jobs_clean.csv 에는 text/tokens 컬럼이 없으므로
    제목 + 본문(자격·우대 포함) + 스킬 토큰을 결합해 문서로 사용한다(본문 중심)."""
    df = pd.read_csv(path)
    title = df["title"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    skills = df["skills"].fillna("").astype(str).str.replace("|", " ", regex=False)
    docs = (title + " " + body + " " + skills).str.strip().tolist()
    return df, docs


def _save_bertopic_bar(info: pd.DataFrame, save_path: str,
                       top_n_topics: int = 10, top_n_words: int = 4) -> None:
    """상위 토픽의 대표 키워드 + 문서 수를 가로 막대그래프로 저장 (토픽 -1 outlier 제외)."""
    info = info[info["Topic"] != -1].copy()
    info = info.sort_values("Count", ascending=False).head(top_n_topics).sort_values("Count")

    def kw_label(row):
        rep = row.get("Representation")
        words = rep[:top_n_words] if isinstance(rep, list) else str(row["Name"]).split("_")[1:top_n_words + 1]
        return f"T{row['Topic']}: " + ", ".join(words)

    labels = info.apply(kw_label, axis=1).tolist()
    counts = info["Count"].tolist()

    setup_korean_font()
    plt.figure(figsize=(11, max(4, 0.7 * len(labels) + 2)))
    bars = plt.barh(labels, counts, color="#4C78A8")
    for bar, c in zip(bars, counts):
        plt.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                 str(int(c)), va="center", fontsize=9)
    plt.title("BERTopic 토픽별 대표 키워드", fontsize=14)
    plt.xlabel("문서 수")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _top_topic_share(info: pd.DataFrame, n_docs: int) -> float:
    """최대 토픽(outlier 제외)이 전체 문서에서 차지하는 비율."""
    sub = info[info["Topic"] != -1]
    return float(sub["Count"].max()) / n_docs if len(sub) and n_docs else 1.0


# ---------------------------------------------------------------------------
# BERTopic — 국내공고(본문 4-4-2). LinkedIn 합산 경로는 코드에 보존(본문 미사용).
# ---------------------------------------------------------------------------
def run_bertopic(docs=None, embeddings=None, encoder=None) -> None:
    from sklearn.feature_extraction.text import CountVectorizer

    if docs is None:
        _, docs = load_domestic_docs(KR_PATH)
    if embeddings is None:
        encoder, embeddings = encode_docs(docs)
    # [코드 보존 — 본문 미사용] 글로벌 합산 시:
    #   _, docs_l, _ = load_docs_and_tokens(LINKEDIN_PATH, text_col="description")
    #   docs = docs + docs_l (임베딩도 함께 재계산 필요)

    stop_words = build_stopwords()
    n_docs = len(docs)

    def fit(min_topic_size):
        vectorizer = CountVectorizer(stop_words=stop_words, min_df=5, ngram_range=(1, 2))
        return train_bertopic(
            docs, embedding_model=encoder, vectorizer_model=vectorizer,
            min_topic_size=min_topic_size, nr_topics="auto",
            calculate_probabilities=False, embeddings=embeddings,
        )

    min_topic_size = 30
    model, topics, _ = fit(min_topic_size)
    info = get_topic_info(model)
    share = _top_topic_share(info, n_docs)

    # ★ 품질 가드: 한 토픽이 전체의 60%를 넘으면 쏠림으로 보고 min_topic_size 낮춰 1회 재시도.
    if share > 0.60:
        print(f"[BERTopic] 토픽 쏠림 감지(최대 토픽 {share:.0%}) → min_topic_size=15 재시도")
        min_topic_size = 15
        model, topics, _ = fit(min_topic_size)
        info = get_topic_info(model)
        share = _top_topic_share(info, n_docs)
        if share > 0.60:
            print(f"[BERTopic] 재시도 후에도 최대 토픽 {share:.0%} — 텍스트 공통 어휘 비중이 높아 분리 한계.")

    # 토픽이 과도하게 많으면 10개로 reduce
    n_topics = int((info["Topic"] != -1).sum())
    if n_topics > 12:
        print(f"[BERTopic] 토픽 {n_topics}개 → 10개로 reduce_topics")
        model.reduce_topics(docs, nr_topics=10)
        info = get_topic_info(model)
        n_topics = int((info["Topic"] != -1).sum())
        share = _top_topic_share(info, n_docs)

    print("[BERTopic] 토픽 요약 (국내공고 기준):")
    print(info.to_string(index=False))

    model.save(f"{MDL_DIR}/bertopic_model")
    try:
        model.visualize_topics().write_html(f"{FIG_DIR}/bertopic_topics.html")
    except Exception as e:
        print(f"[BERTopic] 인터랙티브 html 스킵: {e}")

    _save_bertopic_bar(info, f"{FIG_DIR}/bertopic_topics.png")
    print(f"[BERTopic] 최종 토픽 {n_topics}개 / 최대 토픽 점유율 {share:.1%} "
          f"(min_topic_size={min_topic_size}) / 저장: {FIG_DIR}/bertopic_topics.png")


# ---------------------------------------------------------------------------
# Word2Vec — 주요 키워드 공기어
# ---------------------------------------------------------------------------
def run_word2vec() -> None:
    _, _, tokens_w = load_docs_and_tokens(KR_PATH)
    _, _, tokens_n = load_docs_and_tokens("data/processed/news/news_processed.csv")
    all_tokens = tokens_w + tokens_n

    model = train_word2vec(all_tokens)
    model.save(f"{MDL_DIR}/word2vec.model")

    print("[Word2Vec] 주요 키워드 공기어:")
    for word in ["AI", "Python", "React", "신입", "LLM"]:
        similar = most_similar(model, word)
        print(f"  {word}: {similar[:5]}")


# ---------------------------------------------------------------------------
# Co-occurrence Network
# ---------------------------------------------------------------------------
def run_cooccurrence() -> None:
    _, _, tokens_w = load_docs_and_tokens(KR_PATH)
    edge_df = build_cooccurrence_matrix(tokens_w, min_count=5)
    G = build_graph(edge_df)
    draw_network(G, title="국내공고 기술스택 Co-occurrence Network",
                 save_path=f"{FIG_DIR}/cooccurrence_network.png")
    print("[Co-occurrence] 저장 완료")


# ---------------------------------------------------------------------------
# UMAP — 국내공고 직무군 임베딩 시각화 (본문 4-4-3)
#   다국어 SBERT(paraphrase-multilingual-MiniLM-L12-v2)로 한글 공고 임베딩 → UMAP 2D,
#   직무군(role) 색상. 글로벌(LinkedIn) 경로는 run_umap_linkedin() 에 보존(본문 미사용).
# ---------------------------------------------------------------------------
def run_umap(df=None, embeddings=None) -> None:
    from collections import Counter
    from sklearn.metrics import silhouette_score

    if df is None or embeddings is None:
        df, docs = load_domestic_docs(KR_PATH)
        _, embeddings = encode_docs(docs)

    # ★ 색상 라벨을 7종 JOB_LABELS 로 통일 (confusion matrix 와 동일한 12→7 매핑 공유)
    labels = map_roles_to_labels(df["role"].fillna("other").tolist())
    dist = {k: dist_v for k, dist_v in sorted(Counter(labels).items())}
    print("[UMAP] 7종 라벨 분포:", dist)

    enc_idx = {lab: i for i, lab in enumerate(JOB_LABELS)}
    y = [enc_idx.get(lab, len(JOB_LABELS)) for lab in labels]

    def sil(coords):
        try:
            return float(silhouette_score(coords, y))
        except Exception:
            return float("nan")

    setup_korean_font()
    coords = reduce_embeddings(embeddings, n_neighbors=15, min_dist=0.1, random_state=42)
    s1 = sil(coords)
    params = "n_neighbors=15, min_dist=0.1"

    # ★ 분리가 약하면 (n_neighbors=30, min_dist=0.0)으로 1회 재시도 → 더 나은 쪽 채택
    if not (s1 == s1) or s1 < 0.05:
        coords2 = reduce_embeddings(embeddings, n_neighbors=30, min_dist=0.0, random_state=42)
        s2 = sil(coords2)
        print(f"[UMAP] 1차 분리 약함(silhouette={s1:.3f}) → (n_neighbors=30, min_dist=0.0) 재시도(silhouette={s2:.3f})")
        if (s2 == s2) and (not (s1 == s1) or s2 > s1):
            coords, s1, params = coords2, s2, "n_neighbors=30, min_dist=0.0"

    plot_2d(coords, labels, title="UMAP 직무군 산점도",
            save_path=f"{FIG_DIR}/umap_clusters.png", alpha=0.5, s=8)
    print(f"[UMAP] 최종 {params}, silhouette={s1:.3f} / "
          f"저장: {FIG_DIR}/umap_clusters.png ({len(labels)}건, 라벨 {len(set(labels))}종)")
    if not (s1 == s1) or s1 < 0.05:
        print("[UMAP] 직무군 임베딩이 강하게 분리되지는 않음(공고 텍스트 공통 어휘 비중 높음).")


def run_umap_linkedin() -> None:
    """[코드 보존 — 본 논문 본문 분석 미사용] 글로벌(LinkedIn) 직무군 UMAP."""
    from sentence_transformers import SentenceTransformer
    df, docs, _ = load_docs_and_tokens(LINKEDIN_PATH, text_col="description")

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(docs[:5000], show_progress_bar=True)  # 샘플 5,000건

    labels = df["role"].fillna("other").tolist()[:5000]
    reduce_and_plot(embeddings, labels,
                    title="UMAP — LinkedIn IT Job Category Clusters",
                    save_path=f"{FIG_DIR}/umap_clusters_linkedin.png")
    print("[UMAP/LinkedIn 보존] 저장 완료")


# ---------------------------------------------------------------------------
# 보조 분석 통합 실행 — SBERT 임베딩 1회 계산 후 BERTopic·UMAP 공유
# ---------------------------------------------------------------------------
def run_aux_models() -> None:
    df, docs = load_domestic_docs(KR_PATH)
    print(f"[보조분석] 국내공고 {len(docs)}건 임베딩 계산(공유): {EMBED_MODEL}")
    encoder, embeddings = encode_docs(docs)
    run_bertopic(docs=docs, embeddings=embeddings, encoder=encoder)
    run_umap(df=df, embeddings=embeddings)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_aux_models()
    # Word2Vec / Co-occurrence 는 tokens 컬럼이 있는 데이터에서 별도 실행:
    #   run_word2vec(); run_cooccurrence()
