"""
05_topic_modeling_bertopic.py
담당: 미정

BERTopic 토픽 모델링 + Word2Vec 공기어 + Co-occurrence Network + UMAP 시각화
수업 외 방법론 (가산점 ②)
출력: outputs/figures/, outputs/models/
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ast
import pandas as pd
import matplotlib.pyplot as plt

from src.modeling.bertopic_model import train_bertopic, get_topic_info
from src.visualization._fonts import setup_korean_font
from src.analysis.word2vec_analysis import train_word2vec, most_similar
from src.analysis.cooccurrence_network import build_cooccurrence_matrix, build_graph
from src.visualization.network_plot import draw_network
from src.visualization.umap_plot import reduce_and_plot

KR_PATH  = "data/processed/domestic/kr_jobs_clean.csv"
LINKEDIN_PATH = "data/processed/linkedin/linkedin_processed.csv"
FIG_DIR       = "outputs/figures"
MDL_DIR       = "outputs/models"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)


def load_docs_and_tokens(path: str, text_col: str = "text", token_col: str = "tokens"):
    df = pd.read_csv(path)
    docs = df[text_col].fillna("").tolist()
    token_lists = [ast.literal_eval(r) if isinstance(r, str) else r
                   for r in df[token_col].fillna("[]")]
    return df, docs, token_lists


def load_domestic_docs(path: str = KR_PATH):
    """국내공고 문서 로더. kr_jobs_clean.csv 에는 text/tokens 컬럼이 없으므로
    공고 본문(body) 텍스트를 문서로 사용한다(없으면 title 보강)."""
    df = pd.read_csv(path)
    body = df["body"].fillna("").astype(str)
    title = df["title"].fillna("").astype(str)
    docs = [(b if b.strip() else t) for b, t in zip(body, title)]
    return df, docs


def _save_bertopic_bar(info: pd.DataFrame, save_path: str, top_n_words: int = 5) -> None:
    """토픽별 대표 키워드 + 문서 수를 가로 막대그래프로 저장 (토픽 -1 outlier 제외)."""
    info = info[info["Topic"] != -1].copy().sort_values("Count")

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


# ---------------------------------------------------------------------------
# BERTopic — 국내공고(본문 4-4-2). LinkedIn 합산 경로는 코드에 보존(본문 미사용).
# ---------------------------------------------------------------------------
def run_bertopic() -> None:
    df_w, docs = load_domestic_docs(KR_PATH)
    # [코드 보존 — 본문 미사용] 글로벌 합산 시:
    #   _, docs_l, _ = load_docs_and_tokens(LINKEDIN_PATH, text_col="description")
    #   docs = docs + docs_l

    model, topics, probs = train_bertopic(docs, language="multilingual", nr_topics=7)
    info = get_topic_info(model)
    print("[BERTopic] 토픽 요약 (국내공고 기준):")
    print(info.to_string(index=False))

    model.save(f"{MDL_DIR}/bertopic_model")
    fig = model.visualize_topics()
    fig.write_html(f"{FIG_DIR}/bertopic_topics.html")

    # 토픽별 대표 키워드/문서 수 가로 막대그래프 (본문 그림)
    _save_bertopic_bar(info, f"{FIG_DIR}/bertopic_topics.png")
    n_topics = int((info["Topic"] != -1).sum())
    print(f"[BERTopic] 토픽 {n_topics}개 / 저장: {FIG_DIR}/bertopic_topics.png (+ .html)")


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
def run_umap() -> None:
    from sentence_transformers import SentenceTransformer
    df, docs = load_domestic_docs(KR_PATH)

    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = encoder.encode(docs, show_progress_bar=True)

    labels = df["role"].fillna("other").tolist()
    setup_korean_font()
    reduce_and_plot(embeddings, labels,
                    title="UMAP 직무군 산점도",
                    save_path=f"{FIG_DIR}/umap_clusters.png")
    print(f"[UMAP] 저장: {FIG_DIR}/umap_clusters.png ({len(docs)}건, 직무군 {len(set(labels))}종)")


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
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_bertopic()
    run_word2vec()
    run_cooccurrence()
    run_umap()
