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

from src.modeling.bertopic_model import train_bertopic, get_topic_info
from src.analysis.word2vec_analysis import train_word2vec, most_similar
from src.analysis.cooccurrence_network import build_cooccurrence_matrix, build_graph
from src.visualization.network_plot import draw_network
from src.visualization.umap_plot import reduce_and_plot

KR_PATH  = "data/processed/kr_jobs_clean.csv"
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


# ---------------------------------------------------------------------------
# BERTopic — 국내공고 + LinkedIn 합산
# ---------------------------------------------------------------------------
def run_bertopic() -> None:
    df_w, docs_w, _ = load_docs_and_tokens(KR_PATH)
    df_l, docs_l, _ = load_docs_and_tokens(LINKEDIN_PATH)
    docs = docs_w + docs_l

    model, topics, probs = train_bertopic(docs, nr_topics=7)
    info = get_topic_info(model)
    print("[BERTopic] 토픽 요약:")
    print(info.to_string(index=False))

    model.save(f"{MDL_DIR}/bertopic_model")
    fig = model.visualize_topics()
    fig.write_html(f"{FIG_DIR}/bertopic_topics.html")
    print("[BERTopic] 저장 완료")


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
# UMAP — LinkedIn 직무군 임베딩 시각화
# ---------------------------------------------------------------------------
def run_umap() -> None:
    from sentence_transformers import SentenceTransformer
    df, docs, _ = load_docs_and_tokens(LINKEDIN_PATH)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(docs[:5000], show_progress_bar=True)  # 샘플 5,000건

    labels = df["job_category"].fillna("Other IT").tolist()[:5000]
    reduce_and_plot(embeddings, labels,
                    title="UMAP — LinkedIn IT Job Category Clusters",
                    save_path=f"{FIG_DIR}/umap_clusters.png")
    print("[UMAP] 저장 완료")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_bertopic()
    run_word2vec()
    run_cooccurrence()
    run_umap()
