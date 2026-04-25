import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_v = float(np.min(scores))
    max_v = float(np.max(scores))
    if max_v - min_v < 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_v) / (max_v - min_v)


def _tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class HybridRetriever:
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        bm25_weight: float = 0.45,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_candidates: int = 20,
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reranker = CrossEncoder(reranker_model_name)
        self.bm25_weight = bm25_weight
        self.embedding_weight = 1.0 - bm25_weight
        self.rerank_candidates = max(5, rerank_candidates)
        self.chunks: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._embeddings: np.ndarray | None = None

    def fit(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [chunk.get("text", "") for chunk in chunks]
        self._tokenized_corpus = [_tokenize_for_bm25(text) for text in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks or self._bm25 is None or self._embeddings is None:
            return []

        tokenized_query = _tokenize_for_bm25(query)
        bm25_scores = np.array(self._bm25.get_scores(tokenized_query), dtype=float)
        bm25_scores = _normalize_scores(bm25_scores)

        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        # With normalized vectors, dot product equals cosine similarity.
        semantic_scores = np.dot(self._embeddings, query_embedding)
        semantic_scores = _normalize_scores(semantic_scores)

        hybrid_scores = self.bm25_weight * bm25_scores + self.embedding_weight * semantic_scores
        candidate_count = min(max(top_k, self.rerank_candidates), len(self.chunks))
        candidate_indices = np.argsort(hybrid_scores)[::-1][:candidate_count]

        rerank_pairs = [[query, self.chunks[idx]["text"]] for idx in candidate_indices]
        rerank_scores = np.array(self.reranker.predict(rerank_pairs), dtype=float)
        rerank_scores = _normalize_scores(rerank_scores)
        reranked_order = np.argsort(rerank_scores)[::-1][:top_k]

        results = []
        for local_idx in reranked_order:
            idx = int(candidate_indices[local_idx])
            bm25_score = float(bm25_scores[idx])
            semantic_score = float(semantic_scores[idx])
            hybrid_score = float(hybrid_scores[idx])
            rerank_score = float(rerank_scores[local_idx])
            results.append(
                {
                    "page": self.chunks[idx]["page"],
                    "text": self.chunks[idx]["text"],
                    "chapter_number": self.chunks[idx].get("chapter_number"),
                    "chapter_title": self.chunks[idx].get("chapter_title", ""),
                    "section_number": self.chunks[idx].get("section_number", ""),
                    "section_title": self.chunks[idx].get("section_title", ""),
                    "score": rerank_score,
                    "bm25_score": bm25_score,
                    "semantic_score": semantic_score,
                    "hybrid_score": hybrid_score,
                    "rerank_score": rerank_score,
                }
            )
        return results