from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = None
        self.chunks = []

    def fit(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.chunk_vectors is None or not self.chunks:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()

        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "page": self.chunks[idx]["page"],
                "text": self.chunks[idx]["text"],
                "score": float(similarities[idx])
            })

        return results