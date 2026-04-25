import argparse
import re

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chunker import chunk_pages
from text_cleaner import clean_pdf_pages


def normalize_query_for_search(query: str) -> str:
    q = query.strip().rstrip("?.!").lower()
    patterns = [
        r"^what is (?:the )?(.+)$",
        r"^what are (?:the )?(.+)$",
        r"^define (.+)$",
        r"^explain (.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, q)
        if match:
            return match.group(1).strip()
    return query


def extractive_short_answer(query: str, results: list[dict]) -> str:
    if not results:
        return "No relevant answer was found."
    merged = " ".join((item.get("text", "") or "") for item in results[:3]).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", merged) if s.strip()]
    if not sentences:
        return merged[:280] + ("..." if len(merged) > 280 else "")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([query] + sentences)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    best_idx = int(scores.argmax())
    answer = sentences[best_idx]
    return answer[:320] + ("..." if len(answer) > 320 else "")


def run_baseline(pdf_path: str, query: str, top_k: int = 3):
    print("=== TF-IDF Baseline Demo ===")
    print(f"PDF: {pdf_path}")
    print(f"Query: {query}")
    print("")

    reader = PdfReader(pdf_path)
    raw_pages = [(page.extract_text() or "").strip() for page in reader.pages]
    cleaned_texts = clean_pdf_pages(raw_pages)
    pages = [{"page": i + 1, "text": cleaned_texts[i]} for i in range(len(cleaned_texts))]
    chunks = chunk_pages(pages, min_words=400, max_words=800, overlap_words=75)

    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(texts)

    search_query = normalize_query_for_search(query)
    q_vec = vectorizer.transform([search_query])
    sims = cosine_similarity(q_vec, doc_matrix).flatten()
    ranked_idx = sims.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_idx:
        results.append(
            {
                "page": chunks[idx]["page"],
                "text": chunks[idx]["text"],
                "score": float(sims[idx]),
            }
        )

    answer = extractive_short_answer(query, results)

    print("Short Answer:")
    print(answer)
    print("")
    print("Top Passages:")
    for i, item in enumerate(results, start=1):
        excerpt = item["text"][:450].replace("\n", " ").strip()
        print(f"[Result {i}] Page {item['page']} | Score: {item['score']:.3f}")
        print(excerpt + ("..." if len(item["text"]) > 450 else ""))
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TF-IDF baseline demo for screenshots.")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--query", required=True, help="Question to test")
    parser.add_argument("--top-k", type=int, default=3, help="Top passages to print")
    args = parser.parse_args()
    run_baseline(args.pdf, args.query, top_k=args.top_k)
