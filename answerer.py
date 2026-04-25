import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_query_for_search(query: str) -> str:
    subject = _extract_question_subject(query)
    if not subject:
        return query
    return subject


def _is_noise_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    noise_markers = [
        "chapter",
        "m01_",
        ".indd",
        "pm",
        "am",
    ]
    if any(marker in lower for marker in noise_markers):
        return True
    if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", sentence):
        return True
    if len(re.findall(r"[A-Z]{2,}", sentence)) >= 5:
        return True
    if "we'll " in lower or "this chapter" in lower:
        return True
    return False


def _clean_text(text: str) -> str:
    # Join line-broken words from PDF extraction, then normalize spaces.
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\bM\d{2}_[A-Z0-9_]+\b", " ", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}\s?(?:AM|PM)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bCHAPTER\s+\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_question_subject(query: str) -> str:
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
    return ""


def _tokenize_for_match(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()))


def _collect_candidate_sentences(results: list[dict]) -> list[str]:
    candidates: list[str] = []
    seen = set()
    for item in results[:3]:
        cleaned = _clean_text(item.get("text", ""))
        if not cleaned:
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
        for sentence in sentences:
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            if _is_noise_sentence(sentence):
                continue
            if len(sentence) < 25:
                continue
            candidates.append(sentence)
    return candidates


def _rank_sentences_extractively(query: str, sentences: list[str]) -> list[tuple[float, str]]:
    if not sentences:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([query] + sentences)
    q_vec = matrix[0:1]
    s_vec = matrix[1:]
    tfidf_scores = cosine_similarity(q_vec, s_vec).flatten()

    subject = _extract_question_subject(query)
    query_terms = _tokenize_for_match(subject if subject else query)

    ranked: list[tuple[float, str]] = []
    for idx, sentence in enumerate(sentences):
        sentence_terms = _tokenize_for_match(sentence)
        overlap = len(query_terms & sentence_terms) / max(1, len(query_terms))
        # Keep this extractive: no rewriting, only scoring bonuses.
        score = (0.75 * float(tfidf_scores[idx])) + (0.25 * overlap)
        ranked.append((score, sentence))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def build_short_answer(query: str, results: list[dict]) -> str:
    if not results:
        return "No relevant answer was found."

    candidates = _collect_candidate_sentences(results)
    if not candidates:
        fallback = _clean_text(results[0].get("text", ""))
        return fallback[:280] + ("..." if len(fallback) > 280 else "")

    ranked = _rank_sentences_extractively(query, candidates)
    if not ranked:
        best = candidates[0]
        return best[:280] + ("..." if len(best) > 280 else "")

    best_score, best_sentence = ranked[0]
    selected = [best_sentence]

    # Return two sentences only when both are clearly relevant.
    if len(ranked) > 1:
        second_score, second_sentence = ranked[1]
        if second_score >= 0.65 * best_score and second_score >= 0.08:
            selected.append(second_sentence)

    answer = " ".join(selected).strip()
    if len(answer) > 320:
        answer = answer[:317].rstrip() + "..."
    return answer