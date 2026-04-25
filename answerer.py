import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def classify_question_type(query: str) -> str:
    q = query.strip().lower()

    definition_patterns = [r"\bwhat is\b", r"\bdefine\b", r"\bmeaning of\b"]
    location_patterns = [r"\bwhere\b", r"\bwhich chapter\b", r"\bwhich page\b"]
    count_patterns = [r"\bhow many\b", r"\bnumber of\b"]
    summary_patterns = [r"\bsummarize\b", r"\bsummary of\b", r"\bmainly discuss\b"]

    if any(re.search(p, q) for p in summary_patterns):
        return "summary"
    if any(re.search(p, q) for p in count_patterns):
        return "count"
    if any(re.search(p, q) for p in location_patterns):
        return "location"
    if any(re.search(p, q) for p in definition_patterns):
        return "definition"
    return "default"


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


def _pick_definition_answer(query: str, sentences: list[str]) -> str:
    subject = _extract_question_subject(query)
    subject_terms = _tokenize_for_match(subject if subject else query)
    if not subject_terms:
        ranked = _rank_sentences_extractively(query, sentences)
        return ranked[0][1] if ranked else ""

    def_sentences = []
    for sentence in sentences:
        lower = sentence.lower()
        if not any(k in lower for k in [" is ", " are ", " refers to ", "defined as"]):
            continue
        sent_terms = _tokenize_for_match(sentence)
        overlap = len(sent_terms & subject_terms)
        if overlap == 0:
            continue
        def_sentences.append((overlap, sentence))

    if def_sentences:
        def_sentences.sort(key=lambda x: x[0], reverse=True)
        return def_sentences[0][1]

    ranked = _rank_sentences_extractively(query, sentences)
    return ranked[0][1] if ranked else ""


def _pick_location_answer(query: str, results: list[dict], sentences: list[str]) -> str:
    if not results:
        return "No relevant location was found."

    top_pages = [str(item.get("page", "")) for item in results[:3] if item.get("page") is not None]
    pages_text = ", ".join(top_pages) if top_pages else "unknown pages"

    ranked = _rank_sentences_extractively(query, sentences)
    if ranked:
        best_sentence = ranked[0][1]
        return f"This topic is discussed around page(s): {pages_text}. Most relevant passage: {best_sentence}"

    return f"This topic is discussed around page(s): {pages_text}."


def _pick_count_answer(query: str, results: list[dict], sentences: list[str]) -> str:
    # Rule-based extraction for "how many / number of" style questions.
    candidates = []
    for sentence in sentences:
        nums = re.findall(r"\b\d+\b", sentence)
        if nums:
            candidates.append((len(nums), sentence))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    pages = [item.get("page") for item in results if item.get("page") is not None]
    if pages:
        freq = Counter(pages)
        top_page = freq.most_common(1)[0][0]
        return (
            "I could not find an explicit count in the top passages. "
            f"Try checking chapter overview/table-of-contents areas near page {top_page}."
        )
    return "I could not find an explicit count in the retrieved passages."


def _pick_summary_answer(query: str, sentences: list[str]) -> str:
    ranked = _rank_sentences_extractively(query, sentences)
    if not ranked:
        return "No summary could be formed from the retrieved passages."

    summary_sents = [ranked[0][1]]
    for _, sentence in ranked[1:]:
        # Keep concise and avoid near-duplicate sentences.
        if sentence.lower() in {s.lower() for s in summary_sents}:
            continue
        summary_sents.append(sentence)
        if len(summary_sents) >= 3:
            break
    return " ".join(summary_sents)


def build_short_answer(query: str, results: list[dict]) -> str:
    if not results:
        return "No relevant answer was found."

    candidates = _collect_candidate_sentences(results)
    if not candidates:
        fallback = _clean_text(results[0].get("text", ""))
        return fallback[:280] + ("..." if len(fallback) > 280 else "")

    ranked = _rank_sentences_extractively(query, candidates)
    q_type = classify_question_type(query)

    if q_type == "definition":
        answer = _pick_definition_answer(query, candidates)
    elif q_type == "location":
        answer = _pick_location_answer(query, results, candidates)
    elif q_type == "count":
        answer = _pick_count_answer(query, results, candidates)
    elif q_type == "summary":
        answer = _pick_summary_answer(query, candidates)
    else:
        if not ranked:
            best = candidates[0]
            return best[:280] + ("..." if len(best) > 280 else "")
        best_score, best_sentence = ranked[0]
        selected = [best_sentence]
        if len(ranked) > 1:
            second_score, second_sentence = ranked[1]
            if second_score >= 0.65 * best_score and second_score >= 0.08:
                selected.append(second_sentence)
        answer = " ".join(selected).strip()

    if not answer:
        answer = candidates[0]
    if len(answer) > 320:
        answer = answer[:317].rstrip() + "..."
    return answer