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


def _rank_definition_sentences(query: str, sentences: list[str]) -> list[tuple[float, str]]:
    base_ranked = _rank_sentences_extractively(query, sentences)
    if not base_ranked:
        return []

    subject = _extract_question_subject(query)
    subject_terms = _tokenize_for_match(subject if subject else query)
    subject_text = subject.lower().strip() if subject else ""

    weighted: list[tuple[float, str]] = []
    for base_score, sentence in base_ranked:
        lower = sentence.lower()
        sent_terms = _tokenize_for_match(sentence)
        overlap = len(sent_terms & subject_terms) / max(1, len(subject_terms))
        bonus = 0.0

        if subject_text and subject_text in lower:
            bonus += 0.35
        if re.search(r"\b(is|are)\b", lower):
            bonus += 0.2
        if " refers to " in lower or "defined as" in lower:
            bonus += 0.3
        if re.search(r"^(an?\s+)?[a-z0-9 _-]+\s+is\b", lower):
            bonus += 0.2
        if overlap >= 0.6:
            bonus += 0.2

        weighted.append((base_score + bonus, sentence))

    weighted.sort(key=lambda x: x[0], reverse=True)
    return weighted


def _pick_definition_answer(query: str, sentences: list[str]) -> str:
    ranked = _rank_definition_sentences(query, sentences)
    return ranked[0][1] if ranked else ""


def _extract_chapter_title(text: str) -> str:
    patterns = [
        r"\bchapter\s+\d+\s*[:\-]?\s*([a-zA-Z][^.;\n]{4,120})",
        r"\bch\.\s*\d+\s*[:\-]?\s*([a-zA-Z][^.;\n]{4,120})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" -:;,.")
    return ""


def _pick_location_answer(query: str, results: list[dict], sentences: list[str]) -> str:
    if not results:
        return "No relevant location was found."

    ranked = _rank_sentences_extractively(query, sentences)
    best_sentence = ranked[0][1] if ranked else _clean_text(results[0].get("text", ""))[:180]

    top_result = results[0]
    page = top_result.get("page", "unknown")
    chapter_num = top_result.get("chapter_number")
    chapter_title = (top_result.get("chapter_title") or "").strip()
    section_num = (top_result.get("section_number") or "").strip()
    section_title = (top_result.get("section_title") or "").strip()
    title = _extract_chapter_title(top_result.get("text", "")) or _extract_chapter_title(best_sentence)

    if chapter_num and chapter_title:
        chapter_label = f"Chapter {chapter_num}: {chapter_title}"
    elif chapter_num:
        chapter_label = f"Chapter {chapter_num}"
    elif chapter_title:
        chapter_label = chapter_title
    elif title:
        chapter_label = title
    else:
        chapter_label = "Unknown chapter"

    if section_num and section_title:
        section_label = f"{section_num} {section_title}"
    elif section_num:
        section_label = section_num
    else:
        section_label = "N/A"

    snippet = best_sentence.strip()
    return f"Chapter: {chapter_label} | Section: {section_label} | Page: {page} | Snippet: {snippet}"


def _word_to_int(token: str) -> int | None:
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return number_words.get(token)


def _extract_count_evidence(sentences: list[str]) -> tuple[list[int], list[str]]:
    values: list[int] = []
    evidence: list[str] = []
    count_patterns = [
        r"\b(?:first|last|next)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:chapters?|parts?)\b",
    ]

    for sentence in sentences:
        lower = sentence.lower()
        if "chapter" not in lower and "part" not in lower:
            continue
        found = False
        for pattern in count_patterns:
            for match in re.findall(pattern, lower):
                value = _word_to_int(match)
                if value is not None:
                    values.append(value)
                    found = True
        if found:
            evidence.append(sentence)

    return values, evidence


def _pick_count_answer(query: str, results: list[dict], sentences: list[str]) -> str:
    chapter_numbers = sorted(
        {
            int(item["chapter_number"])
            for item in results
            if item.get("chapter_number") is not None and str(item.get("chapter_number")).isdigit()
        }
    )
    if "chapter" in query.lower() and chapter_numbers:
        return (
            f"From the retrieved structured headings, I can identify chapters: {', '.join(map(str, chapter_numbers))}. "
            f"Current evidence indicates at least {len(chapter_numbers)} chapter(s)."
        )

    values, evidence = _extract_count_evidence(sentences)
    if len(values) >= 2:
        total = sum(values[:2])
        return f"{evidence[0]} {evidence[1]} This suggests a total of about {total}."

    if len(values) == 1:
        return f"{evidence[0]} This indicates a count of about {values[0]}."

    # Fallback: choose sentence with most numeric evidence.
    candidates = []
    for sentence in sentences:
        score = len(re.findall(r"\b\d+\b", sentence))
        if score > 0:
            candidates.append((score, sentence))
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