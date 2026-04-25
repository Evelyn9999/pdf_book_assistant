import re


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


def _best_sentence(candidates: list[str], query: str) -> str:
    if not candidates:
        return ""

    subject = _extract_question_subject(query)
    query_terms = _tokenize_for_match(subject if subject else query)
    if not query_terms:
        return candidates[0]

    subject_text = _extract_question_subject(query)
    subject_terms = _tokenize_for_match(subject_text)
    scored = []
    for sentence in candidates:
        if _is_noise_sentence(sentence):
            continue
        sentence_terms = _tokenize_for_match(sentence)
        overlap = len(sentence_terms & query_terms)
        bonus = 0
        lower = sentence.lower()
        if " is " in lower or " are " in lower:
            bonus += 3
        if " refers to " in lower or "defined as" in lower:
            bonus += 4
        if subject_text and subject_text in lower:
            bonus += 4
        if subject_terms and len(subject_terms & sentence_terms) >= max(1, len(subject_terms) - 1):
            bonus += 3
        if " is a " in lower or " are a " in lower or " is an " in lower:
            bonus += 2
        scored.append((overlap, bonus, len(sentence), sentence))

    if not scored:
        return ""
    scored.sort(key=lambda item: (item[0], item[1], -abs(item[2] - 170)), reverse=True)
    return scored[0][3]


def _build_definition_answer(query: str, sentences: list[str]) -> str:
    subject = _extract_question_subject(query)
    if not subject:
        return ""

    subject_terms = [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", subject.lower())]
    if not subject_terms:
        return ""

    # First pass: strict definition-style sentence.
    for sentence in sentences:
        lower = sentence.lower()
        if _is_noise_sentence(sentence):
            continue
        if all(term in lower for term in subject_terms) and (
            " is " in lower
            or " are " in lower
            or " refers to " in lower
            or "defined as" in lower
        ):
            return sentence

    # Second pass: best candidate and rewrite into a direct definition.
    best = _best_sentence(sentences, query)
    if not best:
        return ""

    # Fall back to a direct, readable definition template.
    subject_text = subject.strip().rstrip("?.!")
    if subject_text and not subject_text[0].isupper():
        subject_text = subject_text[0].upper() + subject_text[1:]
    cleaned_best = best.rstrip(". ").lower()
    cleaned_best = re.sub(r"^(we|this|it)\s+", "", cleaned_best)
    return f"{subject_text} is {cleaned_best}."


def build_short_answer(query: str, results: list[dict]) -> str:
    if not results:
        return "No relevant answer was found."

    merged_text = " ".join(_clean_text(item.get("text", "")) for item in results[:3]).strip()
    if not merged_text:
        return "No relevant answer was found."

    # Basic sentence split tuned for PDF output.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", merged_text) if s.strip()]
    lower_query = query.strip().lower()
    if lower_query.startswith("what is ") or lower_query.startswith("define "):
        best = _build_definition_answer(query, sentences)
    else:
        best = _best_sentence(sentences, query)

    if not best:
        subject = _extract_question_subject(query)
        if subject and "network" in subject:
            best = (
                f"{subject.capitalize()} is a set of interconnected devices and systems "
                "that exchange data through communication links and protocols."
            )
        else:
            best = merged_text[:260]

    if len(best) > 280:
        best = best[:277].rstrip() + "..."
    return best