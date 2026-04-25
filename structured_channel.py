import re


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower()))


def _score_chunk_for_topic(query: str, chunk: dict) -> float:
    query_terms = _tokenize(query)
    if not query_terms:
        return 0.0

    title_blob = " ".join(
        [
            str(chunk.get("chapter_title", "") or ""),
            str(chunk.get("section_title", "") or ""),
        ]
    )
    text_blob = str(chunk.get("text", "") or "")

    title_terms = _tokenize(title_blob)
    text_terms = _tokenize(text_blob)
    title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
    text_overlap = len(query_terms & text_terms) / max(1, len(query_terms))
    return (0.7 * title_overlap) + (0.3 * text_overlap)


def _chapter_label(chunk: dict) -> str:
    ch_num = chunk.get("chapter_number")
    ch_title = (chunk.get("chapter_title") or "").strip()
    if ch_num and ch_title:
        return f"Chapter {ch_num}, \"{ch_title}\""
    if ch_num:
        return f"Chapter {ch_num}"
    if ch_title:
        return ch_title
    return "Unknown chapter"


def _section_label(chunk: dict) -> str:
    sec_num = (chunk.get("section_number") or "").strip()
    sec_title = (chunk.get("section_title") or "").strip()
    if sec_num and sec_title:
        return f"{sec_num} {sec_title}"
    if sec_num:
        return sec_num
    if sec_title:
        return sec_title
    return ""


def _handle_count(query: str, chunks: list[dict], chapter_index: list[dict] | None) -> tuple[bool, str, str, list[dict]]:
    q = query.lower()
    chapter_nums = sorted({int(item["chapter_number"]) for item in (chapter_index or [])})

    if "how many chapters" in q or "number of chapters" in q:
        if not chapter_nums:
            return True, "I could not detect chapter headings reliably from extracted structure.", "structured:no_chapter_numbers", []
        answer = (
            f"By scanning chapter headings, the book appears to have {len(chapter_nums)} chapter(s) "
            f"({', '.join(map(str, chapter_nums))})."
        )
        evidence = []
        for item in (chapter_index or [])[:3]:
            evidence.append(
                {
                    "page": item.get("page"),
                    "chapter_number": item.get("chapter_number"),
                    "chapter_title": item.get("title", ""),
                    "section_number": "",
                    "section_title": "",
                    "text": f"Chapter {item.get('chapter_number')}: {item.get('title', '')}",
                    "score": 1.0,
                }
            )
        return True, answer, "structured:chapter_count", evidence

    chapter_match = re.search(r"\bchapter\s+(\d+)\b", q)
    if chapter_match and ("how many sections" in q or "number of sections" in q):
        target = int(chapter_match.group(1))
        section_nums = sorted(
            {
                str(c.get("section_number"))
                for c in chunks
                if c.get("chapter_number") == target and c.get("section_number")
            }
        )
        if not section_nums:
            return True, f"I could not find section-number headings for chapter {target}.", "structured:no_section_numbers", []
        answer = f"Chapter {target} has about {len(section_nums)} section(s): {', '.join(section_nums)}."
        evidence = [c for c in chunks if c.get("chapter_number") == target and c.get("section_number")][:3]
        return True, answer, "structured:section_count", evidence

    return False, "", "", []


def _handle_location(query: str, chunks: list[dict]) -> tuple[bool, str, str, list[dict]]:
    scored = [(_score_chunk_for_topic(query, c), c) for c in chunks]
    scored = [item for item in scored if item[0] > 0]
    if not scored:
        return True, "I could not find a reliable chapter/section match for this topic.", "structured:no_location_match", []
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0][1]
    page = top.get("page", "unknown")
    chapter = _chapter_label(top)
    section = _section_label(top)
    if section:
        answer = f"The topic is discussed in {chapter}, especially in section {section}, around page {page}."
    else:
        answer = f"The topic is discussed in {chapter}, around page {page}."
    evidence = [item[1] for item in scored[:3]]
    return True, answer, "structured:location", evidence


def _handle_overview(query: str, chunks: list[dict]) -> tuple[bool, str, str, list[dict]]:
    cue_patterns = [r"\bin this book\b", r"\bthis chapter\b", r"\boverview\b", r"\bintroduction\b", r"\bpreface\b"]
    candidates = []
    for chunk in chunks:
        title_blob = " ".join(
            [str(chunk.get("chapter_title", "") or ""), str(chunk.get("section_title", "") or "")]
        ).lower()
        text = str(chunk.get("text", "") or "")
        title_hit = any(k in title_blob for k in ["overview", "preface", "introduction", "summary"])
        cue_hit = any(re.search(p, text.lower()) for p in cue_patterns)
        if title_hit or cue_hit:
            candidates.append(chunk)
    if not candidates:
        return True, "I could not find a clear preface/introduction-style overview passage.", "structured:no_overview_match", []
    ranked = sorted(candidates, key=lambda c: _score_chunk_for_topic(query, c), reverse=True)
    top = ranked[0]
    snippet = re.split(r"(?<=[.!?])\s+", str(top.get("text", "") or "").strip())[0][:220]
    answer = f"Book overview evidence appears in {_chapter_label(top)} (page {top.get('page', 'unknown')}): {snippet}"
    return True, answer, "structured:overview", ranked[:3]


def run_structured_channel(
    query: str, question_type: str, chunks: list[dict], chapter_index: list[dict] | None = None
) -> tuple[bool, str, str, list[dict]]:
    if not chunks:
        return False, "", "", []
    if question_type == "count":
        return _handle_count(query, chunks, chapter_index)
    if question_type == "location":
        return _handle_location(query, chunks)
    if question_type == "overview":
        return _handle_overview(query, chunks)
    return False, "", "", []
