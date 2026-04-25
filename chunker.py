import re


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _last_n_words(text: str, n_words: int) -> str:
    if n_words <= 0:
        return ""
    words = re.findall(r"\S+", text)
    if not words:
        return ""
    return " ".join(words[-n_words:])


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if parts:
        return parts
    # Fallback for text without paragraph separators.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _split_long_paragraph(paragraph: str, max_words: int, overlap_words: int) -> list[str]:
    words = re.findall(r"\S+", paragraph)
    if len(words) <= max_words:
        return [paragraph]

    windows = []
    step = max(1, max_words - overlap_words)
    start = 0
    while start < len(words):
        end = start + max_words
        windows.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return windows


def _extract_structure_from_heading(line: str) -> dict:
    text = line.strip()
    info = {
        "chapter_number": None,
        "chapter_title": "",
        "section_number": "",
        "section_title": "",
    }

    # CHAPTER 1 / Chapter 1: Computer Networks
    chapter_match = re.search(
        r"\bchapter\s+(\d{1,2})\b(?:\s*[:\-]?\s*(.*))?$",
        text,
        flags=re.IGNORECASE,
    )
    if chapter_match:
        info["chapter_number"] = int(chapter_match.group(1))
        title = (chapter_match.group(2) or "").strip(" -:;,.")
        if title:
            info["chapter_title"] = title
        return info

    # 1.1 What Is the Internet?
    section_match = re.match(r"^(\d+\.\d+(?:\.\d+)*)\s+(.+)$", text)
    if section_match:
        info["section_number"] = section_match.group(1).strip()
        info["section_title"] = section_match.group(2).strip(" -:;,.")
        chapter_part = info["section_number"].split(".")[0]
        if chapter_part.isdigit():
            info["chapter_number"] = int(chapter_part)
        return info

    return info


def _infer_page_structure(text: str, state: dict) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chapter_number = state.get("chapter_number")
    chapter_title = state.get("chapter_title", "")
    section_number = state.get("section_number", "")
    section_title = state.get("section_title", "")

    for line in lines[:8]:
        info = _extract_structure_from_heading(line)
        if info["chapter_number"] is not None:
            chapter_number = info["chapter_number"]
        if info["chapter_title"]:
            chapter_title = info["chapter_title"]
        if info["section_number"]:
            section_number = info["section_number"]
        if info["section_title"]:
            section_title = info["section_title"]

    return {
        "chapter_number": chapter_number,
        "chapter_title": chapter_title,
        "section_number": section_number,
        "section_title": section_title,
    }


def chunk_pages(
    pages: list[dict],
    min_words: int = 400,
    max_words: int = 800,
    overlap_words: int = 75,
) -> list[dict]:
    chunks = []
    chunk_id = 0
    overlap_words = max(0, overlap_words)
    max_words = max(50, max_words)
    min_words = max(50, min(min_words, max_words))

    structure_state = {
        "chapter_number": None,
        "chapter_title": "",
        "section_number": "",
        "section_title": "",
    }

    for page_data in pages:
        page_num = page_data["page"]
        text = (page_data.get("text") or "").strip()
        if not text:
            continue

        page_structure = _infer_page_structure(text, structure_state)
        structure_state = page_structure.copy()

        paragraphs = _split_paragraphs(text)
        units = []
        for paragraph in paragraphs:
            units.extend(_split_long_paragraph(paragraph, max_words=max_words, overlap_words=overlap_words))

        current = ""
        for unit in units:
            candidate = unit if not current else f"{current}\n\n{unit}"
            candidate_words = _count_words(candidate)

            if candidate_words <= max_words:
                current = candidate
                continue

            if current:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_num,
                        "text": current.strip(),
                        "chapter_number": page_structure["chapter_number"],
                        "chapter_title": page_structure["chapter_title"],
                        "section_number": page_structure["section_number"],
                        "section_title": page_structure["section_title"],
                    }
                )
                chunk_id += 1
                tail = _last_n_words(current, overlap_words)
                current = f"{tail} {unit}".strip() if tail else unit
            else:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_num,
                        "text": unit.strip(),
                        "chapter_number": page_structure["chapter_number"],
                        "chapter_title": page_structure["chapter_title"],
                        "section_number": page_structure["section_number"],
                        "section_title": page_structure["section_title"],
                    }
                )
                chunk_id += 1
                current = ""

            # If overlap + unit still too long, flush immediately.
            if _count_words(current) > max_words:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_num,
                        "text": current.strip(),
                        "chapter_number": page_structure["chapter_number"],
                        "chapter_title": page_structure["chapter_title"],
                        "section_number": page_structure["section_number"],
                        "section_title": page_structure["section_title"],
                    }
                )
                chunk_id += 1
                current = _last_n_words(current, overlap_words)

        if current:
            if chunks and _count_words(current) < min_words:
                # Merge tiny remainder into previous chunk when possible.
                prev_text = chunks[-1]["text"]
                merged = f"{prev_text}\n\n{current}".strip()
                if _count_words(merged) <= (max_words + min_words // 2):
                    chunks[-1]["text"] = merged
                    continue
            chunks.append({"chunk_id": chunk_id, "page": page_num, "text": current.strip()})
            chunks[-1]["chapter_number"] = page_structure["chapter_number"]
            chunks[-1]["chapter_title"] = page_structure["chapter_title"]
            chunks[-1]["section_number"] = page_structure["section_number"]
            chunks[-1]["section_title"] = page_structure["section_title"]
            chunk_id += 1

    return chunks