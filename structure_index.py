import re


def _looks_like_title(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if len(text) < 4 or len(text) > 120:
        return False
    if re.search(r"\b\d{1,2}[:/]\d{1,2}[:/]\d{2,4}\b", text):
        return False
    if re.search(r"\.(?:indd|pdf|docx?)\b", text, flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"\d+(\.\d+)*", text):
        return False
    return True


def extract_chapter_index(page_texts: list[str]) -> list[dict]:
    chapter_map: dict[int, dict] = {}

    for page_num, raw_text in enumerate(page_texts, start=1):
        lines = [line.strip() for line in (raw_text or "").replace("\r", "\n").split("\n") if line.strip()]
        for i, line in enumerate(lines):
            # Strict heading: line starts with Chapter/CHAPTER and 1-2 digit number.
            m = re.match(r"^\s*(?:chapter|CHAPTER)\s+(\d{1,2})(?:\s*[:\-]\s*(.+))?\s*$", line)
            if not m:
                continue

            chapter_number = int(m.group(1))
            if not (1 <= chapter_number <= 50):
                continue

            title = (m.group(2) or "").strip(" -:;,.")
            if not title and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if _looks_like_title(next_line):
                    title = next_line.strip(" -:;,.")

            if chapter_number not in chapter_map:
                chapter_map[chapter_number] = {
                    "chapter_number": chapter_number,
                    "title": title,
                    "page": page_num,
                }

    chapter_index = sorted(chapter_map.values(), key=lambda x: x["chapter_number"])
    return chapter_index
