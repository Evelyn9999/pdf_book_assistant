import re
from collections import Counter


def _normalize_line_for_repeat_check(line: str) -> str:
    line = line.strip().lower()
    line = re.sub(r"\s+", " ", line)
    return line


def _is_print_metadata_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True

    # Standalone page number lines.
    if re.fullmatch(r"\d{1,4}", stripped):
        return True

    # File names / layout tool markers.
    if re.search(r"\.(?:indd|pdf|docx?)\b", stripped, flags=re.IGNORECASE):
        return True
    if re.search(r"\bM\d{2}_[A-Z0-9_]+\b", stripped):
        return True

    # Date/time printing metadata.
    if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", stripped):
        return True
    if re.search(r"\b\d{1,2}:\d{2}\s?(?:AM|PM)\b", stripped, flags=re.IGNORECASE):
        return True

    # Lines with almost no letters are usually non-content footer/header artifacts.
    letters = len(re.findall(r"[A-Za-z]", stripped))
    if letters <= 2 and len(stripped) <= 12:
        return True

    return False


def _remove_repeated_header_footer_lines(pages_lines: list[list[str]]) -> list[list[str]]:
    if not pages_lines:
        return pages_lines

    edge_counter: Counter[str] = Counter()
    for lines in pages_lines:
        if not lines:
            continue
        edge_candidates = lines[:2] + lines[-2:]
        for line in edge_candidates:
            key = _normalize_line_for_repeat_check(line)
            if key:
                edge_counter[key] += 1

    # Repeated edge lines across many pages are likely headers/footers.
    threshold = max(3, int(len(pages_lines) * 0.2))
    repeated_keys = {key for key, count in edge_counter.items() if count >= threshold}
    if not repeated_keys:
        return pages_lines

    cleaned_pages = []
    for lines in pages_lines:
        cleaned_pages.append(
            [line for line in lines if _normalize_line_for_repeat_check(line) not in repeated_keys]
        )
    return cleaned_pages


def clean_pdf_pages(page_texts: list[str]) -> list[str]:
    cleaned_lines_per_page: list[list[str]] = []

    for raw_text in page_texts:
        text = raw_text or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Merge hyphenated words split by line breaks: net-\nwork -> network.
        text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

        # Normalize intra-line whitespace first.
        text = re.sub(r"[ \t\f\v]+", " ", text)

        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if _is_print_metadata_line(line):
                continue
            lines.append(line)

        cleaned_lines_per_page.append(lines)

    cleaned_lines_per_page = _remove_repeated_header_footer_lines(cleaned_lines_per_page)

    cleaned_pages = []
    for lines in cleaned_lines_per_page:
        # Rebuild lightweight paragraph boundaries before chunking.
        paragraphs = []
        current = []
        for line in lines:
            current.append(line)
            if re.search(r"[.!?]$", line) and len(current) >= 2:
                paragraphs.append(" ".join(current))
                current = []

        if current:
            paragraphs.append(" ".join(current))

        normalized_paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip()]
        page_text = "\n\n".join(normalized_paragraphs).strip()
        cleaned_pages.append(page_text)

    return cleaned_pages
