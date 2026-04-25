def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    chunks = []
    chunk_id = 0

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]

        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "text": chunk_text
                })
                chunk_id += 1

            start += chunk_size - overlap

    return chunks