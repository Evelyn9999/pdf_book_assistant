from pypdf import PdfReader

def read_pdf(file_path: str) -> list[dict]:
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        pages.append({
            "page": i,
            "text": text
        })

    return pages