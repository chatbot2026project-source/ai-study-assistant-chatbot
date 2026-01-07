import os

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


def extract_text_from_pdf(pdf_path):
    if pdfplumber is None:
        return ""

    if not os.path.exists(pdf_path):
        return ""

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=120):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks
