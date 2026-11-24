"""Step 1 - Ingest: turn an uploaded file into text and/or page images.

Text-based PDFs and DOCX files give us text directly (cheap, exact). Scanned
PDFs and photos have no text layer, so we pass page images to the vision model.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz

MIN_CHARS_PER_PAGE = 40
RENDER_DPI = 150
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


@dataclass
class IngestedDocument:
    file_name: str
    kind: str  # "pdf-text" | "pdf-scan" | "pdf-garbled" | "docx" | "image" | "text"
    text: str = ""
    images: list[tuple[str, bytes]] = field(default_factory=list)  # (media_type, bytes)
    pages: int = 1

    @property
    def needs_vision(self) -> bool:
        return not self.text.strip() and bool(self.images)


def ingest_bytes(data: bytes, file_name: str) -> IngestedDocument:
    ext = Path(file_name).suffix.lower()
    if ext == ".pdf":
        return _ingest_pdf(data, file_name)
    if ext == ".docx":
        return _ingest_docx(data, file_name)
    if ext in IMAGE_TYPES:
        return IngestedDocument(file_name, "image", images=[(IMAGE_TYPES[ext], data)])
    if ext in (".txt", ".md"):
        return IngestedDocument(file_name, "text", text=data.decode("utf-8", errors="replace"))
    raise ValueError(f"Unsupported file type: {ext} ({file_name})")


def ingest_path(path: str | Path) -> IngestedDocument:
    p = Path(path)
    return ingest_bytes(p.read_bytes(), p.name)


def clean_text(text: str) -> str:
    """Fix common artefacts of Thai PDF text layers."""
    text = text.replace("ํา", "ำ")          # นิคหิต + สระอา -> สระอำ
    text = text.replace(" ", " ").replace("­", "-")  # nbsp / soft hyphen from some fonts
    return text


def looks_garbled(text: str) -> bool:
    """True when the text layer is likely broken (private-use or stray glyphs between Thai letters).

    Common with Thai fonts whose shaped glyphs lack proper Unicode mappings; such
    files are better read from page images.
    """
    if not text:
        return False
    pua = sum(1 for c in text if 0xE000 <= ord(c) <= 0xF8FF)
    stray = len(re.findall(r"[฀-๿][Ā-ӿ]", text))
    return (pua + stray) / max(1, len(text)) > 0.002


def _ingest_pdf(data: bytes, file_name: str) -> IngestedDocument:
    with fitz.open(stream=data, filetype="pdf") as doc:
        texts = [clean_text(page.get_text("text")) for page in doc]
        pages = len(doc)
        text = "\n".join(texts).strip()
        has_text = sum(len(t.strip()) for t in texts) >= MIN_CHARS_PER_PAGE * pages
        if has_text and not looks_garbled(text):
            return IngestedDocument(file_name, "pdf-text", text=text, pages=pages)
        images = [("image/png", page.get_pixmap(dpi=RENDER_DPI).tobytes("png")) for page in doc]
    return IngestedDocument(file_name, "pdf-scan" if not has_text else "pdf-garbled", images=images, pages=pages)


def _ingest_docx(data: bytes, file_name: str) -> IngestedDocument:
    import docx  # python-docx

    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(dict.fromkeys(cells)))  # merged cells repeat
    return IngestedDocument(file_name, "docx", text="\n".join(parts))
