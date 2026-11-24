"""Free OCR for scanned CVs: Tesseract with the Thai + English models.

Install the engine (not a pip package):
  Colab / Ubuntu:  apt-get install -y tesseract-ocr tesseract-ocr-tha
  Windows:         https://github.com/UB-Mannheim/tesseract/wiki (tick "Thai" in additional languages)
"""
from __future__ import annotations

import io
import os
import shutil
from functools import lru_cache

from PIL import Image, ImageOps

LANGS = os.getenv("TALENTLENS_OCR_LANGS", "tha+eng")


class OCRUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def available() -> bool:
    try:
        import pytesseract

        if not shutil.which("tesseract") and os.name == "nt":
            default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(default):
                pytesseract.pytesseract.tesseract_cmd = default
        return "tha" in pytesseract.get_languages(config="")
    except Exception:
        return False


def _prepare(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("L")
    if img.width < 1800:  # Thai marks are small; upscale low-resolution scans
        scale = 1800 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(img)


def ocr_images(images: list[tuple[str, bytes]]) -> str:
    if not available():
        raise OCRUnavailable("ไม่พบ Tesseract ที่มีภาษาไทย (tesseract-ocr-tha) - อ่านภาพสแกนไม่ได้")
    import pytesseract

    pages = [pytesseract.image_to_string(_prepare(data), lang=LANGS, config="--psm 4") for _, data in images]
    text = "\n".join(pages)
    return text.replace("\u0e4d\u0e32", "\u0e33").strip()
