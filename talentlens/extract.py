"""Step 2 - Extract: CV -> CandidateProfile (+ transcript).

Extractors share one output type:
* ``extract_llm(doc, call)`` - a language model; ``call`` is ``llm.structured_call``
  (Claude, reads images too) or ``local_llm.structured_call`` (open-source via Ollama, text only)
* ``extract_rules``          - regex + lexicon, no model at all (baseline)
"""
from __future__ import annotations

import re
from datetime import date

from . import llm
from .ingest import IngestedDocument
from .lexicon import SKILLS, find_skills
from .schemas import CandidateProfile, Education, Experience, ExtractionResult, Skill

SYSTEM_PROMPT = """You extract structured data from job applicants' CVs/resumes for an HR screening tool.
CVs may be in Thai, English, or both, and may be scanned images.

Rules:
- Copy facts only from the CV. Never guess values that are not there; use null or an empty list.
- Convert Buddhist-era years (พ.ศ., e.g. 2565) to Gregorian (subtract 543). Dates as YYYY-MM.
  If only a year is given, use YYYY-01 for a start and YYYY-12 for an end.
- A current job ("ปัจจุบัน", "present", "now") has end = null.
- For each skill, give an exact short quote from the CV as evidence.
- If asked for a `transcript`, give a faithful plain-text transcription of the whole CV in reading order.
- The text may come from OCR and contain small errors; fix obvious OCR mistakes in names of skills only.
- List any field you could not read confidently in `low_confidence_fields`."""


def extract_llm(doc: IngestedDocument, call=None) -> tuple[ExtractionResult, llm.Usage]:
    call = call or llm.structured_call
    if doc.needs_vision:  # images -> the model reads them and transcribes (Claude only)
        content = [llm.image_block(mt, data) for mt, data in doc.images]
        content.append({"type": "text", "text": f"The images above are the pages of CV file '{doc.file_name}'. Extract the profile and transcribe the CV."})
        return call(SYSTEM_PROMPT, content, ExtractionResult)
    # text layer (or OCR) available -> ask for the profile only; the text itself is the transcript
    content = [{"type": "text", "text": f"<cv file='{doc.file_name}'>\n{doc.text}\n</cv>\n\nExtract the profile."}]
    prof, usage = call(SYSTEM_PROMPT, content, CandidateProfile)
    return ExtractionResult(profile=prof, transcript=doc.text), usage


# ------------------------------------------------------------ rule-based path
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+66\s?|0)\d{1,2}[-\s]?\d{3}[-\s]?\d{4}")
MONTHS = {m: i + 1 for i, m in enumerate(["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."])}
MONTHS.update({m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])})
DATE = r"(?:[A-Za-zก-๙.]{3,9}\s*\d{4}|\d{1,2}/\d{4}|\d{4}(?:-\d{1,2})?)"
RANGE_RE = re.compile(rf"({DATE})\s*(?:-|–|—|to|ถึง)\s*({DATE}|ปัจจุบัน|present|now|current)", re.I)
PRESENT = {"ปัจจุบัน", "present", "now", "current"}
EDU_RE = re.compile(r"(ปริญญา|บัณฑิต|bachelor|master|b\.?sc|b\.?a\.?|bba|m\.?sc|mba|ph\.?d|ปวส|ปวช)", re.I)
CERT_RE = re.compile(r"(toeic\s*:?\s*\d{3}|ielts\s*:?\s*[\d.]+|toefl\s*:?\s*\d+|\bcpa\b|ผู้สอบบัญชีรับอนุญาต|ผู้สอบบัญชีภาษีอากร|tax auditor|\bta\b)", re.I)
TITLE_GENDER = {"นางสาว": "female", "นาง": "female", "นาย": "male", "ms.": "female", "mrs.": "female", "miss": "female", "mr.": "male"}


def _gregorian(y: int) -> int:
    return y - 543 if y > 2400 else y


def parse_date(s: str, end: bool = False) -> str | None:
    """'ก.ค. 2569' | 'Jun 2021' | '06/2564' | '2021-06' | '2021' -> 'YYYY-MM' (None = present)."""
    s = s.strip()
    if s.lower() in PRESENT:
        return None
    y = int(re.search(r"\d{4}", s).group(0))
    m = None
    if mm := re.match(r"(\d{1,2})/\d{4}", s):
        m = int(mm.group(1))
    elif mm := re.match(r"\d{4}-(\d{1,2})", s):
        m = int(mm.group(1))
    else:
        word = re.match(r"([A-Za-zก-๙.]+)", s)
        if word:
            m = MONTHS.get(word.group(1).lower()[:3]) or MONTHS.get(word.group(1))
    return f"{_gregorian(y):04d}-{m or (12 if end else 1):02d}"


def _role_parts(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\s*(?:\||@| ที่ | at )\s*", text) if p.strip()]


def extract_rules(doc: IngestedDocument) -> ExtractionResult:
    if doc.needs_vision:
        raise ValueError("ไฟล์สแกน/รูปภาพต้องใช้โหมด Claude API (rule-based อ่านได้เฉพาะไฟล์ที่มีข้อความ)")
    text = doc.text
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    name = lines[0] if lines else None
    gender = None
    if name:
        low = name.lower()
        for title, g in TITLE_GENDER.items():
            if low.startswith(title):
                gender = g
                break

    birth_year = None
    m = re.search(r"(?:เกิด|ปีเกิด|date of birth|born|dob)\D{0,20}(\d{4})", text, re.I)
    if m:
        birth_year = _gregorian(int(m.group(1)))
    else:
        m = re.search(r"(?:อายุ|age)\s*:?\s*(\d{2})", text, re.I)
        if m:
            birth_year = date.today().year - int(m.group(1))

    experience = []
    education = []
    for i, line in enumerate(lines):
        r = RANGE_RE.search(line)
        if r and not EDU_RE.search(line):
            start, end = parse_date(r.group(1)), parse_date(r.group(2), end=True)
            rest = (line[: r.start()] + " " + line[r.end():]).strip(" |,:-–")
            if "|" not in rest:  # title/org on a neighbouring line
                prev = lines[i - 1] if i else ""
                nxt = lines[i + 1] if i + 1 < len(lines) else ""
                rest = prev if "|" in prev and not RANGE_RE.search(prev) else nxt if "|" in nxt else rest
            parts = _role_parts(rest)
            experience.append(Experience(
                title=parts[0] if parts else rest,
                organization=parts[1] if len(parts) > 1 else "",
                start=start, end=end, highlights=[],
            ))
        elif EDU_RE.search(line):
            years = [int(y) for y in re.findall(r"(\d{4})", line)]
            parts = [p.strip() for p in re.split(r"\s*(?:\||,)\s*", line) if p.strip()]
            education.append(Education(
                degree=parts[0], field=parts[1] if len(parts) > 1 else "",
                institution=parts[2] if len(parts) > 2 else "",
                year=_gregorian(max(years)) if years else None,
            ))

    skill_ids = find_skills(text)
    skills = []
    for sid in skill_ids:
        quote = next((l for l in lines if any(f.strip() in l.lower() for f in SKILLS[sid])), sid)
        skills.append(Skill(name=sid, evidence=quote))

    certs = sorted({c.group(0).strip() for c in CERT_RE.finditer(text)})
    langs = [l for l in ("ไทย", "อังกฤษ", "Thai", "English", "จีน", "Chinese", "ญี่ปุ่น", "Japanese") if l in text]

    profile = CandidateProfile(
        full_name=name, email=(EMAIL_RE.search(text) or [None])[0], phone=(PHONE_RE.search(text) or [None])[0],
        gender=gender, birth_year=birth_year, has_photo=False, summary=None,
        education=education, experience=experience, skills=skills,
        certifications=certs, languages=langs, low_confidence_fields=[],
    )
    return ExtractionResult(profile=profile, transcript=text)
