"""Step 3 - Redact (blind screening): hide identity attributes before scoring.

Hidden: name, contact details, national ID, gender/title words, birth date/age,
photo references and institution names. Skills and experience stay visible.
"""
from __future__ import annotations

import re

from .schemas import CandidateProfile

MASK = "[ปิดบัง]"

PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),                                    # email
    re.compile(r"(?:\+66\s?|0)\d{1,2}[-\s]?\d{3}[-\s]?\d{4}"),                 # phone
    re.compile(r"\b\d[-\s]?\d{4}[-\s]?\d{5}[-\s]?\d{2}[-\s]?\d\b"),            # Thai national ID
    re.compile(r"(?:วันเกิด|เกิด|ปีเกิด|date of birth|dob|born)\s*:?[^\n|]*", re.I),   # birth date
    re.compile(r"(?:อายุ|age)\s*:?\s*\d{2}\s*(?:ปี|years?(?: old)?)?", re.I),
    re.compile(r"(?:เพศ|gender|sex)\s*:?\s*\S+", re.I),
    re.compile(r"(?:สถานภาพ|marital status)\s*:?\s*\S+", re.I),
    re.compile(r"(?:ศาสนา|religion|สัญชาติ|nationality)\s*:?\s*\S+", re.I),
    re.compile(r"(?<![ก-๙])(?:นางสาว|นาง|นาย)(?=[ก-๙])"),                       # Thai titles
    re.compile(r"\b(?:Mr|Mrs|Ms|Miss)\.?\s", re.I),
]


def redact_text(text: str, profile: CandidateProfile | None = None) -> str:
    out = text
    if profile:
        names = []
        if profile.full_name:
            names += [profile.full_name] + [p for p in re.split(r"\s+", profile.full_name) if len(p) > 1]
        names += [e.institution for e in profile.education if e.institution and len(e.institution) > 3]
        for n in sorted(set(names), key=len, reverse=True):
            out = re.sub(re.escape(n), MASK, out, flags=re.I)
    for pat in PATTERNS:
        out = pat.sub(MASK, out)
    return out
