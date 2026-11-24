"""End-to-end screening: file -> ingest (-> OCR) -> extract -> redact -> score -> validate.

Three methods share every step except the "brain":
* ``rules`` - regex + keywords (no model)
* ``local`` - open-source model via Ollama (free, e.g. on a Colab GPU)
* ``llm``   - Claude API (paid; reads scanned images directly)
Scans are OCR'd with Tesseract for ``rules`` and ``local``.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from datetime import date
from typing import Literal

from . import extract, jd_parser, llm, local_llm, ocr, redact, score, validate
from .ingest import IngestedDocument, ingest_bytes
from .lexicon import normalize_skill_names
from .schemas import CandidateResult, Flag, JobSpec

Mode = Literal["llm", "local", "rules"]
MODE_LABELS = {"llm": "Claude API", "local": "โมเดล open-source (Ollama)", "rules": "Rule-based"}


def _caller(mode: Mode):
    return llm.structured_call if mode == "llm" else local_llm.structured_call


def parse_jd(jd_text: str, mode: Mode) -> JobSpec:
    if mode == "rules":
        return jd_parser.parse_jd_rules(jd_text)
    spec, _ = jd_parser.parse_jd_llm(jd_text, _caller(mode))
    return spec


def prepare(doc: IngestedDocument, mode: Mode, use_ocr: bool = True) -> tuple[IngestedDocument, bool]:
    """OCR scanned pages for methods that cannot read images. Returns (doc, ocr_used)."""
    if not doc.needs_vision or mode == "llm":
        return doc, False
    if not use_ocr:
        raise ValueError("ไฟล์สแกน/รูปภาพ - เปิด OCR หรือใช้ Claude API")
    return replace(doc, text=ocr.ocr_images(doc.images)), True


def screen_document(doc: IngestedDocument, job: JobSpec, mode: Mode = "llm", blind: bool = True,
                    today: date | None = None, use_ocr: bool = True) -> CandidateResult:
    t0 = time.perf_counter()
    usage = llm.Usage()
    doc, ocr_used = prepare(doc, mode, use_ocr)

    if mode == "rules":
        ext = extract.extract_rules(doc)
    else:
        ext, u = extract.extract_llm(doc, _caller(mode))
        usage.add(u)
    profile, text = ext.profile, ext.transcript

    years = score.total_years(profile, today)
    scored_text = redact.redact_text(text, profile) if blind else text

    if mode == "rules":
        raw = score.score_rules(scored_text, years, job)
    else:
        raw, u = score.score_llm(scored_text, years, job, _caller(mode))
        usage.add(u)

    verified, total, failed = score.verify_and_total(raw, job, scored_text)
    flags = validate.validate(profile, years, verified, today)
    if ocr_used:
        flags.append(Flag(level="info", field="ocr", message="อ่านจากภาพด้วย OCR - ข้อความอาจคลาดเคลื่อน ควรตรวจกับไฟล์จริง"))
    skill_names = [s.name for s in profile.skills] + profile.certifications

    return CandidateResult(
        candidate_id=hashlib.sha1(doc.file_name.encode()).hexdigest()[:8],
        file_name=doc.file_name,
        profile=profile,
        total_years_exp=years,
        skills_normalized=normalize_skill_names(skill_names),
        criterion_scores=verified,
        total_score=total,
        must_have_failed=failed,
        flags=flags,
        redacted_text=scored_text,
        seconds=round(time.perf_counter() - t0, 2),
        input_tokens=usage.input_tokens if mode == "llm" else 0,
        output_tokens=usage.output_tokens if mode == "llm" else 0,
    )


def screen_files(files: list[tuple[str, bytes]], job: JobSpec, mode: Mode = "llm", blind: bool = True,
                 on_progress=None) -> tuple[list[CandidateResult], list[tuple[str, str]]]:
    """Screen many files; returns (results ranked by score, errors)."""
    results, errors = [], []
    for i, (name, data) in enumerate(files):
        try:
            results.append(screen_document(ingest_bytes(data, name), job, mode, blind))
        except (llm.LLMError, ocr.OCRUnavailable, ValueError) as e:
            errors.append((name, str(e)))
        if on_progress:
            on_progress(i + 1, len(files), name)
    results.sort(key=lambda r: (len(r.must_have_failed) == 0, r.total_score), reverse=True)
    return results, errors
