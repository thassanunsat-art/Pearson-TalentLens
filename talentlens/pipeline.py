"""End-to-end screening: file -> ingest -> extract -> redact -> score -> validate."""
from __future__ import annotations

import hashlib
import time
from datetime import date
from typing import Literal

from . import extract, jd_parser, llm, redact, score, validate
from .ingest import IngestedDocument, ingest_bytes
from .lexicon import normalize_skill_names
from .schemas import CandidateResult, JobSpec

Mode = Literal["llm", "rules"]


def parse_jd(jd_text: str, mode: Mode) -> JobSpec:
    if mode == "llm":
        spec, _ = jd_parser.parse_jd_llm(jd_text)
        return spec
    return jd_parser.parse_jd_rules(jd_text)


def screen_document(doc: IngestedDocument, job: JobSpec, mode: Mode = "llm", blind: bool = True,
                    today: date | None = None) -> CandidateResult:
    t0 = time.perf_counter()
    usage = llm.Usage()

    if mode == "llm":
        ext, u = extract.extract_llm(doc)
        usage.add(u)
    else:
        ext = extract.extract_rules(doc)
    profile, text = ext.profile, ext.transcript

    years = score.total_years(profile, today)
    scored_text = redact.redact_text(text, profile) if blind else text

    if mode == "llm":
        raw, u = score.score_llm(scored_text, years, job)
        usage.add(u)
    else:
        raw = score.score_rules(scored_text, years, job)

    verified, total, failed = score.verify_and_total(raw, job, scored_text)
    flags = validate.validate(profile, years, verified, today)
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
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


def screen_files(files: list[tuple[str, bytes]], job: JobSpec, mode: Mode = "llm", blind: bool = True,
                 on_progress=None) -> tuple[list[CandidateResult], list[tuple[str, str]]]:
    """Screen many files; returns (results ranked by score, errors)."""
    results, errors = [], []
    for i, (name, data) in enumerate(files):
        try:
            results.append(screen_document(ingest_bytes(data, name), job, mode, blind))
        except (llm.LLMError, ValueError) as e:
            errors.append((name, str(e)))
        if on_progress:
            on_progress(i + 1, len(files), name)
    results.sort(key=lambda r: (len(r.must_have_failed) == 0, r.total_score), reverse=True)
    return results, errors
