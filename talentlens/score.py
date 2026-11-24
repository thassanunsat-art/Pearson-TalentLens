"""Step 4 - Score each criterion, verify evidence, compute the weighted total.

The model only judges each criterion (0 / 50 / 100 with quotes). The total is
computed here in code, so it is deterministic and auditable.
"""
from __future__ import annotations

import json
import re
from datetime import date

from rapidfuzz import fuzz

from . import llm
from .schemas import CandidateProfile, CriterionScore, JobSpec, ScoringResult, VerifiedScore

EVIDENCE_MATCH_THRESHOLD = 90  # rapidfuzz partial_ratio, 0-100

SYSTEM_PROMPT = """You are a fair, careful recruiter screening a CV against job criteria.
Identity details in the CV are masked as [ปิดบัง]; never let them matter.

For every criterion return:
- score: 100 if the CV clearly meets it, 50 if partly (e.g. related but weaker skill, slightly
  fewer years), 0 if there is no evidence.
- evidence: exact quotes copied character-for-character from the CV text (short phrases).
  Do not paraphrase or translate quotes. Empty list when score is 0.
- reasoning: one short sentence, in the same language as the criterion.
Recognise synonyms and implied skills (e.g. "pandas" implies Python, "Pivot Table/VLOOKUP"
implies advanced Excel). For experience criteria count only relevant roles."""


# ------------------------------------------------------------ experience math
def _month_index(ym: str | None, today: date) -> int | None:
    if ym is None:
        return today.year * 12 + today.month - 1
    m = re.match(r"(\d{4})-(\d{1,2})", ym)
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2)) - 1


def total_years(profile: CandidateProfile, today: date | None = None) -> float:
    """Total experience in years, counting overlapping jobs once."""
    today = today or date.today()
    months: set[int] = set()
    for e in profile.experience:
        s = _month_index(e.start, today)
        t = _month_index(e.end, today)
        if s is None or t is None or t < s:
            continue
        months.update(range(s, t + 1))
    return round(len(months) / 12, 1)


# ------------------------------------------------------------------ scoring
def score_llm(redacted_text: str, years: float, job: JobSpec, call=None) -> tuple[list[CriterionScore], llm.Usage]:
    call = call or llm.structured_call
    reqs = [r.model_dump(include={"id", "criterion", "type", "min_years"}) for r in job.requirements]
    prompt = (
        f"<job title='{job.title}'>\n{json.dumps(reqs, ensure_ascii=False, indent=1)}\n</job>\n\n"
        f"<cv total_experience_years='{years}'>\n{redacted_text}\n</cv>\n\n"
        "Score every criterion by its id."
    )
    result, usage = call(SYSTEM_PROMPT, [{"type": "text", "text": prompt}], ScoringResult)
    return result.scores, usage


def score_rules(text: str, years: float, job: JobSpec) -> list[CriterionScore]:
    """Keyword + years baseline: no understanding of synonyms beyond the keyword list."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    low = text.lower()
    out = []
    for r in job.requirements:
        if r.min_years:
            s = 100 if years >= r.min_years else 50 if years >= r.min_years / 2 else 0
            ev = [l for l in lines if re.search(r"\d{4}\s*(?:-|–|ถึง|to)", l)][:2] if s else []
            out.append(CriterionScore(requirement_id=r.id, score=s, evidence=ev, reasoning=f"ประสบการณ์รวม {years} ปี"))
            continue
        hit = next((k for k in r.keywords if k.strip() and k.strip().lower() in low), None)
        ev = [next((l for l in lines if hit.strip().lower() in l.lower()), hit)] if hit else []
        out.append(CriterionScore(requirement_id=r.id, score=100 if hit else 0, evidence=ev,
                                  reasoning=f"พบคำว่า '{hit.strip()}'" if hit else "ไม่พบคำสำคัญ"))
    return out


# ------------------------------------------------------- verify + aggregate
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def evidence_found(quote: str, text: str) -> bool:
    q, t = _norm(quote), _norm(text)
    if len(q) < 3:
        return False
    return q in t or fuzz.partial_ratio(q, t) >= EVIDENCE_MATCH_THRESHOLD


def verify_and_total(raw: list[CriterionScore], job: JobSpec, cv_text: str) -> tuple[list[VerifiedScore], float, list[str]]:
    by_id = {s.requirement_id: s for s in raw}
    verified: list[VerifiedScore] = []
    for r in job.requirements:
        s = by_id.get(r.id) or CriterionScore(requirement_id=r.id, score=0, evidence=[], reasoning="ไม่ได้รับคะแนนจากโมเดล")
        good = [q for q in s.evidence if evidence_found(q, cv_text)]
        ok = s.score == 0 or bool(good)
        verified.append(VerifiedScore(
            requirement_id=r.id, criterion=r.criterion, type=r.type, weight=r.weight,
            score=s.score if ok else 0, llm_score=s.score, evidence=s.evidence,
            evidence_verified=ok, reasoning=s.reasoning,
        ))
    total = weighted_total(verified)
    failed = [v.criterion for v in verified if v.type == "must" and v.score == 0]
    return verified, total, failed


def weighted_total(scores: list[VerifiedScore]) -> float:
    wsum = sum(s.weight for s in scores)
    if not wsum:
        return 0.0
    return round(sum(s.weight * s.score for s in scores) / wsum, 1)
