"""Offline tests - no API calls (the LLM path is exercised with a fake)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from talentlens import extract, llm, score
from talentlens.ingest import clean_text, ingest_path, looks_garbled
from talentlens.jd_parser import parse_jd_rules
from talentlens.lexicon import find_skills
from talentlens.pipeline import screen_document
from talentlens.redact import MASK, redact_text
from talentlens.schemas import (CandidateProfile, CriterionScore, Education, Experience, ExtractionResult, JobSpec,
                                ScoringResult, Skill)
from talentlens.validate import validate

DATA = Path(__file__).resolve().parent.parent / "data"
TODAY = date(2026, 11, 30)


def profile(**kw) -> CandidateProfile:
    base = dict(full_name="นางสาวสุภาพร ศรีสุข", email="a@example.com", phone="081-234-5678", gender="female",
                birth_year=1995, has_photo=False, summary=None, education=[], experience=[], skills=[],
                certifications=[], languages=[], low_confidence_fields=[])
    base.update(kw)
    return CandidateProfile(**base)


def job(name="data_analyst") -> JobSpec:
    return JobSpec.model_validate_json((DATA / "jd" / f"{name}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ units
def test_total_years_counts_overlap_once():
    p = profile(experience=[
        Experience(title="A", organization="X", start="2020-01", end="2021-12", highlights=[]),
        Experience(title="B", organization="Y", start="2021-01", end="2022-12", highlights=[]),
    ])
    assert score.total_years(p, TODAY) == 3.0


def test_current_job_runs_to_today():
    p = profile(experience=[Experience(title="A", organization="X", start="2026-01", end=None, highlights=[])])
    assert score.total_years(p, TODAY) == pytest.approx(11 / 12, abs=0.05)


def test_weighted_total_and_must_fail():
    spec = job()
    raw = [CriterionScore(requirement_id=r.id, score=100, evidence=["SQL"], reasoning="") for r in spec.requirements]
    raw[0] = CriterionScore(requirement_id="sql", score=0, evidence=[], reasoning="")
    verified, total, failed = score.verify_and_total(raw, spec, "Python, SQL, Power BI")
    w = {r.id: r.weight for r in spec.requirements}
    assert total == pytest.approx(100 * (sum(w.values()) - w["sql"]) / sum(w.values()), abs=0.1)
    assert failed == [spec.requirements[0].criterion]


def test_unverified_evidence_is_zeroed():
    spec = job()
    raw = [CriterionScore(requirement_id="sql", score=100, evidence=["10 years of Snowflake SQL"], reasoning="")]
    verified, _, _ = score.verify_and_total(raw, spec, "I like Excel and Word.")
    sql = verified[0]
    assert sql.llm_score == 100 and sql.score == 0 and not sql.evidence_verified


def test_evidence_fuzzy_match_tolerates_spacing():
    assert score.evidence_found("Python  (pandas,NumPy)", "Skills: Python (pandas, NumPy), SQL")


def test_redaction_hides_identity_keeps_skills():
    text = "นางสาวสุภาพร ศรีสุข\nอีเมล: a@example.com\nโทร: 081-234-5678\nปีเกิด: พ.ศ. 2538\nเพศ: หญิง\nทักษะ: SQL, Python"
    out = redact_text(text, profile())
    for secret in ("สุภาพร", "ศรีสุข", "a@example.com", "081-234-5678", "2538", "หญิง"):
        assert secret not in out
    assert "SQL, Python" in out and MASK in out


def test_lexicon_synonyms():
    assert set(find_skills("pandas, BigQuery, Pivot Table")) >= {"python", "sql", "excel"}
    assert "r" not in find_skills("Recorded journals")


def test_parse_dates_thai_and_english():
    assert extract.parse_date("ก.ค. 2569") == "2026-07"
    assert extract.parse_date("06/2564") == "2021-06"
    assert extract.parse_date("Jun 2021") == "2021-06"
    assert extract.parse_date("2020", end=True) == "2020-12"
    assert extract.parse_date("ปัจจุบัน") is None


def test_clean_and_garbled_detection():
    assert clean_text("จํากัด") == "จำกัด"
    assert looks_garbled("สุขสวัสดิѻ โซลูชัѷน" * 5)
    assert not looks_garbled("บริษัท สยามดาต้า จำกัด")


def test_validate_flags_impossible_dates():
    p = profile(birth_year=2000, experience=[Experience(title="A", organization="X", start="2012-01", end="2011-01", highlights=[])])
    msgs = [f.message for f in validate(p, 0.0, [], TODAY)]
    assert any("วันสิ้นสุดก่อน" in m for m in msgs)
    assert any("ก่อนอายุ 15" in m for m in msgs)


def test_jd_rules_parser_reads_sections():
    spec = parse_jd_rules((DATA / "jd" / "accountant.txt").read_text(encoding="utf-8"))
    assert [r.type for r in spec.requirements] == ["must"] * 3 + ["nice"] * 3
    assert spec.requirements[1].min_years == 1


# ------------------------------------------------------------ integration
def test_ingest_all_sample_formats():
    kinds = {ingest_path(p).kind for p in (DATA / "cv" / "accountant").iterdir()}
    assert kinds == {"pdf-text", "docx", "image"}


def test_rules_pipeline_on_sample_cv():
    gold = json.loads((DATA / "ground_truth" / "accountant.json").read_text(encoding="utf-8"))[2]
    res = screen_document(ingest_path(DATA / "cv" / gold["file"]), job("accountant"), mode="rules", today=TODAY)
    assert res.profile.email == gold["profile"]["email"]
    assert 0 <= res.total_score <= 100
    assert MASK in res.redacted_text


def test_llm_pipeline_with_fake_model(monkeypatch):
    """The LLM branch end-to-end, with structured_call replaced by canned answers."""
    path = DATA / "cv" / "data_analyst" / "DA03.pdf"
    doc = ingest_path(path)
    spec = job()

    def fake_call(system, content, output_model, max_tokens=16000):
        if output_model is CandidateProfile:  # text CVs: profile only, transcript = text layer
            prof = profile(full_name=doc.text.splitlines()[0], skills=[Skill(name="SQL", evidence="SQL")],
                           education=[Education(degree="B.Sc.", field="Statistics", institution="X", year=2018)],
                           experience=[Experience(title="Data Analyst", organization="Y", start="2022-01", end=None, highlights=[])])
            return prof, llm.Usage(1000, 200)
        assert output_model is ScoringResult
        assert MASK in content[0]["text"]  # the scorer only sees redacted text
        scores = [CriterionScore(requirement_id=r.id, score=100, evidence=["made-up quote not in the CV"], reasoning="")
                  for r in spec.requirements]
        return ScoringResult(scores=scores), llm.Usage(2000, 300)

    monkeypatch.setattr(llm, "structured_call", fake_call)
    res = screen_document(doc, spec, mode="llm", today=TODAY)
    assert res.input_tokens == 3000 and res.output_tokens == 500
    assert res.total_score == 0  # every fabricated quote is rejected
    assert all(not s.evidence_verified for s in res.criterion_scores)


# ------------------------------------------------------- free backends
class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, json.dumps(payload)

    def json(self):
        return self.payload


def test_local_llm_retries_invalid_json(monkeypatch):
    from talentlens import local_llm

    answers = iter(['{"scores": "not a list"}', '{"scores": []}'])
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json)
        return FakeResponse({"message": {"content": next(answers)}, "prompt_eval_count": 50, "eval_count": 10})

    monkeypatch.setattr(local_llm.httpx, "post", fake_post)
    result, usage = local_llm.structured_call("sys", [{"type": "text", "text": "cv"}], ScoringResult)
    assert result.scores == [] and usage.input_tokens == 100
    assert sent[0]["options"]["num_ctx"] >= 8192 and sent[0]["options"]["temperature"] == 0
    assert isinstance(sent[0]["format"], dict)  # JSON schema constrained decoding
    assert len(sent[1]["messages"]) == 4  # second try includes the validation error


def test_local_llm_rejects_images():
    from talentlens import local_llm

    with pytest.raises(llm.LLMError):
        local_llm.structured_call("sys", [{"type": "image", "source": {}}], ScoringResult)


def test_local_mode_uses_ollama_backend(monkeypatch):
    from talentlens import local_llm

    calls = []

    def fake_local(system, content, output_model, max_tokens=16000):
        calls.append(output_model.__name__)
        if output_model is CandidateProfile:
            return profile(), llm.Usage(10, 10)
        return ScoringResult(scores=[]), llm.Usage(10, 10)

    def fail(*a, **k):
        raise AssertionError("Claude must not be called in local mode")

    monkeypatch.setattr(local_llm, "structured_call", fake_local)
    monkeypatch.setattr(llm, "structured_call", fail)
    res = screen_document(ingest_path(DATA / "cv" / "accountant" / "AC03.pdf"), job("accountant"), mode="local", today=TODAY)
    assert calls == ["CandidateProfile", "ScoringResult"]
    assert res.input_tokens == 0  # free backend: no API cost reported


def test_scan_without_ocr_is_reported():
    from talentlens import ocr
    from talentlens.pipeline import prepare

    doc = ingest_path(DATA / "cv" / "accountant" / "AC15.jpg")
    with pytest.raises(ValueError):
        prepare(doc, "rules", use_ocr=False)
    if not ocr.available():
        with pytest.raises(ocr.OCRUnavailable):
            prepare(doc, "rules", use_ocr=True)


@pytest.mark.skipif("not __import__('talentlens.ocr', fromlist=['x']).available()", reason="Tesseract with Thai not installed")
def test_ocr_reads_sample_scan():
    from talentlens.pipeline import prepare

    gold = json.loads((DATA / "ground_truth" / "accountant.json").read_text(encoding="utf-8"))
    g = next(x for x in gold if x["file"].endswith("AC15.jpg"))
    doc, used = prepare(ingest_path(DATA / "cv" / g["file"]), "rules")
    assert used and g["profile"]["email"] in doc.text
