"""Data models shared by every stage of the TalentLens pipeline.

LLM-facing models keep every field required (nullable where a value may be
missing) so they work with Claude structured outputs.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- CV profile
class Education(BaseModel):
    degree: str = Field(description="Degree level and name, e.g. 'ปริญญาตรี บัญชีบัณฑิต' or 'B.Sc. Statistics'")
    field: str = Field(description="Field of study, e.g. 'Accounting', 'Statistics'")
    institution: str
    year: int | None = Field(description="Graduation year in Gregorian calendar (convert พ.ศ. by subtracting 543)")


class Experience(BaseModel):
    title: str
    organization: str
    start: str | None = Field(description="Start month as YYYY-MM (Gregorian)")
    end: str | None = Field(description="End month as YYYY-MM, or null if this is the current job")
    highlights: list[str] = Field(description="Key responsibilities or achievements, copied or lightly condensed from the CV")


class Skill(BaseModel):
    name: str = Field(description="Skill as written in the CV")
    evidence: str = Field(description="Exact short quote from the CV that shows this skill")


class CandidateProfile(BaseModel):
    full_name: str | None
    email: str | None
    phone: str | None
    gender: str | None = Field(description="Only if explicitly stated or implied by a title such as นาย/นางสาว/Mr./Ms.")
    birth_year: int | None = Field(description="Gregorian birth year if stated or derivable from age")
    has_photo: bool
    summary: str | None
    education: list[Education]
    experience: list[Experience]
    skills: list[Skill]
    certifications: list[str] = Field(description="Certificates and test scores, e.g. 'TOEIC 785', 'CPA'")
    languages: list[str]
    low_confidence_fields: list[str] = Field(
        description="Names of fields above that were hard to read or ambiguous; empty if none"
    )


class ExtractionResult(BaseModel):
    """What the extractor returns: a profile plus a transcription of the CV."""

    profile: CandidateProfile
    transcript: str = Field(
        description="Full plain-text transcription of the CV in reading order, original language, no commentary"
    )


# --------------------------------------------------------- job requirements
class JobRequirement(BaseModel):
    id: str = Field(description="Short slug, e.g. 'sql', 'exp-2y'")
    criterion: str = Field(description="One requirement in plain words, in the JD's language")
    type: Literal["must", "nice"]
    weight: int = Field(description="Importance 1-5; must-haves usually 4-5")
    min_years: float | None = Field(description="Minimum years of relevant experience if this criterion is about experience")
    keywords: list[str] = Field(description="Words or synonyms that signal this criterion in a CV (Thai and English)")


class JobSpec(BaseModel):
    title: str
    requirements: list[JobRequirement]


# ------------------------------------------------------------------ scoring
class CriterionScore(BaseModel):
    requirement_id: str
    score: Literal[0, 50, 100] = Field(description="0 = no evidence, 50 = partial, 100 = clearly met")
    evidence: list[str] = Field(description="Exact quotes copied from the CV text that support the score; empty when score is 0")
    reasoning: str = Field(description="One short sentence explaining the score")


class ScoringResult(BaseModel):
    scores: list[CriterionScore]


class Flag(BaseModel):
    level: Literal["info", "warning", "error"]
    field: str
    message: str


class VerifiedScore(BaseModel):
    requirement_id: str
    criterion: str
    type: Literal["must", "nice"]
    weight: int
    score: int
    llm_score: int
    evidence: list[str]
    evidence_verified: bool
    reasoning: str


class CandidateResult(BaseModel):
    candidate_id: str
    file_name: str
    profile: CandidateProfile
    total_years_exp: float
    skills_normalized: list[str]
    criterion_scores: list[VerifiedScore]
    total_score: float
    must_have_failed: list[str]
    flags: list[Flag]
    redacted_text: str
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
