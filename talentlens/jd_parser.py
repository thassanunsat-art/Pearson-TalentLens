"""JD -> JobSpec (list of weighted must/nice requirements)."""
from __future__ import annotations

import re

from . import llm
from .lexicon import SKILLS, find_skills
from .schemas import JobRequirement, JobSpec

SYSTEM_PROMPT = """You turn a job description (Thai or English) into screening criteria for HR.
Return one requirement per distinct qualification. Mark explicit requirements as "must" and
preferred/bonus items ("จะพิจารณาเป็นพิเศษ", "preferred", "nice to have") as "nice".
Weights: 4-5 for must-haves, 1-3 for nice-to-haves. Put a numeric min_years only on
experience criteria. Give Thai and English keywords/synonyms a recruiter would look for."""


def parse_jd_llm(jd_text: str, call=None) -> tuple[JobSpec, llm.Usage]:
    spec, usage = (call or llm.structured_call)(SYSTEM_PROMPT, [{"type": "text", "text": f"<jd>\n{jd_text}\n</jd>"}], JobSpec)
    for r in spec.requirements:
        r.weight = min(5, max(1, r.weight))
    return spec, usage


MUST_HEAD = re.compile(r"(คุณสมบัติ(?:ที่ต้องมี|ผู้สมัคร)?|requirements?|must[- ]have|qualifications?)", re.I)
NICE_HEAD = re.compile(r"(จะพิจารณาเป็นพิเศษ|ที่มีก็ดี|preferred|nice[- ]to[- ]have|bonus)", re.I)
STOPWORDS = re.compile(r"(วุฒิ|ปริญญาตรี|ปริญญาโท|สาขา|มีประสบการณ์|ประสบการณ์|อย่างน้อย|\d+\s*ปี|ความรู้ด้าน|มีความรู้|ใบอนุญาต|สามารถ|ใช้|ได้|มี|หรือ|ขึ้นไป|งาน)")
BULLET = re.compile(r"^\s*(?:[-•*●▪]|\d+[.)])\s*")


def parse_jd_rules(jd_text: str) -> JobSpec:
    """Headings + bullets parser for JDs written in the usual Thai/English layout."""
    lines = [l.rstrip() for l in jd_text.splitlines()]
    title = next((l.strip() for l in lines if l.strip()), "Untitled")
    title = re.sub(r"^(ตำแหน่ง|position|job title)\s*:?\s*", "", title, flags=re.I)
    section = None
    reqs: list[JobRequirement] = []
    for line in lines:
        if NICE_HEAD.search(line) and not BULLET.match(line):
            section = "nice"
            continue
        if MUST_HEAD.search(line) and not BULLET.match(line):
            section = "must"
            continue
        if section and BULLET.match(line):
            crit = BULLET.sub("", line).strip()
            years = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:ปี|years?)", crit, re.I)
            skills = find_skills(crit)
            stripped = STOPWORDS.sub(" ", crit)
            keywords = [f.strip() for s in skills for f in SKILLS[s]] or [w for w in re.split(r"[\s,/()]+", stripped) if len(w) > 2][:5]
            reqs.append(JobRequirement(
                id=f"r{len(reqs) + 1}", criterion=crit, type=section,
                weight=5 if section == "must" else 2,
                min_years=float(years.group(1)) if years else None,
                keywords=keywords,
            ))
    return JobSpec(title=title, requirements=reqs)
