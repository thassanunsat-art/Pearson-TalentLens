"""Step 5 - Validate: sanity rules that flag records for HR to double-check."""
from __future__ import annotations

import re
from datetime import date

from .schemas import CandidateProfile, Flag, VerifiedScore

EMAIL_OK = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


def validate(profile: CandidateProfile, years: float, scores: list[VerifiedScore], today: date | None = None) -> list[Flag]:
    today = today or date.today()
    flags: list[Flag] = []
    now = f"{today.year:04d}-{today.month:02d}"

    if not profile.full_name:
        flags.append(Flag(level="warning", field="full_name", message="ไม่พบชื่อผู้สมัคร"))
    if profile.email and not EMAIL_OK.match(profile.email):
        flags.append(Flag(level="warning", field="email", message=f"รูปแบบอีเมลไม่ถูกต้อง: {profile.email}"))
    if not profile.email and not profile.phone:
        flags.append(Flag(level="warning", field="contact", message="ไม่พบช่องทางติดต่อ"))

    for e in profile.experience:
        label = f"{e.title} @ {e.organization}".strip(" @")
        if e.start and e.end and e.end < e.start:
            flags.append(Flag(level="error", field="experience", message=f"วันสิ้นสุดก่อนวันเริ่มงาน: {label}"))
        if e.start and e.start > now:
            flags.append(Flag(level="error", field="experience", message=f"วันเริ่มงานอยู่ในอนาคต: {label}"))
        if not e.start:
            flags.append(Flag(level="info", field="experience", message=f"ไม่ระบุวันเริ่มงาน: {label}"))

    if profile.birth_year:
        working_age_years = today.year - profile.birth_year - 15
        if years > working_age_years:
            flags.append(Flag(level="error", field="experience", message=f"ประสบการณ์ {years} ปี มากกว่าที่เป็นไปได้ตามอายุ"))
        starts = [int(e.start[:4]) for e in profile.experience if e.start and re.match(r"\d{4}", e.start)]
        if starts and min(starts) < profile.birth_year + 15:
            flags.append(Flag(level="warning", field="experience", message="เริ่มทำงานก่อนอายุ 15 ปี - ตรวจปีเกิดหรือปี พ.ศ./ค.ศ."))
    if years > 45:
        flags.append(Flag(level="error", field="experience", message=f"ประสบการณ์รวม {years} ปี ผิดปกติ"))

    for s in scores:
        if not s.evidence_verified:
            flags.append(Flag(level="warning", field=s.requirement_id,
                              message=f"หลักฐานไม่ตรงกับ CV - ตัดคะแนน '{s.criterion}' จาก {s.llm_score} เป็น 0"))
    for f in profile.low_confidence_fields:
        flags.append(Flag(level="info", field=f, message="โมเดลอ่านส่วนนี้ได้ไม่ชัด - กรุณาตรวจ"))
    return flags
