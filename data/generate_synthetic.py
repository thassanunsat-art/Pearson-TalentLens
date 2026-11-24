"""Generate the synthetic evaluation set: 40 fictional CVs + answer keys.

Every person, phone number and email here is made up. Run once; outputs are
committed so the group (and graders) don't need the fonts:

    python data/generate_synthetic.py --font-dir C:/Windows/Fonts

Outputs
  data/jd/<job>.json                  gold JobSpec (criteria ids used by the answer key)
  data/cv/<job>/<id>.{pdf,docx,jpg}   CVs: 10 text PDF + 4 DOCX + 6 scanned JPG per job
  data/cv_counterfactual/<id>_cf.pdf  same CV, swapped name/gender (bias test)
  data/ground_truth/<job>.json        gold profile + gold score per criterion
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from talentlens.lexicon import normalize_skill_names  # noqa: E402
from talentlens.schemas import JobRequirement, JobSpec  # noqa: E402

TODAY = (2026, 11)  # reference month for "present" - eval uses the same date
SEED = 2026

# ----------------------------------------------------------------- job specs
JOBS = {
    "data_analyst": JobSpec(title="นักวิเคราะห์ข้อมูล (Data Analyst)", requirements=[
        JobRequirement(id="sql", criterion="ใช้ SQL ดึงข้อมูลจากฐานข้อมูลได้", type="must", weight=5, min_years=None, keywords=["sql", "mysql", "postgresql", "bigquery"]),
        JobRequirement(id="py_r", criterion="เขียนโปรแกรม Python หรือ R เพื่อวิเคราะห์ข้อมูลได้", type="must", weight=4, min_years=None, keywords=["python", "pandas", "r programming", "rstudio"]),
        JobRequirement(id="exp", criterion="มีประสบการณ์ด้านการวิเคราะห์ข้อมูลอย่างน้อย 2 ปี", type="must", weight=5, min_years=2, keywords=["analyst", "วิเคราะห์ข้อมูล"]),
        JobRequirement(id="bi", criterion="ใช้ Power BI หรือ Tableau ทำ dashboard ได้", type="nice", weight=3, min_years=None, keywords=["power bi", "tableau"]),
        JobRequirement(id="degree", criterion="จบการศึกษาสาขาสถิติ คณิตศาสตร์ หรือวิทยาการข้อมูล", type="nice", weight=2, min_years=None, keywords=["สถิติ", "statistics", "คณิตศาสตร์", "mathematics", "data science", "วิทยาการข้อมูล"]),
        JobRequirement(id="english", criterion="ภาษาอังกฤษดี (TOEIC 700 ขึ้นไป)", type="nice", weight=2, min_years=None, keywords=["toeic", "ielts"]),
    ]),
    "accountant": JobSpec(title="เจ้าหน้าที่บัญชี (Accountant)", requirements=[
        JobRequirement(id="degree", criterion="วุฒิปริญญาตรี สาขาการบัญชี", type="must", weight=5, min_years=None, keywords=["บัญชีบัณฑิต", "การบัญชี", "accounting", "accountancy"]),
        JobRequirement(id="exp", criterion="มีประสบการณ์งานบัญชีอย่างน้อย 1 ปี", type="must", weight=4, min_years=1, keywords=["บัญชี", "accountant"]),
        JobRequirement(id="excel", criterion="ใช้ Excel ขั้นสูงได้ (Pivot Table, VLOOKUP)", type="must", weight=4, min_years=None, keywords=["pivot", "vlookup", "xlookup", "advanced excel", "excel ขั้นสูง"]),
        JobRequirement(id="software", criterion="ใช้โปรแกรมบัญชี Express หรือ SAP ได้", type="nice", weight=3, min_years=None, keywords=["express", "sap"]),
        JobRequirement(id="license", criterion="มีใบอนุญาต CPA หรือ TA", type="nice", weight=3, min_years=None, keywords=["cpa", "tax auditor", "ผู้สอบบัญชี"]),
        JobRequirement(id="tax", criterion="มีความรู้ด้านภาษีอากร (ภ.ง.ด., ภ.พ.30)", type="nice", weight=2, min_years=None, keywords=["ภาษี", "tax", "ภ.ง.ด.", "ภ.พ.30"]),
    ]),
}

# Phrases per criterion level: {criterion: {100: [...], 50: [...]}} - level 0 adds nothing.
# Some 100-level phrases deliberately avoid the JD's words (synonyms / implied skills).
PHRASES = {
    "data_analyst": {
        "sql": {100: ["SQL (PostgreSQL, MySQL)", "เขียนคิวรี SQL ดึงข้อมูลยอดขาย", "Google BigQuery", "T-SQL บน Microsoft SQL Server"],
                50: ["Microsoft Access (สร้างตารางและรายงานพื้นฐาน)"]},
        "py_r": {100: ["Python (pandas, NumPy)", "pandas / scikit-learn", "R (tidyverse, ggplot2)", "Jupyter Notebook วิเคราะห์ข้อมูลด้วย pandas"],
                 50: ["เคยเรียนเขียนโปรแกรม Python เบื้องต้น (คอร์สออนไลน์)"]},
        "bi": {100: ["Power BI (DAX)", "Tableau", "สร้าง dashboard ด้วย Power BI"], 50: ["Looker Studio"]},
    },
    "accountant": {
        "excel": {100: ["Excel ขั้นสูง (Pivot Table, VLOOKUP)", "Advanced Excel: Pivot Table, XLOOKUP, Power Query"],
                  50: ["Microsoft Office (Word, Excel, PowerPoint)"]},
        "software": {100: ["โปรแกรมบัญชี Express", "SAP Business One", "SAP FI"], 50: ["QuickBooks"]},
        "tax": {100: ["ยื่นแบบ ภ.ง.ด.1, ภ.ง.ด.3, ภ.ง.ด.53 และ ภ.พ.30", "Thai tax filing (VAT, withholding tax)"],
                50: ["ความรู้ภาษีอากรเบื้องต้น"]},
    },
}
FILLER_SKILLS = {"data_analyst": ["Microsoft Word", "Google Sheets", "การนำเสนอ (Presentation)", "Canva"],
                 "accountant": ["Microsoft Word", "การจัดเก็บเอกสาร", "Google Sheets", "งานธุรการ"]}

DEGREES = {
    "data_analyst": {100: [("วิทยาศาสตรบัณฑิต", "สถิติ", "B.Sc.", "Statistics"), ("วิทยาศาสตรบัณฑิต", "คณิตศาสตร์", "B.Sc.", "Mathematics"),
                           ("วิทยาศาสตรบัณฑิต", "วิทยาการข้อมูล", "B.Sc.", "Data Science")],
                     50: [("เศรษฐศาสตรบัณฑิต", "เศรษฐศาสตร์", "B.Econ.", "Economics"), ("วิศวกรรมศาสตรบัณฑิต", "วิศวกรรมคอมพิวเตอร์", "B.Eng.", "Computer Engineering")],
                     0: [("บริหารธุรกิจบัณฑิต", "การตลาด", "BBA", "Marketing"), ("ศิลปศาสตรบัณฑิต", "ภาษาไทย", "B.A.", "Thai Language")]},
    "accountant": {100: [("บัญชีบัณฑิต", "การบัญชี", "B.Acc.", "Accounting")],
                   50: [("บริหารธุรกิจบัณฑิต", "การเงิน", "BBA", "Finance")],
                   0: [("ศิลปศาสตรบัณฑิต", "ภาษาอังกฤษ", "B.A.", "English"), ("บริหารธุรกิจบัณฑิต", "การตลาด", "BBA", "Marketing")]},
}
UNIS = [("มหาวิทยาลัยศรีนครินทรวิโรฒ", "Srinakharinwirot University"), ("มหาวิทยาลัยเกษตรศาสตร์", "Kasetsart University"),
        ("มหาวิทยาลัยขอนแก่น", "Khon Kaen University"), ("มหาวิทยาลัยเชียงใหม่", "Chiang Mai University"),
        ("มหาวิทยาลัยรามคำแหง", "Ramkhamhaeng University"), ("มหาวิทยาลัยธรรมศาสตร์", "Thammasat University")]

ROLES = {
    "data_analyst": {"rel": [("นักวิเคราะห์ข้อมูล", "Data Analyst"), ("นักวิเคราะห์ธุรกิจ", "Business Analyst"), ("เจ้าหน้าที่วิเคราะห์ข้อมูลการตลาด", "Marketing Data Analyst")],
                     "irr": [("พนักงานขาย", "Sales Executive"), ("เจ้าหน้าที่บริการลูกค้า", "Customer Service Officer"), ("ครูสอนพิเศษ", "Tutor")]},
    "accountant": {"rel": [("พนักงานบัญชี", "Accountant"), ("ผู้ช่วยผู้สอบบัญชี", "Audit Assistant"), ("เจ้าหน้าที่บัญชีเจ้าหนี้", "Accounts Payable Officer")],
                   "irr": [("แคชเชียร์", "Cashier"), ("เจ้าหน้าที่ธุรการ", "Admin Officer"), ("พนักงานขายหน้าร้าน", "Retail Sales Staff")]},
}
HIGHLIGHTS = {
    "data_analyst": [("จัดทำรายงานยอดขายรายสัปดาห์ให้ฝ่ายบริหาร", "Built weekly sales reports for management"),
                     ("วิเคราะห์พฤติกรรมลูกค้าเพื่อวางแผนโปรโมชัน", "Analysed customer behaviour to plan promotions")],
    "accountant": [("บันทึกบัญชีและกระทบยอดบัญชีธนาคารประจำเดือน", "Recorded journals and monthly bank reconciliations"),
                   ("จัดทำงบการเงินประจำเดือนส่งผู้จัดการ", "Prepared monthly financial statements")],
    "irr": [("ดูแลลูกค้าและประสานงานกับทีม", "Served customers and coordinated with the team")],
}
COMPANIES = [("บริษัท สยามดาต้า โซลูชั่น จำกัด", "Siam Data Solutions Co., Ltd."), ("บริษัท เจริญค้าปลีก จำกัด", "Charoen Retail Co., Ltd."),
             ("บริษัท อีสานโลจิสติกส์ จำกัด", "Isan Logistics Co., Ltd."), ("บริษัท ชลบุรีการผลิต จำกัด", "Chonburi Manufacturing Co., Ltd."),
             ("สำนักงานบัญชี ตัวอย่างการบัญชี", "Example Accounting Office"), ("บริษัท ไทยฟู้ดส์ซัพพลาย จำกัด", "Thai Foods Supply Co., Ltd.")]
NAMES = {
    "male": {"th": ["ธนพล", "กิตติพัฒน์", "ณัฐวุฒิ", "ปิยะ", "วีรภัทร", "สมชาย"], "en": ["Thanaphon", "Kittiphat", "Nattawut", "Piya", "Weeraphat", "Somchai"]},
    "female": {"th": ["สุภาพร", "ณัฐธิดา", "พิมพ์ชนก", "กมลวรรณ", "ศิริพร", "อรอุมา"], "en": ["Supaporn", "Natthida", "Pimchanok", "Kamonwan", "Siriporn", "On-uma"]},
    "last": {"th": ["ศรีสุข", "วงศ์ทอง", "แก้วมณี", "บุญมา", "รัตนพันธ์", "ทองคำ", "สุขสวัสดิ์", "ใจดี"],
             "en": ["Srisuk", "Wongthong", "Kaewmanee", "Boonma", "Rattanaphan", "Thongkham", "Suksawat", "Jaidee"]},
}
TH_MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]
EN_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ------------------------------------------------------------ person model
def months_to_ym(m: int) -> str:
    return f"{m // 12:04d}-{m % 12 + 1:02d}"


def build_person(rng: random.Random, job: str, idx: int, lang: str) -> dict:
    q = rng.random()  # latent candidate quality

    def level(bias: float = 0.0) -> int:
        x = q + bias + rng.uniform(-0.35, 0.35)
        return 100 if x > 0.6 else 50 if x > 0.35 else 0

    gender = rng.choice(["male", "female"])
    first = rng.randrange(6)
    last = rng.randrange(8)
    birth_year = rng.randint(1981, 2003)
    now = TODAY[0] * 12 + TODAY[1] - 1
    gold = {}

    # experience: relevant years by level of the 'exp' criterion
    need = JOBS[job].requirements[[r.id for r in JOBS[job].requirements].index("exp")].min_years
    lv = level(0.05)
    rel_years = {100: rng.uniform(need + 0.5, need + 5), 50: rng.uniform(need * 0.5, need * 0.95), 0: rng.uniform(0, need * 0.3)}[lv]
    gold["exp"] = lv
    jobs = []
    end = now if rng.random() < 0.7 else now - rng.randint(1, 5)
    rel_months = int(round(rel_years * 12))
    if rel_months >= 3:
        if rel_months > 30 and rng.random() < 0.5:  # split into two relevant jobs
            a = rel_months // 2
            jobs.append(("rel", end - a + 1, end))
            jobs.append(("rel", end - rel_months + 1 - 2, end - a - 2))
            end = end - rel_months - 3
        else:
            jobs.append(("rel", end - rel_months + 1, end))
            end = end - rel_months - 2
    if rng.random() < 0.6:  # an irrelevant job (traps total-years counting)
        irr = rng.randint(10, 40)
        jobs.append(("irr", end - irr + 1, end))
    earliest_ok = (birth_year + 21) * 12
    jobs = [(k, max(s, earliest_ok), e) for k, s, e in jobs if e > earliest_ok]
    # gold level from the months that actually ended up on the CV
    rel_total = sum(e - s + 1 for k, s, e in jobs if k == "rel") / 12
    gold["exp"] = 100 if rel_total >= need else 50 if rel_total >= need / 2 else 0

    experience = []
    for kind, s, e in jobs:
        ti = rng.randrange(3)
        title = ROLES[job]["rel" if kind == "rel" else "irr"][ti]
        comp = COMPANIES[rng.randrange(len(COMPANIES))]
        hl = HIGHLIGHTS[job if kind == "rel" else "irr"][rng.randrange(len(HIGHLIGHTS[job if kind == "rel" else "irr"]))]
        experience.append({"kind": kind, "title": title, "org": comp, "start": months_to_ym(s),
                           "end": None if e == now else months_to_ym(e), "highlight": hl})

    # education
    deg_lv = level(0.1) if job == "accountant" else level(-0.05)
    gold["degree"] = deg_lv
    deg = rng.choice(DEGREES[job][deg_lv])
    uni = rng.choice(UNIS)
    grad_year = birth_year + 22

    # skill phrases
    skills: list[str] = []
    for cid, levels in PHRASES[job].items():
        lv = level()
        gold[cid] = lv
        if lv:
            skills.append(rng.choice(levels[lv]))
    skills += rng.sample(FILLER_SKILLS[job], 2)
    rng.shuffle(skills)

    certs: list[str] = []
    if job == "data_analyst":
        lv = level(-0.1)
        gold["english"] = lv
        if lv == 100:
            certs.append(rng.choice([f"TOEIC {rng.randrange(705, 960, 5)}", f"IELTS {rng.choice(['6.5', '7.0', '7.5'])}"]))
        elif lv == 50:
            certs.append(f"TOEIC {rng.randrange(500, 695, 5)}")
    else:
        lv = level(-0.2)
        gold["license"] = lv
        if lv == 100:
            certs.append(rng.choice(["ผู้สอบบัญชีรับอนุญาต (CPA)", "ผู้สอบบัญชีภาษีอากร (TA)"]))
        elif lv == 50:
            certs.append("กำลังสอบ CPA (ผ่านแล้ว 3 วิชา)")

    return {
        "id": f"{'DA' if job == 'data_analyst' else 'AC'}{idx:02d}", "job": job, "lang": lang, "gender": gender,
        "first": first, "last": last, "birth_year": birth_year,
        "email": f"{NAMES[gender]['en'][first].lower().replace('-', '')}.{NAMES['last']['en'][last][:3].lower()}{idx}@example.com",
        "phone": f"08{rng.randint(1, 9)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
        "education": {"deg": deg, "uni": uni, "year": grad_year},
        "experience": experience, "skills": skills, "certs": certs, "gold": gold,
        "date_style": rng.choice(["iso", "slash_be", "thai_month", "en_month"]) if lang == "th" else rng.choice(["iso", "en_month"]),
    }


# ----------------------------------------------------------------- rendering
def fmt_date(ym: str | None, style: str, lang: str) -> str:
    if ym is None:
        return "ปัจจุบัน" if lang == "th" else "Present"
    y, m = int(ym[:4]), int(ym[5:])
    if style == "iso":
        return ym
    if style == "slash_be":
        return f"{m:02d}/{y + 543}"
    if style == "thai_month":
        return f"{TH_MONTHS[m - 1]} {y + 543}"
    return f"{EN_MONTHS[m - 1]} {y}"


def person_name(p: dict, with_title: bool = True) -> str:
    g, lang = p["gender"], p["lang"]
    first, last = NAMES[g][lang][p["first"]], NAMES["last"][lang][p["last"]]
    if not with_title:
        return f"{first} {last}"
    if lang == "th":
        return f"{'นาย' if g == 'male' else 'นางสาว'}{first} {last}"
    return f"{'Mr.' if g == 'male' else 'Ms.'} {first} {last}"


def sections(p: dict) -> dict:
    th = p["lang"] == "th"
    L = (lambda t, e: t if th else e)
    deg = p["education"]["deg"]
    uni = p["education"]["uni"]
    year = p["education"]["year"]
    year_txt = str(year + 543) if th and p["date_style"] != "iso" else str(year)
    edu = f"{deg[0] if th else deg[2]} {L('สาขา', 'in')} {deg[1] if th else deg[3]} | {uni[0] if th else uni[1]} | {year_txt}"
    exp = []
    for e in p["experience"]:
        rng_txt = f"{fmt_date(e['start'], p['date_style'], p['lang'])} – {fmt_date(e['end'], p['date_style'], p['lang'])}"
        exp.append((rng_txt, e["title"][0] if th else e["title"][1], e["org"][0] if th else e["org"][1], e["highlight"][0] if th else e["highlight"][1]))
    age_line = L(f"ปีเกิด: พ.ศ. {p['birth_year'] + 543}", f"Date of birth: {p['birth_year']}")
    return {
        "name": person_name(p),
        "contact": [f"{L('อีเมล', 'Email')}: {p['email']}", f"{L('โทร', 'Phone')}: {p['phone']}", age_line,
                    L("เพศ: " + ("ชาย" if p["gender"] == "male" else "หญิง"), "Gender: " + p["gender"].capitalize())],
        "headings": {"edu": L("การศึกษา", "Education"), "exp": L("ประสบการณ์ทำงาน", "Work Experience"),
                     "skills": L("ทักษะ", "Skills"), "certs": L("ใบรับรองและผลสอบ", "Certifications"),
                     "lang": L("ภาษา", "Languages"), "profile": L("ข้อมูลส่วนตัว", "Personal Details")},
        "edu": edu, "exp": exp, "skills": p["skills"], "certs": p["certs"],
        "langs": L("ไทย (ภาษาแม่), อังกฤษ", "Thai (native), English"),
    }


THAI_MARKS = set(chr(c) for c in list(range(0x0E31, 0x0E32)) + list(range(0x0E34, 0x0E3B)) + list(range(0x0E47, 0x0E4F)))
BLUE = (0.12, 0.31, 0.47)


class Canvas:
    """Minimal line-layout engine (PyMuPDF insert_text keeps Thai text extractable)."""

    def __init__(self, page: fitz.Page, font_dir: Path):
        self.page = page
        self.fonts = {False: ("lw", str(font_dir / "LeelawUI.ttf")), True: ("lwb", str(font_dir / "LeelaUIb.ttf"))}
        self.objs = {b: fitz.Font(fontfile=f) for b, (_, f) in self.fonts.items()}

    def wrap(self, text: str, width: float, size: float, bold: bool) -> list[str]:
        font = self.objs[bold]
        lines, cur = [], ""
        for ch in text:
            trial = cur + ch
            if font.text_length(trial, fontsize=size) <= width or not cur or ch in THAI_MARKS:
                cur = trial
                continue
            cut = cur.rfind(" ")
            if cut > 0:
                lines.append(cur[:cut])
                cur = cur[cut + 1:] + ch
            else:
                while cur and cur[-1] in THAI_MARKS | {"ไ", "เ", "แ", "โ", "ใ"}:  # don't split a syllable
                    ch, cur = cur[-1] + ch, cur[:-1]
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines

    def text(self, x: float, y: float, text: str, width: float, size: float = 9.5, bold: bool = False,
             color=(0, 0, 0), lead: float = 1.45) -> float:
        name, file = self.fonts[bold]
        for line in self.wrap(text, width, size, bold):
            self.page.insert_text((x, y), line, fontname=name, fontfile=file, fontsize=size, color=color)
            y += size * lead
        return y

    def heading(self, x: float, y: float, text: str, width: float) -> float:
        y += 6
        self.text(x, y, text, width, size=11.5, bold=True, color=BLUE)
        self.page.draw_line((x, y + 4), (x + width, y + 4), color=BLUE, width=0.6)
        return y + 18


def write_pdf(p: dict, layout: str, path: Path, font_dir: Path) -> None:
    s = sections(p)
    h = s["headings"]
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    c = Canvas(page, font_dir)
    y = c.text(40, 64, s["name"], 515, size=18, bold=True) + 4

    def block_contact(x, y, w):
        y = c.heading(x, y, h["profile"], w)
        for line in s["contact"]:
            y = c.text(x, y, line, w)
        return y

    def block_list(x, y, w, title, items):
        y = c.heading(x, y, title, w)
        for it in items or ["-"]:
            y = c.text(x + 8, y, "• " + it, w - 8)
        return y

    def block_exp(x, y, w):
        y = c.heading(x, y, h["exp"], w)
        for r, t, o, hl in s["exp"]:
            if layout == "table":
                top = y
                y1 = c.text(x, y, r, 110)
                y2 = c.text(x + 118, y, f"{t} | {o}", w * 0.42)
                y3 = c.text(x + 126 + w * 0.42, y, hl, w - 126 - w * 0.42)
                y = max(y1, y2, y3) + 2
                page.draw_line((x, y - 9), (x + w, y - 9), color=(0.7, 0.7, 0.7), width=0.4)
                y += 2
            else:
                y = c.text(x, y, f"{t} | {o}", w, bold=True)
                y = c.text(x, y, r, w, color=(0.3, 0.3, 0.3))
                y = c.text(x + 8, y, "- " + hl, w - 8) + 3
        return y

    def block_edu(x, y, w):
        y = c.heading(x, y, h["edu"], w)
        return c.text(x, y, s["edu"], w)

    if layout == "two-col":
        yl = block_contact(40, y, 165)
        yl = block_list(40, yl, 165, h["skills"], s["skills"])
        yl = block_list(40, yl, 165, h["certs"], s["certs"])
        block_list(40, yl, 165, h["lang"], [s["langs"]])
        page.draw_line((218, y), (218, 800), color=(0.8, 0.8, 0.8), width=0.5)
        yr = block_exp(232, y, 323)
        block_edu(232, yr, 323)
    else:
        y = block_contact(40, y, 515)
        y = block_exp(40, y, 515)
        y = block_edu(40, y, 515)
        y = block_list(40, y, 515, h["skills"], s["skills"])
        y = block_list(40, y, 515, h["certs"], s["certs"])
        block_list(40, y, 515, h["lang"], [s["langs"]])
    doc.subset_fonts()
    doc.save(path, garbage=4, deflate=True)


def write_scan(pdf_path: Path, out_path: Path, rng: random.Random) -> None:
    with fitz.open(pdf_path) as doc:
        png = doc[0].get_pixmap(dpi=110).tobytes("png")
    img = Image.open(io.BytesIO(png)).convert("L")
    img = img.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=255).filter(ImageFilter.GaussianBlur(0.6))
    img.save(out_path, "JPEG", quality=60)


def write_docx(p: dict, layout: str, path: Path) -> None:
    import docx
    from docx.oxml.ns import qn

    s = sections(p)
    h = s["headings"]
    d = docx.Document()
    style = d.styles["Normal"]
    style.font.name = "Tahoma"
    style.element.rPr.rFonts.set(qn("w:cs"), "Tahoma")
    d.add_heading(s["name"], level=1)
    d.add_heading(h["profile"], level=2)
    for c in s["contact"]:
        d.add_paragraph(c)
    d.add_heading(h["exp"], level=2)
    if layout == "table":
        t = d.add_table(rows=0, cols=3)
        for r, ti, o, hl in s["exp"]:
            cells = t.add_row().cells
            cells[0].text, cells[1].text, cells[2].text = r, f"{ti} | {o}", hl
    else:
        for r, ti, o, hl in s["exp"]:
            d.add_paragraph(f"{ti} | {o}")
            d.add_paragraph(r)
            d.add_paragraph(f"- {hl}")
    d.add_heading(h["edu"], level=2)
    d.add_paragraph(s["edu"])
    d.add_heading(h["skills"], level=2)
    for x in s["skills"]:
        d.add_paragraph(x, style="List Bullet")
    d.add_heading(h["certs"], level=2)
    for x in s["certs"] or ["-"]:
        d.add_paragraph(x, style="List Bullet")
    d.add_heading(h["lang"], level=2)
    d.add_paragraph(s["langs"])
    d.save(path)


def gold_record(p: dict, file_name: str, fmt: str, layout: str) -> dict:
    job = JOBS[p["job"]]
    weights = {r.id: r.weight for r in job.requirements}
    total = round(sum(weights[k] * v for k, v in p["gold"].items()) / sum(weights.values()), 1)
    return {
        "id": p["id"], "file": file_name, "format": fmt, "layout": layout, "lang": p["lang"],
        "profile": {
            "full_name": person_name(p, with_title=False), "email": p["email"], "phone": p["phone"],
            "gender": p["gender"], "birth_year": p["birth_year"], "grad_year": p["education"]["year"],
            "experience": [{"start": e["start"], "end": e["end"], "relevant": e["kind"] == "rel"} for e in p["experience"]],
            "skills": normalize_skill_names(p["skills"] + p["certs"]),
        },
        "gold_scores": p["gold"], "gold_total": total,
        "must_have_failed": [r.id for r in job.requirements if r.type == "must" and p["gold"][r.id] == 0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font-dir", default="C:/Windows/Fonts", help="folder containing tahoma.ttf and tahomabd.ttf")
    args = ap.parse_args()
    font_dir = Path(args.font_dir)
    rng = random.Random(SEED)

    (ROOT / "ground_truth").mkdir(exist_ok=True)
    cf_dir = ROOT / "cv_counterfactual"
    cf_dir.mkdir(exist_ok=True)
    cf_pairs = []
    for job, spec in JOBS.items():
        (ROOT / "jd" / f"{job}.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        out = ROOT / "cv" / job
        out.mkdir(parents=True, exist_ok=True)
        records = []
        plan = ["pdf"] * 10 + ["docx"] * 4 + ["scan"] * 6
        for i, fmt in enumerate(plan, 1):
            lang = "th" if i % 2 else "en"
            p = build_person(rng, job, i, lang)
            layout = ["classic", "two-col", "table"][i % 3] if fmt != "docx" else ["classic", "table"][i % 2]
            if fmt == "pdf":
                fname = f"{p['id']}.pdf"
                write_pdf(p, layout, out / fname, font_dir)
            elif fmt == "docx":
                fname = f"{p['id']}.docx"
                write_docx(p, layout, out / fname)
            else:
                fname = f"{p['id']}.jpg"
                tmp = out / f"_{p['id']}.pdf"
                write_pdf(p, layout, tmp, font_dir)
                write_scan(tmp, out / fname, rng)
                tmp.unlink()
            records.append(gold_record(p, f"{job}/{fname}", fmt, layout))
            if fmt == "pdf" and i <= 5:  # counterfactual twin: swap gender + name, keep everything else
                twin = dict(p, gender="female" if p["gender"] == "male" else "male", first=(p["first"] + 3) % 6, last=(p["last"] + 4) % 8)
                write_pdf(twin, layout, cf_dir / f"{p['id']}_cf.pdf", font_dir)
                cf_pairs.append({"job": job, "original": f"{job}/{fname}", "counterfactual": f"{p['id']}_cf.pdf"})
        (ROOT / "ground_truth" / f"{job}.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{job}: {len(records)} CVs")
    (ROOT / "ground_truth" / "counterfactual_pairs.json").write_text(json.dumps(cf_pairs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"counterfactual pairs: {len(cf_pairs)}")


if __name__ == "__main__":
    main()
