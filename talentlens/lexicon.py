"""Skill lexicon: canonical skill id -> surface forms seen in CVs (Thai + English).

Used to normalise skills for evaluation (skill F1) and by the rule-based
(offline) extractor. The LLM path does not depend on it for scoring.
"""
from __future__ import annotations

import re

SKILLS: dict[str, list[str]] = {
    "sql": ["sql", "mysql", "postgresql", "postgres", "t-sql", "bigquery", "เขียนคิวรี", "query ฐานข้อมูล"],
    "python": ["python", "pandas", "numpy", "scikit-learn", "jupyter", "ไพธอน"],
    "r": ["r programming", "rstudio", "tidyverse", "ggplot2", "ภาษา r"],
    "power_bi": ["power bi", "powerbi", "dax"],
    "tableau": ["tableau"],
    "excel": ["excel", "pivot table", "vlookup", "xlookup", "power query", "เอ็กเซล"],
    "statistics": ["statistics", "statistical", "สถิติ", "regression", "hypothesis testing", "การถดถอย"],
    "machine_learning": ["machine learning", "แมชชีนเลิร์นนิง"],
    "etl": ["etl", "data pipeline", "airflow"],
    "express": ["express accounting", "โปรแกรม express", "express"],
    "sap": ["sap", "sap fi", "sap b1"],
    "formula": ["formula accounting", "โปรแกรม formula"],
    "tax": ["ภาษี", "tax", "ภ.ง.ด.", "ภ.พ.30", "vat", "ภาษีมูลค่าเพิ่ม", "ภาษีหัก ณ ที่จ่าย", "withholding tax"],
    "financial_reporting": ["งบการเงิน", "financial statement", "financial reporting", "ปิดงบ", "month-end closing"],
    "audit": ["audit", "ตรวจสอบบัญชี", "internal control", "ควบคุมภายใน"],
    "cpa": ["cpa", "ผู้สอบบัญชีรับอนุญาต"],
    "ta": ["tax auditor", "ผู้สอบบัญชีภาษีอากร", " ta "],
    "english": ["toeic", "ielts", "toefl", "english", "ภาษาอังกฤษ"],
    "presentation": ["presentation", "นำเสนอ", "storytelling"],
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", f" {text.lower()} ")


def find_skills(text: str) -> list[str]:
    """Return canonical skill ids whose surface forms appear in text."""
    t = _norm(text)
    found = []
    for skill, forms in SKILLS.items():
        for form in forms:
            f = form.lower()
            if f.strip() in ("r", "ta", "express", "sap"):
                # short tokens: require word boundaries
                if re.search(rf"(?<![a-z]){re.escape(f.strip())}(?![a-z])", t):
                    found.append(skill)
                    break
            elif f in t:
                found.append(skill)
                break
    return found


def normalize_skill_names(names: list[str]) -> list[str]:
    """Map free-text skill names (e.g. from the LLM) to canonical ids."""
    out: set[str] = set()
    for n in names:
        out.update(find_skills(n))
    return sorted(out)
