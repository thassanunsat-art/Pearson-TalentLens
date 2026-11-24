"""Step 7 - Store HR decisions (audit log) and export results."""
from __future__ import annotations

import io
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .schemas import CandidateResult

DB_PATH = Path(__file__).resolve().parent.parent / "talentlens.db"

SCHEMA = """CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decided_at TEXT NOT NULL,
    job_title TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    system_score REAL NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT,
    reviewer TEXT
)"""


def _conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute(SCHEMA)
    return con


def save_decision(job_title: str, result: CandidateResult, decision: str, reason: str, reviewer: str,
                  db_path: Path = DB_PATH) -> None:
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO decisions (decided_at, job_title, candidate_id, file_name, system_score, decision, reason, reviewer)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), job_title, result.candidate_id, result.file_name,
             result.total_score, decision, reason, reviewer),
        )


def load_decisions(db_path: Path = DB_PATH) -> pd.DataFrame:
    with _conn(db_path) as con:
        return pd.read_sql_query("SELECT * FROM decisions ORDER BY id DESC", con)


def results_table(results: list[CandidateResult], blind: bool) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(results, 1):
        row = {
            "อันดับ": rank,
            "รหัส": r.candidate_id,
            "ชื่อ": "(ปิดบัง)" if blind else (r.profile.full_name or "-"),
            "ไฟล์": r.file_name,
            "คะแนนรวม": r.total_score,
            "ผ่านเกณฑ์บังคับ": "ใช่" if not r.must_have_failed else "ไม่ (" + ", ".join(r.must_have_failed) + ")",
            "ประสบการณ์ (ปี)": r.total_years_exp,
            "ข้อควรตรวจ": sum(1 for f in r.flags if f.level != "info"),
        }
        for s in r.criterion_scores:
            row[s.criterion] = s.score
        rows.append(row)
    return pd.DataFrame(rows)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="shortlist")
    return buf.getvalue()
