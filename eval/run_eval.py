"""Evaluate TalentLens against the synthetic answer key.

    python eval/run_eval.py                 # rule-based + TF-IDF (free, offline)
    python eval/run_eval.py --mode llm      # Claude API (costs money; results are cached)
    python eval/run_eval.py --mode all

Writes eval/results/REPORT.md, summary.json and per_cv_<method>.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from talentlens import llm  # noqa: E402
from talentlens.ingest import ingest_path  # noqa: E402
from talentlens.pipeline import screen_document  # noqa: E402
from talentlens.schemas import CandidateProfile, CandidateResult, Experience, JobSpec  # noqa: E402
from talentlens.score import total_years  # noqa: E402

from baseline import tfidf_scores  # noqa: E402

TODAY = date(2026, 11, 30)  # must match data/generate_synthetic.py TODAY
DATA = ROOT / "data"
OUT = ROOT / "eval" / "results"
CACHE = ROOT / "eval" / "cache"
JOBS = ["data_analyst", "accountant"]
TITLES = re.compile(r"^(นางสาว|นาง|นาย|mr\.?|mrs\.?|ms\.?|miss)\s*", re.I)


# ---------------------------------------------------------------- helpers
def norm_name(s: str | None) -> str:
    return re.sub(r"\s+", " ", TITLES.sub("", (s or "").strip())).lower()


def digits(s: str | None) -> str:
    return re.sub(r"\D", "", s or "")


def gold_years(g: dict) -> float:
    prof = CandidateProfile(full_name=None, email=None, phone=None, gender=None, birth_year=None, has_photo=False,
                            summary=None, education=[], skills=[], certifications=[], languages=[], low_confidence_fields=[],
                            experience=[Experience(title="", organization="", start=e["start"], end=e["end"], highlights=[]) for e in g["profile"]["experience"]])
    return total_years(prof, TODAY)


def ndcg_at_k(system: list[float], gold: list[float], k: int = 5) -> float:
    order = sorted(range(len(system)), key=lambda i: system[i], reverse=True)[:k]
    ideal = sorted(gold, reverse=True)[:k]
    dcg = sum(gold[i] / math.log2(r + 2) for r, i in enumerate(order))
    idcg = sum(g / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def f1(pred: set, gold: set) -> tuple[int, int, int]:
    return len(pred & gold), len(pred - gold), len(gold - pred)


# ------------------------------------------------------------- run system
def run_one(path: Path, job: JobSpec, mode: str, blind: bool = True) -> CandidateResult | None:
    key = CACHE / mode / ("blind" if blind else "open") / (path.stem + ".json")
    if mode == "llm" and key.exists():
        return CandidateResult.model_validate_json(key.read_text(encoding="utf-8"))
    doc = ingest_path(path)
    if mode == "rules" and doc.needs_vision:
        return None
    res = screen_document(doc, job, mode=mode, blind=blind, today=TODAY)
    if mode == "llm":
        key.parent.mkdir(parents=True, exist_ok=True)
        key.write_text(res.model_dump_json(), encoding="utf-8")
    return res


def evaluate_method(mode: str, limit: int | None) -> tuple[dict, pd.DataFrame]:
    rows, rank_stats, crit_rows = [], [], []
    tp = fp = fn = 0
    for job_name in JOBS:
        job = JobSpec.model_validate_json((DATA / "jd" / f"{job_name}.json").read_text(encoding="utf-8"))
        gold = json.loads((DATA / "ground_truth" / f"{job_name}.json").read_text(encoding="utf-8"))[:limit]
        sys_totals, gold_totals = [], []
        for g in gold:
            try:
                r = run_one(DATA / "cv" / g["file"], job, mode)
            except llm.LLMError as e:
                print(f"  ! {g['id']}: {e}")
                r = None
            row = {"job": job_name, "id": g["id"], "format": g["format"], "lang": g["lang"], "layout": g["layout"],
                   "gold_total": g["gold_total"], "processed": r is not None}
            if r is None:  # unreadable for this method -> treated as score 0 in the ranking
                sys_totals.append(0.0)
                gold_totals.append(g["gold_total"])
                rows.append(row)
                continue
            p, gp = r.profile, g["profile"]
            gy = gold_years(g)
            row.update({
                "sys_total": r.total_score,
                "name_ok": norm_name(p.full_name) == norm_name(gp["full_name"]),
                "email_ok": (p.email or "").lower() == gp["email"],
                "phone_ok": digits(p.phone) == digits(gp["phone"]),
                "birth_year_ok": p.birth_year == gp["birth_year"],
                "grad_year_ok": max((e.year or 0 for e in p.education), default=0) == gp["grad_year"],
                "years_sys": r.total_years_exp, "years_gold": gy, "years_ok": abs(r.total_years_exp - gy) <= 0.5,
                "must_pass_ok": (not r.must_have_failed) == (not g["must_have_failed"]),
                "evidence_verified": all(s.evidence_verified for s in r.criterion_scores),
                "seconds": r.seconds, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            })
            a, b, c = f1(set(r.skills_normalized), set(gp["skills"]))
            tp, fp, fn = tp + a, fp + b, fn + c
            for s in r.criterion_scores:
                gs = g["gold_scores"].get(s.requirement_id)
                crit_rows.append({"job": job_name, "criterion": s.requirement_id, "sys": s.score, "gold": gs})
            rows.append(row)
            sys_totals.append(r.total_score)
            gold_totals.append(g["gold_total"])
        rho = spearmanr(sys_totals, gold_totals).statistic
        rank_stats.append({"job": job_name, "spearman": rho, "ndcg@5": ndcg_at_k(sys_totals, gold_totals)})

    df = pd.DataFrame(rows)
    done = df[df.processed]
    crit = pd.DataFrame(crit_rows)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    summary = {
        "method": mode,
        "cvs": len(df), "processed": int(df.processed.sum()),
        **{k: float(done[k].mean()) for k in ["name_ok", "email_ok", "phone_ok", "birth_year_ok", "grad_year_ok", "years_ok", "must_pass_ok", "evidence_verified"] if k in done},
        "years_mae": float((done.years_sys - done.years_gold).abs().mean()) if len(done) else None,
        "skills_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "criterion_exact": float((crit.sys == crit.gold).mean()) if len(crit) else None,
        "criterion_within_50": float(((crit.sys - crit.gold).abs() <= 50).mean()) if len(crit) else None,
        "spearman_mean": float(pd.DataFrame(rank_stats).spearman.mean()),
        "ndcg5_mean": float(pd.DataFrame(rank_stats)["ndcg@5"].mean()),
        "per_job": rank_stats,
        "sec_per_cv": float(done.seconds.mean()) if len(done) else None,
        "usd_per_cv": llm.Usage(int(done.input_tokens.sum()), int(done.output_tokens.sum())).cost_usd / max(1, len(done)),
    }
    return summary, df


def evaluate_tfidf(limit: int | None) -> dict:
    stats = []
    for job_name in JOBS:
        jd = (DATA / "jd" / f"{job_name}.txt").read_text(encoding="utf-8")
        gold = json.loads((DATA / "ground_truth" / f"{job_name}.json").read_text(encoding="utf-8"))[:limit]
        texts = []
        for g in gold:
            d = ingest_path(DATA / "cv" / g["file"])
            texts.append(d.text)  # scans have no text -> similarity 0
        sims = tfidf_scores(jd, texts)
        gt = [g["gold_total"] for g in gold]
        stats.append({"job": job_name, "spearman": spearmanr(sims, gt).statistic, "ndcg@5": ndcg_at_k(list(sims), gt)})
    return {"method": "tfidf", "per_job": stats,
            "spearman_mean": float(pd.DataFrame(stats).spearman.mean()), "ndcg5_mean": float(pd.DataFrame(stats)["ndcg@5"].mean())}


def counterfactual(mode: str) -> dict | None:
    pairs = json.loads((DATA / "ground_truth" / "counterfactual_pairs.json").read_text(encoding="utf-8"))
    out = {}
    for blind in (True, False):
        diffs = []
        for pr in pairs:
            job = JobSpec.model_validate_json((DATA / "jd" / f"{pr['job']}.json").read_text(encoding="utf-8"))
            a = run_one(DATA / "cv" / pr["original"], job, mode, blind)
            b = run_one(DATA / "cv_counterfactual" / pr["counterfactual"], job, mode, blind)
            if a and b:
                diffs.append(abs(a.total_score - b.total_score))
        out["blind" if blind else "open"] = {"pairs": len(diffs), "mean_abs_diff": sum(diffs) / len(diffs) if diffs else None,
                                             "max_abs_diff": max(diffs) if diffs else None}
    return out


# ----------------------------------------------------------------- report
def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v * 100:.0f}%" if pct else f"{v:.2f}"


def report(summaries: list[dict], cf: dict) -> str:
    lines = ["# ผลการประเมิน TalentLens", "",
             f"ชุดข้อมูล: CV สมมติ 40 ฉบับ (2 ตำแหน่ง) · วันอ้างอิง {TODAY.isoformat()}", "",
             "| ตัวชี้วัด | " + " | ".join(s["method"] for s in summaries) + " |",
             "| --- | " + " | ".join("---" for _ in summaries) + " |"]
    metrics = [("อ่านไฟล์ได้ (จาก 40)", "processed", None), ("ชื่อถูกต้อง", "name_ok", True), ("อีเมลถูกต้อง", "email_ok", True),
               ("เบอร์โทรถูกต้อง", "phone_ok", True), ("ปีเกิดถูกต้อง", "birth_year_ok", True), ("ปีจบถูกต้อง", "grad_year_ok", True),
               ("ปีประสบการณ์ (±0.5 ปี)", "years_ok", True), ("Skills F1", "skills_f1", False),
               ("คะแนนรายเกณฑ์ตรงเฉลย", "criterion_exact", True), ("คะแนนรายเกณฑ์ห่าง ≤ 1 ระดับ", "criterion_within_50", True),
               ("ผ่าน/ไม่ผ่านเกณฑ์บังคับตรงเฉลย", "must_pass_ok", True), ("หลักฐานพบใน CV ทุกข้อ", "evidence_verified", True),
               ("Spearman ρ (อันดับ)", "spearman_mean", False), ("NDCG@5", "ndcg5_mean", False),
               ("วินาทีต่อ CV", "sec_per_cv", False), ("USD ต่อ CV", "usd_per_cv", False)]
    for label, key, pct in metrics:
        vals = []
        for s in summaries:
            v = s.get(key)
            vals.append(str(v) if pct is None and v is not None else fmt(v, bool(pct)) if v is not None else "-")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    if cf:
        lines += ["", "## ทดสอบความเป็นธรรม (สลับชื่อ/เพศ ใน CV เดิม)", "", "| วิธี | โหมด | คู่ | ต่างเฉลี่ย | ต่างสูงสุด |", "| --- | --- | --- | --- | --- |"]
        for m, res in cf.items():
            for k, v in res.items():
                lines.append(f"| {m} | {k} | {v['pairs']} | {fmt(v['mean_abs_diff'])} | {fmt(v['max_abs_diff'])} |")
    lines += ["", "หมายเหตุ: ไฟล์ที่วิธีใดอ่านไม่ได้ (เช่น rule-based กับภาพสแกน) นับคะแนน 0 ในการจัดอันดับ",
              "TF-IDF ใช้เฉพาะการจัดอันดับ (ไม่ดึงข้อมูล)"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rules", "llm", "all"], default="rules")
    ap.add_argument("--limit", type=int, default=None, help="CVs per job (for a cheap trial run)")
    args = ap.parse_args()
    modes = ["rules", "llm"] if args.mode == "all" else [args.mode]
    if "llm" in modes and not llm.api_available():
        sys.exit("ANTHROPIC_API_KEY is not set")

    OUT.mkdir(parents=True, exist_ok=True)
    summaries, cf = [], {}
    for m in modes:
        print(f"== {m}")
        s, df = evaluate_method(m, args.limit)
        df.to_csv(OUT / f"per_cv_{m}.csv", index=False, encoding="utf-8-sig")
        summaries.append(s)
        cf[m] = counterfactual(m)
    summaries.append(evaluate_tfidf(args.limit))
    (OUT / "summary.json").write_text(json.dumps({"summaries": summaries, "counterfactual": cf}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    md = report(summaries, cf)
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
