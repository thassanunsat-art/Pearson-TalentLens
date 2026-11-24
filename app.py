"""TalentLens - Streamlit UI for first-round CV screening.

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from talentlens import llm, store
from talentlens.pipeline import parse_jd, screen_files
from talentlens.schemas import CandidateResult, JobRequirement, JobSpec

DATA = Path(__file__).parent / "data"
SAMPLE_JOBS = {"นักวิเคราะห์ข้อมูล (Data Analyst)": "data_analyst", "เจ้าหน้าที่บัญชี (Accountant)": "accountant"}
LEVEL_ICON = {"error": "🔴", "warning": "🟠", "info": "🔵"}

st.set_page_config(page_title="TalentLens - คัดกรอง CV", page_icon="🔎", layout="wide")
st.title("🔎 TalentLens")
st.caption("ระบบช่วยคัดกรองใบสมัครงานรอบแรก: อ่าน CV → ดึงข้อมูล → ให้คะแนนพร้อมหลักฐาน → HR ตัดสิน")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("ตั้งค่า")
    has_key = llm.api_available()
    mode_label = st.radio(
        "วิธีอ่านและให้คะแนน",
        ["Claude API (แนะนำ)", "Rule-based (ออฟไลน์)"],
        index=0 if has_key else 1,
        help="Rule-based ใช้ regex + คำสำคัญ ไม่ต้องมี API key แต่อ่านไฟล์สแกนไม่ได้และไม่เข้าใจคำพ้องความหมาย",
    )
    mode = "llm" if mode_label.startswith("Claude") else "rules"
    if mode == "llm" and not has_key:
        st.error("ไม่พบ ANTHROPIC_API_KEY - ตั้งค่าในไฟล์ .env หรือ environment ก่อน")
    if mode == "llm":
        st.caption(f"โมเดล: `{llm.MODEL}` · effort: `{llm.EFFORT}`")
    blind = st.toggle("โหมดปิดชื่อ (blind screening)", value=True,
                      help="ซ่อนชื่อ เพศ อายุ ช่องทางติดต่อ และชื่อสถาบัน ก่อนส่งให้โมเดลให้คะแนน")
    reviewer = st.text_input("ชื่อผู้ตรวจ (สำหรับ audit log)", value="HR")

# --------------------------------------------------------- step 1: the job
st.subheader("1. ตำแหน่งงานและเกณฑ์")
src = st.radio("แหล่ง JD", ["ตัวอย่าง", "วางข้อความ JD"], horizontal=True)
if src == "ตัวอย่าง":
    label = st.selectbox("เลือกตำแหน่งตัวอย่าง", list(SAMPLE_JOBS))
    job_key = SAMPLE_JOBS[label]
    jd_text = (DATA / "jd" / f"{job_key}.txt").read_text(encoding="utf-8")
    with st.expander("ดูประกาศรับสมัครงาน"):
        st.text(jd_text)
    if st.session_state.get("job_source") != job_key:
        st.session_state.job = JobSpec.model_validate_json((DATA / "jd" / f"{job_key}.json").read_text(encoding="utf-8"))
        st.session_state.job_source = job_key
        st.session_state.pop("results", None)
else:
    job_key = None
    jd_text = st.text_area("วางประกาศรับสมัครงาน (JD)", height=220)
    if st.button("แยกเกณฑ์จาก JD", disabled=not jd_text.strip()):
        with st.spinner("กำลังอ่าน JD..."):
            try:
                st.session_state.job = parse_jd(jd_text, mode)
                st.session_state.job_source = "custom"
                st.session_state.pop("results", None)
            except llm.LLMError as e:
                st.error(str(e))

job: JobSpec | None = st.session_state.get("job")
if job:
    st.markdown(f"**{job.title}** - แก้เกณฑ์และน้ำหนักได้ในตาราง (must = ต้องมี, nice = มีก็ดี)")
    df = pd.DataFrame([{**r.model_dump(), "keywords": ", ".join(r.keywords)} for r in job.requirements])
    edited = st.data_editor(
        df, num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "id": st.column_config.TextColumn("รหัส", width="small"),
            "criterion": st.column_config.TextColumn("เกณฑ์", width="large"),
            "type": st.column_config.SelectboxColumn("ประเภท", options=["must", "nice"], width="small"),
            "weight": st.column_config.NumberColumn("น้ำหนัก", min_value=1, max_value=5, step=1, width="small"),
            "min_years": st.column_config.NumberColumn("ปีขั้นต่ำ", min_value=0.0, step=0.5, width="small"),
            "keywords": st.column_config.TextColumn("คำสำคัญ (ใช้ในโหมด rule-based)"),
        },
    )
    reqs = []
    for i, row in edited.fillna({"min_years": 0}).iterrows():
        if not str(row.get("criterion") or "").strip():
            continue
        reqs.append(JobRequirement(
            id=str(row["id"] or f"r{i + 1}"), criterion=str(row["criterion"]), type=row["type"] or "nice",
            weight=int(row["weight"] or 1), min_years=float(row["min_years"]) or None,
            keywords=[k.strip() for k in str(row["keywords"] or "").split(",") if k.strip()],
        ))
    job = JobSpec(title=job.title, requirements=reqs)

# --------------------------------------------------------- step 2: the CVs
st.subheader("2. ใบสมัคร (CV)")
cv_src = st.radio("แหล่ง CV", ["อัปโหลด", "ใช้ CV ตัวอย่างของตำแหน่งนี้"], horizontal=True, disabled=job is None)
files: list[tuple[str, bytes]] = []
if cv_src == "อัปโหลด":
    ups = st.file_uploader("PDF, DOCX, JPG, PNG (หลายไฟล์ได้)", type=["pdf", "docx", "jpg", "jpeg", "png"], accept_multiple_files=True)
    files = [(u.name, u.getvalue()) for u in ups or []]
elif job_key:
    paths = sorted((DATA / "cv" / job_key).glob("*.*"))
    if mode == "rules":
        paths = [p for p in paths if p.suffix != ".jpg"]
        st.caption("โหมด rule-based ข้ามไฟล์รูปสแกน (.jpg)")
    n = st.slider("จำนวน CV", 1, len(paths), min(8, len(paths)))
    files = [(p.name, p.read_bytes()) for p in paths[:n]]
else:
    st.info("CV ตัวอย่างมีเฉพาะตำแหน่งตัวอย่าง")

run = st.button("🚀 คัดกรอง", type="primary", disabled=not (job and files and job.requirements) or (mode == "llm" and not llm.api_available()))
if run:
    bar = st.progress(0.0, text="เริ่ม...")
    results, errors = screen_files(files, job, mode=mode, blind=blind,
                                   on_progress=lambda i, n, name: bar.progress(i / n, text=f"{i}/{n} · {name}"))
    bar.empty()
    st.session_state.results, st.session_state.errors = results, errors
    st.session_state.result_job, st.session_state.result_blind = job, blind

# -------------------------------------------------------- step 3: results
results: list[CandidateResult] | None = st.session_state.get("results")
if results is not None:
    st.subheader("3. ผลการจัดอันดับ")
    for name, err in st.session_state.get("errors", []):
        st.warning(f"{name}: {err}")
    if not results:
        st.stop()
    rblind = st.session_state.result_blind
    table = store.results_table(results, rblind)
    tok_in = sum(r.input_tokens for r in results)
    tok_out = sum(r.output_tokens for r in results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV ที่อ่านได้", len(results))
    c2.metric("ผ่านเกณฑ์บังคับ", sum(1 for r in results if not r.must_have_failed))
    c3.metric("เวลาเฉลี่ยต่อ CV", f"{sum(r.seconds for r in results) / len(results):.1f} วิ")
    c4.metric("ค่า API โดยประมาณ", f"${llm.Usage(tok_in, tok_out).cost_usd:.3f}")
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={"คะแนนรวม": st.column_config.ProgressColumn("คะแนนรวม", min_value=0, max_value=100, format="%.1f")})

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ ดาวน์โหลด CSV", table.to_csv(index=False).encode("utf-8-sig"), "shortlist.csv", "text/csv")
    d2.download_button("⬇️ ดาวน์โหลด Excel", store.to_excel_bytes(table), "shortlist.xlsx")

    st.subheader("4. ตรวจรายคนและตัดสิน")
    for rank, r in enumerate(results, 1):
        who = "(ปิดบัง)" if rblind else (r.profile.full_name or r.file_name)
        badge = "✅" if not r.must_have_failed else "❌"
        n_flags = sum(1 for f in r.flags if f.level != "info")
        with st.expander(f"#{rank} {badge} {who} · {r.file_name} · {r.total_score:.1f} คะแนน" + (f" · ⚠️ {n_flags}" if n_flags else "")):
            left, right = st.columns([3, 2])
            with left:
                for s in r.criterion_scores:
                    icon = {100: "🟢", 50: "🟡", 0: "⚪"}.get(s.score, "⚪")
                    st.markdown(f"{icon} **{s.criterion}** `{s.type}` · w{s.weight} → **{s.score}**")
                    st.caption(s.reasoning)
                    for q in s.evidence:
                        st.markdown(f"> {q}" + ("" if s.evidence_verified else "  ⚠️ *ไม่พบใน CV*"))
                if r.flags:
                    st.markdown("**ข้อควรตรวจ**")
                    for f in r.flags:
                        st.markdown(f"{LEVEL_ICON[f.level]} {f.message}")
            with right:
                st.markdown(f"**ประสบการณ์รวม:** {r.total_years_exp} ปี  \n**ทักษะ (normalized):** {', '.join(r.skills_normalized) or '-'}")
                prof = r.profile.model_dump()
                if rblind:
                    for k in ("full_name", "email", "phone", "gender", "birth_year"):
                        prof[k] = "(ปิดบัง)"
                with st.popover("ข้อมูลที่ดึงได้ (JSON)"):
                    st.json(prof)
                with st.popover("ข้อความที่ส่งให้โมเดลให้คะแนน"):
                    st.text(r.redacted_text)
                decision = st.radio("การตัดสิน", ["ผ่าน", "รอพิจารณา", "ไม่ผ่าน"], horizontal=True, key=f"d_{r.candidate_id}", index=1)
                reason = st.text_input("เหตุผล", key=f"why_{r.candidate_id}")
                if st.button("บันทึก", key=f"save_{r.candidate_id}"):
                    store.save_decision(st.session_state.result_job.title, r, decision, reason, reviewer)
                    st.success("บันทึกแล้ว")

    with st.expander("📜 Audit log (การตัดสินที่บันทึกไว้)"):
        st.dataframe(store.load_decisions(), width="stretch", hide_index=True)
