# TalentLens: Document Intelligence สำหรับคัดกรองใบสมัครงาน

TalentLens อ่าน CV (PDF, Word, ภาพสแกน ทั้งไทยและอังกฤษ) ดึงข้อมูลผู้สมัครออกมาเป็นโครงสร้าง
ให้คะแนนเทียบกับประกาศรับสมัครงาน (JD) **พร้อมประโยคหลักฐานที่ตรวจแล้วว่ามีอยู่ใน CV จริง**
แล้วจัดอันดับให้ฝ่ายบุคคลตัดสิน โดยคนยังเป็นผู้ตัดสินสุดท้ายเสมอ

> งานกลุ่มรายวิชา Document Intelligence · โดเมน: งานสรรหาบุคลากร (HR Recruitment)
> เอกสารออกแบบฉบับเต็ม: *(ใส่ลิงก์เอกสารออกแบบของกลุ่ม)* · วิดีโอนำเสนอ: *(ใส่ลิงก์ YouTube)*

## ปัญหา

การคัดกรอง CV รอบแรกทำด้วยมือทั้งหมด จึง **ช้า** (อ่านทีละฉบับ รูปแบบไม่เหมือนกัน บางฉบับเป็นภาพสแกน)
**ไม่สม่ำเสมอ** (แต่ละคนใช้เกณฑ์ต่างกัน) **พลาดคนที่เหมาะ** (จับคำตรงตัว เช่น CV เขียน "pandas" แต่ JD เขียน "Python")
**ตรวจย้อนหลังไม่ได้** (ไม่บันทึกเหตุผล) และ **มีอคติ** (เห็นชื่อ เพศ อายุ ก่อนเห็นทักษะ)

## กระบวนการ

```mermaid
flowchart LR
  subgraph asis[ปัจจุบัน: คนทำทุกขั้น]
    A1[รับ CV] --> A2[เปิดอ่านทีละฉบับ] --> A3[จดลง Excel] --> A4[ตัดสินเอง<br/>ไม่บันทึกเหตุผล] --> A5[ส่ง shortlist]
  end
  subgraph tobe[ใหม่: ระบบทำงานซ้ำ คนตัดสินใจ]
    B1[อัปโหลด CV + JD] --> B2[ดึงข้อมูล + ปิดชื่อ] --> B3[ให้คะแนนพร้อมหลักฐาน] --> B4[HR ตรวจ แก้ และตัดสิน] --> B5[ส่งออก shortlist + audit log]
  end
```

## สถาปัตยกรรม

```mermaid
flowchart LR
  CV[ไฟล์ CV<br/>PDF / DOCX / JPG] --> I[1. Ingest<br/>PyMuPDF, python-docx<br/>สแกนหรือข้อความเพี้ยน → ภาพ]
  I --> E[2. Extract<br/>Claude vision + structured outputs<br/>→ CandidateProfile]
  E --> R[3. Redact<br/>ซ่อนชื่อ เพศ อายุ ติดต่อ สถาบัน]
  JD[ประกาศงาน JD] --> P[JD parser<br/>→ เกณฑ์ must / nice + น้ำหนัก]
  R --> S[4. Score<br/>0 / 50 / 100 ต่อเกณฑ์ + quote]
  P --> S
  S --> V[5. Validate<br/>quote ต้องพบใน CV<br/>วันที่ / อายุ สมเหตุสมผล]
  V --> H[6. HR review<br/>Streamlit]
  H --> X[7. Export<br/>CSV / Excel + SQLite audit log]
```

หลักการออกแบบที่สำคัญ

- **LLM ตัดสินรายเกณฑ์ ส่วนโค้ดคำนวณคะแนนรวม**: `คะแนนรวม = Σ(น้ำหนัก × คะแนนเกณฑ์) / Σ น้ำหนัก` ได้ผลคงที่และตรวจสอบได้
- **กันโมเดลแต่งข้อมูล (hallucination)**: ทุก quote ถูกตรวจด้วย fuzzy match กับข้อความ CV ถ้าไม่พบ คะแนนเกณฑ์นั้นเป็น 0 และติดธง
- **Blind screening**: โมเดลที่ให้คะแนนเห็นเฉพาะข้อความที่ปิดข้อมูลระบุตัวตนแล้ว
- **รองรับเอกสารไทยจริง**: แปลงปี พ.ศ. แก้สระอำที่แตก (`ํา` → `ำ`) และตรวจ text layer ที่เพี้ยนจากฟอนต์ไทย แล้วส่งเป็นภาพแทน
- **มีโหมดออฟไลน์ (rule-based)**: ใช้ regex + พจนานุกรมทักษะ ใช้เดโมได้โดยไม่มี API key และเป็น baseline ตอนประเมินผล

## วิธีติดตั้งและรัน

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # ใส่ ANTHROPIC_API_KEY (ไม่ใส่ก็ได้ ถ้าจะใช้โหมด rule-based)
streamlit run app.py
```

> **Windows:** ถ้า `pip install anthropic` ล้มเหลวเพราะ path ยาวเกิน ให้สร้าง venv ในโฟลเดอร์ที่ path สั้น
> (เช่น `C:\venvs\talentlens`) หรือเปิด Long Paths ใน Windows

ในแอป: เลือกตำแหน่งตัวอย่าง (หรือวาง JD เอง) → ปรับเกณฑ์และน้ำหนัก → อัปโหลด CV หรือใช้ CV ตัวอย่าง →
กด **คัดกรอง** → ดูอันดับ หลักฐาน และข้อควรตรวจ → บันทึกการตัดสิน → ดาวน์โหลด CSV/Excel

ค่าเริ่มต้นใช้โมเดล `claude-opus-5-5` ที่ effort `medium` (เปลี่ยนได้ด้วย `TALENTLENS_MODEL` และ `TALENTLENS_EFFORT`)
และเปิด server-side fallback ไว้ ถ้าโมเดลปฏิเสธคำขอ API จะลองใหม่ด้วยโมเดลสำรองในคำขอเดียวกัน

## ชุดข้อมูลและการประเมินผล

ทุกอย่างใน `data/` เป็น **ข้อมูลสมมติ** ไม่มี CV ของบุคคลจริง สร้างได้ซ้ำด้วย `python data/generate_synthetic.py`

| | Data Analyst | Accountant |
| --- | --- | --- |
| JD | `data/jd/data_analyst.txt` | `data/jd/accountant.txt` |
| CV | 20 ฉบับ: PDF 10, DOCX 4, ภาพสแกน 6 · ไทย/อังกฤษอย่างละครึ่ง · 3 เลย์เอาต์ | เหมือนกัน |
| เฉลย | `data/ground_truth/*.json` (ข้อมูลส่วนตัว ประสบการณ์ ทักษะ คะแนนรายเกณฑ์) | |
| ทดสอบความเป็นธรรม | `data/cv_counterfactual/` CV เดิมที่สลับชื่อและเพศ 10 คู่ | |

CV ตั้งใจใส่ความยากแบบที่เจอในงานจริง ได้แก่ คำพ้องความหมาย ("pandas" = Python, "BigQuery" = SQL),
ทักษะระดับบางส่วน (เช่น เคยเรียน Python เบื้องต้น, Microsoft Access), งานที่ไม่เกี่ยวข้องกับตำแหน่ง
(หลอกการนับปีประสบการณ์), รูปแบบวันที่หลายแบบ (พ.ศ./ค.ศ., `ก.ค. 2569`, `06/2564`, `Jun 2021`)
และเลย์เอาต์ 2 คอลัมน์

```bash
python eval/run_eval.py               # rule-based + TF-IDF (ฟรี)
python eval/run_eval.py --mode all    # + Claude API (เสียค่า API; ผลถูก cache ใน eval/cache/)
python -m pytest -q                   # unit + integration tests (ไม่เรียก API)
```

ผลล่าสุดอยู่ใน [`eval/results/REPORT.md`](eval/results/REPORT.md) ตัวชี้วัดหลักได้แก่ ความถูกต้องรายฟิลด์, Skills F1,
คะแนนรายเกณฑ์ตรงเฉลย, Spearman ρ และ NDCG@5 ของอันดับ, สัดส่วนหลักฐานที่พบใน CV, ความต่างของคะแนนเมื่อสลับชื่อ/เพศ,
เวลาและค่า API ต่อ CV

## โครงสร้าง repo

```
app.py                      Streamlit UI
talentlens/
  ingest.py                 อ่านไฟล์ → ข้อความ/ภาพ, แก้ข้อความไทยเพี้ยน
  extract.py                CV → CandidateProfile (Claude หรือ rule-based)
  jd_parser.py              JD → เกณฑ์ must/nice
  redact.py                 blind screening
  score.py                  ให้คะแนนรายเกณฑ์, ตรวจหลักฐาน, คะแนนรวม
  validate.py               กฎความสมเหตุสมผล
  pipeline.py               ต่อทุกขั้นเข้าด้วยกัน
  store.py                  audit log (SQLite) + export
  llm.py, schemas.py, lexicon.py
data/                       JD, CV สมมติ, เฉลย, ตัวสร้างข้อมูล
eval/                       run_eval.py, baseline.py (TF-IDF), results/
docs/video_script.md        โครงวิดีโอนำเสนอ 20 นาที
tests/
```

## ข้อจำกัดและจริยธรรม

- ระบบเป็น **เครื่องมือช่วยตัดสินใจ** ไม่ใช่ผู้ตัดสิน ทุกการตัดสินบันทึกชื่อผู้ตรวจและเหตุผล
- ผลประเมินมาจากข้อมูลสมมติที่กลุ่มสร้างเอง CV จริงมีความหลากหลายมากกว่านี้
- การใช้งานจริงต้องแจ้งผู้สมัครว่าใช้ AI ช่วยคัดกรอง กำหนดระยะเวลาเก็บข้อมูล และส่งข้อมูลให้ผู้ให้บริการ API ตาม PDPA
- ห้าม commit CV จริง ไฟล์ `.env` หรือ API key ขึ้น repo
