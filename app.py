# app.py
import os, json, time, uuid
from pathlib import Path
import streamlit as st

from src.pipeline import run_pipeline

st.set_page_config(page_title="MuayThai Jab DWT (Tempo-Invariant)", layout="wide")
st.title("🥊 MuayThai Jab Analyzer (BlazePose + DTW) — Tempo-Invariant")

# ---------- paths ----------
ROOT = Path(__file__).parent
CFG_PATH = ROOT / "configs" / "jab_left.yaml"
UPLOADS_DIR = ROOT / "uploads"
OUTPUTS_DIR = ROOT / "outputs"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ---------- session state init ----------
for k, v in {
    "coach_path": None,
    "student_path": None,
    "res": None,           # dict ผลลัพธ์ pipeline (เก็บตลอดการใช้งาน)
    "run_id": None,        # ไว้ทำชื่อไฟล์ไม่ชนกัน
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- uploader ----------
col1, col2 = st.columns(2)
coach_file = col1.file_uploader("โค้ช (mp4/mov/mkv)", type=["mp4","mov","mkv"], key="coach_upl")
student_file = col2.file_uploader("ผู้เรียน (mp4/mov/mkv)", type=["mp4","mov","mkv"], key="student_upl")

def save_to_uploads(upfile, prefix):
    ext = Path(upfile.name).suffix.lower() or ".mp4"
    fname = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    fpath = UPLOADS_DIR / fname
    with open(fpath, "wb") as f:
        f.write(upfile.read())
    return str(fpath)

# ---------- action buttons ----------
run_col, reset_col = st.columns([1,1])
run_clicked = run_col.button("เริ่มวิเคราะห์", type="primary")
reset_clicked = reset_col.button("รีเซ็ตหน้า", type="secondary")

if reset_clicked:
    st.session_state["coach_path"] = None
    st.session_state["student_path"] = None
    st.session_state["res"] = None
    st.session_state["run_id"] = None
    st.rerun()

# กดประมวลผล => เซฟไฟล์ถาวร + เรียก pipeline + เก็บลง session_state
if run_clicked:
    if not coach_file or not student_file:
        st.error("กรุณาอัปโหลดวิดีโอทั้งสองฝั่งก่อน")
    else:
        try:
            coach_path = save_to_uploads(coach_file, "coach")
            student_path = save_to_uploads(student_file, "student")
            st.session_state["coach_path"] = coach_path
            st.session_state["student_path"] = student_path
            st.session_state["run_id"] = uuid.uuid4().hex[:8]

            with st.spinner("กำลังประมวลผล..."):
                res = run_pipeline(coach_path, student_path, str(CFG_PATH), outputs_dir=str(OUTPUTS_DIR))
            st.session_state["res"] = res
            st.success("เสร็จแล้ว ✅")
        except Exception as e:
            st.exception(e)

# ---------- show results from session_state (ไม่หายเมื่อเลื่อนสไลเดอร์) ----------
res = st.session_state.get("res")
if res:
    st.subheader("ผลลัพธ์การวิเคราะห์")
    st.json(res, expanded=False)

    # วิดีโอผลลัพธ์
    video_path = res.get("outputs", {}).get("video")
    if video_path and os.path.exists(video_path):
        st.subheader("วิดีโอผลลัพธ์ (Side-by-side along DTW path)")
        # st.video รองรับ path ได้ตรง ๆ
        st.video(video_path)
        # ปุ่มดาวน์โหลดไฟล์
        with open(video_path, "rb") as vf:
            st.download_button("ดาวน์โหลดวิดีโอ", data=vf, file_name=Path(video_path).name, mime="video/mp4")
    else:
        st.warning("ไม่พบไฟล์วิดีโอผลลัพธ์")
        if res.get("errors", {}).get("render"):
            st.error(f"Render error: {res['errors']['render']}")

    # สไลเดอร์ดูความคืบหน้า DTW (ใช้ข้อมูลในหน่วยความจำ ไม่อ่านไฟล์ใหม่)
    st.subheader("เลื่อนดูทีละ progress/frame (ตาม DTW path)")
    path_len = int(res.get("path_len", 0))
    if path_len > 0:
        idx = st.slider("ตำแหน่งใน DTW path", 0, path_len - 1, 0)
        st.caption("สไลด์เพื่อดูคู่ดัชนีเฟรมของโค้ช/ผู้เรียน ที่ถูกจับคู่ด้วย DTW")
        # แสดงคู่ index เฉย ๆ (หากต้องการโชว์เฟรมจริง ต้องเขียน renderer เฉพาะเฟรม)
        st.write({"dtw_index": idx})
    else:
        st.info("ไม่มีข้อมูล DTW path")
else:
    st.info("อัปโหลดวิดีโอทั้งสองฝั่งแล้วกด 'เริ่มวิเคราะห์' เพื่อดูผลลัพธ์")
