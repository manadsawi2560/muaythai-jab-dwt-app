# app.py
import os, json, uuid, time
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yaml

from src.pipeline import run_pipeline
from src.features import _pivot_landmarks
from src.renderer import draw_skeleton  # ฟังก์ชันวาดเส้น

# ---------------- Page setup ----------------
st.set_page_config(page_title="MuayThai Jab DWT (Tempo-Invariant)", layout="wide")
st.title("🥊 MuayThai Jab Analyzer (BlazePose + DTW) — Tempo-Invariant")

# ---------------- Paths ----------------
ROOT = Path(__file__).parent
CFG_PATH = ROOT / "configs" / "jab_left.yaml"
UPLOADS_DIR = ROOT / "uploads"
OUTPUTS_DIR = ROOT / "outputs"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# ---------------- Session state ----------------
defaults = {
    "coach_path": None,
    "student_path": None,
    "res": None,              # dict ผลลัพธ์จาก pipeline
    "run_id": None,

    # viewer state
    "viewer_ready": False,
    "framesA": None, "framesB": None,   # dict {frame_idx: np.ndarray(BGR)}
    "wA": 0, "hA": 0, "wB": 0, "hB": 0,
    "blankA": None, "blankB": None,
    "coach_xyz": None, "student_xyz": None,
    "path_pairs": None,
    "colors": None,
    "dtw_idx": 0,            # index ปัจจุบันของเฟรมใน DTW path
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- Helpers ----------------
def _save_to_uploads(upfile, prefix: str) -> str:
    ext = Path(upfile.name).suffix.lower() or ".mp4"
    fname = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
    fpath = UPLOADS_DIR / fname
    with open(fpath, "wb") as f:
        f.write(upfile.read())
    return str(fpath)

def _preload_frames(video_path: str, needed_indices: set):
    """อ่านทีละเฟรม (เดินหน้า) ถึง max index ที่ต้องใช้ แล้วเก็บเฉพาะเฟรมที่อยู่ใน needed_indices"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    frames = {}
    idx, max_need = 0, (max(needed_indices) if needed_indices else -1)
    while idx <= max_need:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if idx in needed_indices:
            frames[idx] = fr.copy()
        idx += 1
    cap.release()
    blank = np.ones((h, w, 3), dtype=np.uint8) * 255
    return frames, w, h, blank

def _reset_all():
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

def _draw_scores_badge(img: np.ndarray, scores: dict, origin=(10, 28)) -> None:
    """วาดคะแนน (Final + รายส่วน) ที่มุมภาพ"""
    if not scores:
        return
    fs = 0.65
    th = 2
    y = origin[1]
    # พื้นหลังโปร่งบาง ๆ
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (300, 110), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    # ข้อความ
    final_s = float(scores.get("final_score", 0.0))
    cv2.putText(img, f"Score {final_s:.1f}", (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th)
    y += 24
    if "punch" in scores:
        cv2.putText(img, f"Punch {scores['punch']['score']:.1f}", (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 180, 40), th)
        y += 22
    if "core" in scores:
        cv2.putText(img, f"Core  {scores['core']['score']:.1f}", (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 120, 240), th)
        y += 22
    if "feet" in scores:
        cv2.putText(img, f"Feet  {scores['feet']['score']:.1f}", (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, fs, (220, 60, 200), th)

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ---------------- Uploaders ----------------
col1, col2 = st.columns(2)
coach_file = col1.file_uploader("โค้ช (mp4/mov/mkv)", type=["mp4", "mov", "mkv"], key="coach_upl")
student_file = col2.file_uploader("ผู้เรียน (mp4/mov/mkv)", type=["mp4", "mov", "mkv"], key="student_upl")

run_col, reset_col = st.columns([1, 1])
if reset_col.button("รีเซ็ตหน้า", type="secondary"):
    _reset_all()

if run_col.button("เริ่มวิเคราะห์", type="primary"):
    if not coach_file or not student_file:
        st.error("กรุณาอัปโหลดวิดีโอทั้งสองฝั่งก่อน")
    else:
        try:
            coach_path = _save_to_uploads(coach_file, "coach")
            student_path = _save_to_uploads(student_file, "student")
            st.session_state["coach_path"] = coach_path
            st.session_state["student_path"] = student_path
            st.session_state["run_id"] = uuid.uuid4().hex[:8]

            with st.spinner("กำลังประมวลผล..."):
                res = run_pipeline(coach_path, student_path, str(CFG_PATH), outputs_dir=str(OUTPUTS_DIR))
            st.session_state["res"] = res
            st.session_state["dtw_idx"] = 0
            # รีเซ็ต viewer เพื่อเตรียมข้อมูลใหม่
            st.session_state["viewer_ready"] = False
            st.success("เสร็จแล้ว ✅")
        except Exception as e:
            st.exception(e)

# ---------------- Results ----------------
res = st.session_state["res"]
if res:
    st.subheader("ผลลัพธ์การวิเคราะห์")
    st.json(res, expanded=False)

    # วิดีโอ (ซ้าย–ขวา) — มีคะแนนซ้อนในคลิปแล้ว
    video_path = res.get("outputs", {}).get("video")
    if video_path and os.path.exists(video_path):
        st.subheader("วิดีโอผลลัพธ์ (ซ้าย–ขวา)")
        st.video(video_path)
        with open(video_path, "rb") as vf:
            st.download_button("ดาวน์โหลดวิดีโอ", data=vf, file_name=Path(video_path).name, mime="video/mp4")
    else:
        st.info("วิดีโอเล่นไม่ได้ในเบราว์เซอร์นี้ หรือไม่มีไฟล์ — ดูด้วย Frame Viewer ด้านล่างแทน")

    # ---------------- Frame-by-Frame Viewer (ไม่ใช้วิดีโอ) ----------------
    st.subheader("Frame-by-Frame Viewer (ซ้าย–ขวาพร้อม skeleton + คะแนนบนภาพ)")

    # เตรียมทรัพยากรครั้งเดียวหลังได้ผลลัพธ์
    if not st.session_state["viewer_ready"]:
        try:
            cfg = yaml.safe_load(open(CFG_PATH, "r", encoding="utf-8"))
            st.session_state["colors"] = cfg["colors"]

            coach_csv = OUTPUTS_DIR / "coach_landmarks.csv"
            student_csv = OUTPUTS_DIR / "student_landmarks.csv"
            coach_df = pd.read_csv(coach_csv)
            student_df = pd.read_csv(student_csv)
            coach_xyz, _ = _pivot_landmarks(coach_df)
            student_xyz, _ = _pivot_landmarks(student_df)
            st.session_state["coach_xyz"] = coach_xyz
            st.session_state["student_xyz"] = student_xyz

            path_pairs = res.get("path") or []
            st.session_state["path_pairs"] = path_pairs

            needA = {i for i, _ in path_pairs}
            needB = {j for _, j in path_pairs}
            framesA, wA, hA, blankA = _preload_frames(st.session_state["coach_path"], needA)
            framesB, wB, hB, blankB = _preload_frames(st.session_state["student_path"], needB)
            st.session_state.update({
                "framesA": framesA, "wA": wA, "hA": hA, "blankA": blankA,
                "framesB": framesB, "wB": wB, "hB": hB, "blankB": blankB,
                "viewer_ready": True
            })
        except Exception as e:
            st.exception(e)

    path_pairs = st.session_state["path_pairs"] or []
    path_len = len(path_pairs)

    if path_len > 0:
        # ---- Controls ที่ Sidebar ----
        st.sidebar.header("Viewer controls")
        viewer_width = st.sidebar.slider("ความกว้างภาพต่อฝั่ง (px)", 280, 960, 480, 20, key="vw")

        # ปุ่มเลื่อนเฟรมแทนสไลเดอร์
        step = st.sidebar.radio("Step", [1, 5], horizontal=True, index=0)
        cprev, cnext = st.sidebar.columns(2)
        if cprev.button("◀︎ Prev"):
            st.session_state["dtw_idx"] = _clamp(st.session_state["dtw_idx"] - step, 0, path_len - 1)
        if cnext.button("Next ▶︎"):
            st.session_state["dtw_idx"] = _clamp(st.session_state["dtw_idx"] + step, 0, path_len - 1)
        cstart, cend = st.sidebar.columns(2)
        if cstart.button("⏮️ First"):
            st.session_state["dtw_idx"] = 0
        if cend.button("Last ⏭️"):
            st.session_state["dtw_idx"] = path_len - 1

        # แสดง index ปัจจุบัน
        st.sidebar.write(f"ตำแหน่งใน DTW path: **{st.session_state['dtw_idx']} / {path_len-1}**")

        # ---- ดึงคู่เฟรมตาม index ปัจจุบัน ----
        idx = st.session_state["dtw_idx"]
        iA, iB = path_pairs[idx]

        frA = st.session_state["framesA"].get(iA, st.session_state["blankA"]).copy()
        frB = st.session_state["framesB"].get(iB, st.session_state["blankB"]).copy()

        # วาด skeleton + แปะคะแนนบนภาพ
        colors = st.session_state["colors"]
        coach_xyz = st.session_state["coach_xyz"]
        student_xyz = st.session_state["student_xyz"]
        if iA < coach_xyz.shape[0]:
            frA = draw_skeleton(frA, coach_xyz[iA, :, :2], colors)
        if iB < student_xyz.shape[0]:
            frB = draw_skeleton(frB, student_xyz[iB, :, :2], colors)

        # วาดคะแนน (ใช้คะแนนรวมของทั้งคลิป; ถ้าต้องการต่อเฟรม ต้องคำนวณเพิ่ม)
        _draw_scores_badge(frA, res.get("scores", {}))
        _draw_scores_badge(frB, res.get("scores", {}))

        # แสดงสองภาพซ้าย-ขวา
        c1, c2 = st.columns(2, gap="small")
        c1.image(cv2.cvtColor(frA, cv2.COLOR_BGR2RGB), caption=f"Coach frame {iA}", width=viewer_width)
        c2.image(cv2.cvtColor(frB, cv2.COLOR_BGR2RGB), caption=f"Student frame {iB}", width=viewer_width)

        st.caption("กดปุ่ม Prev/Next หรือเลือก step เป็น 1/5 เพื่อเลื่อนดูเฟรมทีละขั้น — ไม่ติดปัญหา codec ของเบราว์เซอร์")
    else:
        st.info("ไม่มีข้อมูล DTW path ให้แสดง (ตรวจว่า pipeline บันทึก field 'path' แล้วหรือไม่)")
else:
    st.info("อัปโหลดวิดีโอทั้งสองฝั่งแล้วกด ‘เริ่มวิเคราะห์’ เพื่อดูผลลัพธ์")
