# src/renderer.py  (no random seeking; preload needed frames)
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set
from .pose_extractor import NAME2IDX

PUNCH_CONNS = [("LEFT_SHOULDER","LEFT_ELBOW"),("LEFT_ELBOW","LEFT_WRIST")]
CORE_CONNS  = [("LEFT_SHOULDER","RIGHT_SHOULDER"),("LEFT_HIP","RIGHT_HIP"),
               ("LEFT_SHOULDER","LEFT_HIP"),("RIGHT_SHOULDER","RIGHT_HIP")]
FEET_CONNS  = [("LEFT_HIP","LEFT_KNEE"),("LEFT_KNEE","LEFT_ANKLE"),
               ("RIGHT_HIP","RIGHT_KNEE"),("RIGHT_KNEE","RIGHT_ANKLE")]
REST_CONNS  = []  # เติมได้ถ้าอยากวาดส่วนอื่นจาง ๆ

def _pt(img_w, img_h, lm):
    # ป้องกัน NaN
    x = 0.0 if np.isnan(lm[0]) else lm[0]
    y = 0.0 if np.isnan(lm[1]) else lm[1]
    return int(x*img_w), int(y*img_h)

def draw_skeleton(frame: np.ndarray, lms_xy: np.ndarray, colors: Dict[str, Tuple[int,int,int]], alpha_rest: float = 0.25) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()

    def draw_conns(conns, color, thickness=4):
        for a,b in conns:
            ia, ib = NAME2IDX[a], NAME2IDX[b]
            pa = _pt(w,h,lms_xy[ia]); pb = _pt(w,h,lms_xy[ib])
            cv2.line(out, pa, pb, color, thickness, cv2.LINE_AA)

    if REST_CONNS:
        overlay = out.copy()
        draw_conns(REST_CONNS, colors.get('rest',(180,180,180)), 2)
        cv2.addWeighted(overlay, alpha_rest, out, 1-alpha_rest, 0, out)

    draw_conns(PUNCH_CONNS, tuple(colors['punch']), 6)
    draw_conns(CORE_CONNS,  tuple(colors['core']),  6)
    draw_conns(FEET_CONNS,  tuple(colors['feet']),  6)
    return out

def _preload_needed_frames(video_path: str, needed: Set[int]):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    max_need = max(needed) if needed else -1

    frames = {}
    idx = 0
    # อ่านเรียงตั้งแต่เฟรม 0 ถึง max_need ทีละเฟรม
    while idx <= max_need:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if idx in needed:
            frames[idx] = fr.copy()
        idx += 1

    cap.release()
    # เฟรมสำรองสีขาวถ้าไม่มี
    default = np.ones((h, w, 3), dtype=np.uint8) * 255
    return frames, w, h, default

def render_side_by_side(coach_video: str, student_video: str,
                         coach_xyz: np.ndarray, student_xyz: np.ndarray,
                         path: List[Tuple[int,int]], colors: Dict[str, Tuple[int,int,int]],
                         out_path: str, out_fps: int = 30,
                         overlay_text: Dict = None,
                         impact_coach: int = None, impact_student: int = None):

    # เตรียมชุด index ที่ต้องใช้จริง ๆ
    needA = {i for i,_ in path}
    needB = {j for _,j in path}

    # โหลดเฉพาะเฟรมที่ต้องใช้ (ไม่ seek แบบสุ่ม)
    framesA, wA, hA, blankA = _preload_needed_frames(coach_video, needA)
    framesB, wB, hB, blankB = _preload_needed_frames(student_video, needB)

    width  = max(wA, wB)
    height = hA + hB

    # เขียนวิดีโอ
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {out_path}")

    for iA, iB in path:
        frA = framesA.get(iA, blankA)
        frB = framesB.get(iB, blankB)

        # กัน index เกิน
        if iA >= coach_xyz.shape[0] or iB >= student_xyz.shape[0]:
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            vw.write(canvas)
            continue

        lmsA = coach_xyz[iA, :, :2]
        lmsB = student_xyz[iB, :, :2]
        frA = draw_skeleton(frA, lmsA, colors)
        frB = draw_skeleton(frB, lmsB, colors)

        if overlay_text:
            cv2.putText(frA, f"Coach: {overlay_text.get('coach','')}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(frB, f"Student: {overlay_text.get('student','')}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        if impact_coach is not None and iA == impact_coach:
            cv2.putText(frA, "IMPACT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        if impact_student is not None and iB == impact_student:
            cv2.putText(frB, "IMPACT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        canvas[:hA, :wA] = frA
        canvas[hA:hA+hB, :wB] = frB
        vw.write(canvas)

    vw.release()
