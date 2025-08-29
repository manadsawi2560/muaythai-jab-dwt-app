# src/renderer.py
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from .pose_extractor import NAME2IDX

PUNCH_CONNS = [("LEFT_SHOULDER","LEFT_ELBOW"),("LEFT_ELBOW","LEFT_WRIST")]
CORE_CONNS  = [("LEFT_SHOULDER","RIGHT_SHOULDER"),("LEFT_HIP","RIGHT_HIP"),
               ("LEFT_SHOULDER","LEFT_HIP"),("RIGHT_SHOULDER","RIGHT_HIP")]
FEET_CONNS  = [("LEFT_HIP","LEFT_KNEE"),("LEFT_KNEE","LEFT_ANKLE"),
               ("RIGHT_HIP","RIGHT_KNEE"),("RIGHT_KNEE","RIGHT_ANKLE")]
REST_CONNS  = []  # เติมได้ถ้าต้องวาดส่วนอื่นแบบจาง

def _pt(img_w, img_h, lm):
    x = 0.0 if np.isnan(lm[0]) else lm[0]
    y = 0.0 if np.isnan(lm[1]) else lm[1]
    return int(x*img_w), int(y*img_h)

def draw_skeleton(frame: np.ndarray, lms_xy: np.ndarray,
                  colors: Dict[str, Tuple[int,int,int]], alpha_rest: float = 0.25) -> np.ndarray:
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
    while idx <= max_need:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        if idx in needed:
            frames[idx] = fr.copy()
        idx += 1

    cap.release()
    default = np.ones((h, w, 3), dtype=np.uint8) * 255
    return frames, w, h, default

def _draw_scores_badge(img: np.ndarray, scores: Dict, origin=(10, 28)) -> None:
    """วาดคะแนน (Final + รายส่วน) ที่มุมภาพ"""
    if not scores: return
    fs = 0.65
    th = 2
    y = origin[1]
    # พื้นหลังโปร่งบาง ๆ
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (300, 110), (255,255,255), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    # ข้อความ
    final_s = scores.get("final_score", 0.0)
    cv2.putText(img, f"Score {final_s:.1f}", (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), th)
    y += 24
    if "punch" in scores:
        cv2.putText(img, f"Punch {scores['punch']['score']:.1f}", (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, fs, (40,180,40), th)
        y += 22
    if "core" in scores:
        cv2.putText(img, f"Core  {scores['core']['score']:.1f}",  (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, fs, (40,120,240), th)
        y += 22
    if "feet" in scores:
        cv2.putText(img, f"Feet  {scores['feet']['score']:.1f}",  (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, fs, (220,60,200), th)

def render_side_by_side(coach_video: str, student_video: str,
                         coach_xyz: np.ndarray, student_xyz: np.ndarray,
                         path: List[Tuple[int,int]], colors: Dict[str, Tuple[int,int,int]],
                         out_path: str, out_fps: int = 30,
                         overlay_text: Dict = None,
                         impact_coach: Optional[int] = None, impact_student: Optional[int] = None,
                         overlay_scores: Optional[Dict] = None,
                         layout: str = "horizontal"):
    """
    layout: "horizontal" = ซ้าย-ขวา, "vertical" = บน-ล่าง
    """
    needA = {i for i,_ in path}
    needB = {j for _,j in path}

    framesA, wA, hA, blankA = _preload_needed_frames(coach_video, needA)
    framesB, wB, hB, blankB = _preload_needed_frames(student_video, needB)

    if layout == "horizontal":
        width  = wA + wB
        height = max(hA, hB)
    else:
        width  = max(wA, wB)
        height = hA + hB

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, out_fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {out_path}")

    for iA, iB in path:
        frA = framesA.get(iA, blankA).copy()
        frB = framesB.get(iB, blankB).copy()

        if iA < coach_xyz.shape[0]:
            frA = draw_skeleton(frA, coach_xyz[iA, :, :2], colors)
        if iB < student_xyz.shape[0]:
            frB = draw_skeleton(frB, student_xyz[iB, :, :2], colors)

        if overlay_text:
            cv2.putText(frA, f"Coach: {overlay_text.get('coach','')}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(frB, f"Student: {overlay_text.get('student','')}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        if impact_coach is not None and iA == impact_coach:
            cv2.putText(frA, "IMPACT", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        if impact_student is not None and iB == impact_student:
            cv2.putText(frB, "IMPACT", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # ใส่คะแนน (วาดบนทั้งสองฝั่ง)
        if overlay_scores:
            _draw_scores_badge(frA, overlay_scores)
            _draw_scores_badge(frB, overlay_scores)

        if layout == "horizontal":
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            canvas[:hA, :wA] = frA
            canvas[:hB, wA:wA+wB] = frB
        else:
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            canvas[:hA, :wA] = frA
            canvas[hA:hA+hB, :wB] = frB

        vw.write(canvas)

    vw.release()
