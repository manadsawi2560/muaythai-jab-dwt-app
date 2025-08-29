# src/scoring.py
import numpy as np
from typing import Dict, List, Tuple

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

# -------- score mappings --------

def map_score_linear(rmse_val: float, tol_at_zero: float) -> float:
    """
    Linear map -> 100 when rmse=0, 0 when rmse=tol_at_zero (or more)
    """
    s = 100.0 * (1.0 - rmse_val / max(tol_at_zero, 1e-6))
    return float(np.clip(s, 0.0, 100.0))

def map_score_logistic(rmse_val: float, m: float, k: float) -> float:
    """
    Logistic map -> 0..100 (สูง = ดี)
      m: จุดกึ่งกลางของ RMSE (score ~ 50)
      k: ความชัน (ใหญ่ = ชัน)
    """
    return float(100.0 / (1.0 + np.exp(k * (rmse_val - m))))

# -------- impact & alignment helpers --------

def impact_from_aux(aux: Dict[str, np.ndarray]) -> int:
    """
    เลือกเฟรมที่ระยะไหล่–ข้อมือซ้าย (coach) สูงสุดเป็น impact
    """
    ext = aux["wrist_shoulder_dist"]
    return int(np.nanargmax(ext))

def map_impact_to_student(path: List[Tuple[int,int]], coach_impact: int) -> int:
    """
    หา index ฝั่งผู้เรียนที่ align กับ coach_impact บน DTW path
    """
    for i, j in path:
        if i == coach_impact:
            return int(j)
    # ถ้าไม่มี exact match ให้หา j ที่ใกล้เคียงที่สุด
    if not path:
        return 0
    idx = int(np.argmin([abs(i - coach_impact) for i, j in path]))
    return int(path[idx][1])

# -------- main scoring --------

def score_segments(coach_feats: Dict[str, np.ndarray],
                   student_feats: Dict[str, np.ndarray],
                   path: List[Tuple[int,int]],
                   tolerances: Dict[str, float],
                   weights: Dict[str, float],
                   mapping_cfg: Dict) -> Dict:
    """
    คำนวณ RMSE รายส่วนบนเฟรมที่ถูกจัดแนวด้วย DTW แล้ว map เป็นคะแนน
    รองรับ mapping แบบ 'logistic' หรือ 'linear'
    """
    out = {}

    use_log = (mapping_cfg.get("mapping", "logistic") == "logistic")
    log_m   = float(mapping_cfg.get("logistic", {}).get("m", 0.40))
    log_k   = float(mapping_cfg.get("logistic", {}).get("k", 6.0))
    lin_tol = float(mapping_cfg.get("linear",   {}).get("tol_at_zero", 0.35))

    for seg in ["punch", "core", "feet"]:
        A = coach_feats[seg]; B = student_feats[seg]
        A_al = np.array([A[i] for i, j in path])
        B_al = np.array([B[j] for i, j in path])
        r = rmse(A_al, B_al)
        if use_log:
            s = map_score_logistic(r, log_m, log_k)
        else:
            s = map_score_linear(r, tolerances.get(seg, lin_tol))
        out[seg] = {"rmse": r, "score": s}

    final = sum(out[k]["score"] * weights.get(k, 1/3) for k in out)
    out["final_score"] = final
    return out
