import numpy as np
from typing import Dict, List, Tuple

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))




def map_score(rmse_val: float, tol: float) -> float:
    s = 100.0 * (1.0 - (rmse_val / max(tol, 1e-6)))
    return float(max(0.0, min(100.0, s)))




def impact_from_aux(aux: Dict[str, np.ndarray]) -> int:
    # ใช้เฟรมที่ระยะไหล่–ข้อมือมากที่สุด (เหยียดสุด) เป็น impact
    ext = aux['wrist_shoulder_dist']
    return int(np.nanargmax(ext))




def score_segments(coach_feats: Dict[str, np.ndarray], student_feats: Dict[str, np.ndarray],
            path: List[Tuple[int,int]], tolerances: Dict[str, float], weights: Dict[str, float]) -> Dict:
    out = {}
    for seg in ["punch","core","feet"]:
        A = coach_feats[seg]; B = student_feats[seg]
        A_al = np.array([A[i] for i,j in path])
        B_al = np.array([B[j] for i,j in path])
        r = rmse(A_al, B_al)
        s = map_score(r, tolerances.get(seg, 0.3))
        out[seg] = {"rmse": r, "score": s}
    final = sum(out[k]["score"] * weights.get(k, 1/3) for k in out)
    out["final_score"] = final
    return out




def map_impact_to_student(path: List[Tuple[int,int]], coach_impact: int) -> int:
    for i,j in path:
        if i == coach_impact:
            return int(j)
    idx = np.argmin([abs(i - coach_impact) for i,j in path])
    return int(path[idx][1])