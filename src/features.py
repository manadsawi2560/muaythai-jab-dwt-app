import numpy as np, pandas as pd
from typing import Dict, Tuple, Optional
from .pose_extractor import LANDMARK_NAMES, NAME2IDX

# === Utilities ===


def _pivot_landmarks(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    T = int(df["frame"].max() + 1)
    xyz = np.full((T, 33, 3), np.nan, dtype=np.float32)
    vis = np.zeros((T, 33), dtype=np.float32)
    for _, r in df.iterrows():
        t = int(r.frame); j = int(r.id)
        xyz[t, j, 0] = r.x; xyz[t, j, 1] = r.y; xyz[t, j, 2] = r.z
        vis[t, j] = r.vis
    return xyz, vis

def _normalize(xyz: np.ndarray) -> np.ndarray:
    LHIP, RHIP = NAME2IDX['LEFT_HIP'], NAME2IDX['RIGHT_HIP']
    mid_hip = (xyz[:, LHIP, :2] + xyz[:, RHIP, :2]) / 2.0
    hip_w = np.linalg.norm(xyz[:, LHIP, :2] - xyz[:, RHIP, :2], axis=1)
    hip_w[hip_w < 1e-6] = np.nanmedian(hip_w[hip_w > 0]) or 1.0
    out = xyz.copy()
    out[:, :, 0] = (xyz[:, :, 0] - mid_hip[:, None, 0]) / hip_w[:, None]
    out[:, :, 1] = (xyz[:, :, 1] - mid_hip[:, None, 1]) / hip_w[:, None]
    out[:, :, 2] = xyz[:, :, 2]
    return out

def _angle(a, b, c):
    ab = a - b; cb = c - b
    num = (ab * cb).sum(axis=-1)
    den = (np.linalg.norm(ab, axis=-1) * np.linalg.norm(cb, axis=-1) + 1e-8)
    cosv = np.clip(num / den, -1, 1)
    return np.arccos(cosv)

def _robust_scale_feat(X: np.ndarray, ref_stats: Optional[Dict] = None):
    if ref_stats is not None:
        med = ref_stats['median']; mad = ref_stats['mad']
    else:
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med[None, :]), axis=0) + 1e-8
    Z = (X - med[None, :]) / (1.4826 * mad[None, :])
    return Z, {'median': med, 'mad': mad}

def compute_features(df: pd.DataFrame, fps: float, ref_stats: Optional[Dict[str, Dict]] = None):
    xyz, vis = _pivot_landmarks(df)
    T = xyz.shape[0]
    norm = _normalize(xyz)


    LW, LE, LS = NAME2IDX['LEFT_WRIST'], NAME2IDX['LEFT_ELBOW'], NAME2IDX['LEFT_SHOULDER']
    RS, LH, RH = NAME2IDX['RIGHT_SHOULDER'], NAME2IDX['LEFT_HIP'], NAME2IDX['RIGHT_HIP']
    LK, RK, LA, RA = NAME2IDX['LEFT_KNEE'], NAME2IDX['RIGHT_KNEE'], NAME2IDX['LEFT_ANKLE'], NAME2IDX['RIGHT_ANKLE']


    wrist_xy = norm[:, LW, :2]
    elbow_xy = norm[:, LE, :2]
    shoulder_xy = norm[:, LS, :2]


    elbow_ang = _angle(shoulder_xy, elbow_xy, wrist_xy) # radians (ใหญ่ ~ เหยียดสุด)
    sw_dist = np.linalg.norm(wrist_xy - shoulder_xy, axis=-1) # ระยะเหยียดแขน


    LSh_xy, RSh_xy = norm[:, LS, :2], norm[:, RS, :2]
    LH_xy, RH_xy = norm[:, LH, :2], norm[:, RH, :2]
    torso_vec = (LSh_xy + RSh_xy)/2 - (LH_xy + RH_xy)/2
    torso_len = np.linalg.norm(torso_vec, axis=-1)
    shoulder_vec = RSh_xy - LSh_xy
    hip_vec = RH_xy - LH_xy


    def _angle2(v1, v2):
        num = (v1 * v2).sum(axis=-1)
        den = (np.linalg.norm(v1,axis=-1)*np.linalg.norm(v2,axis=-1) + 1e-8)
        cs = np.clip(num/den, -1, 1)
        return np.arccos(cs)


    torsion = _angle2(shoulder_vec, hip_vec)
    lean = _angle2(torso_vec, np.stack([np.zeros(T), -np.ones(T)], axis=1))


    LA_xy, RA_xy = norm[:, LA, :2], norm[:, RA, :2]
    LK_xy, RK_xy = norm[:, LK, :2], norm[:, RK, :2]
    stance = np.linalg.norm(RA_xy - LA_xy, axis=-1)
    knee_flex = _angle(LH_xy, LK_xy, LA_xy) + _angle(RH_xy, RK_xy, RA_xy)


    punch_raw = np.stack([elbow_ang, sw_dist], axis=1) # 2D (no speed)
    core_raw = np.stack([torsion, lean, torso_len], axis=1)
    feet_raw = np.stack([stance, knee_flex], axis=1)


    if ref_stats is None:
        punch, p_stats = _robust_scale_feat(punch_raw, None)
        core, c_stats = _robust_scale_feat(core_raw, None)
        feet, f_stats = _robust_scale_feat(feet_raw, None)
    else:
        punch, p_stats = _robust_scale_feat(punch_raw, ref_stats['punch'])
        core, c_stats = _robust_scale_feat(core_raw, ref_stats['core'])
        feet, f_stats = _robust_scale_feat(feet_raw, ref_stats['feet'])


    global_feat = np.concatenate([punch, core, feet], axis=1)


    stats = { 'punch': p_stats, 'core': c_stats, 'feet': f_stats }


    aux = {
    'elbow_angle': elbow_ang,
    'wrist_shoulder_dist': sw_dist
    }


    return {"punch": punch, "core": core, "feet": feet, "global": global_feat, "aux": aux}, stats