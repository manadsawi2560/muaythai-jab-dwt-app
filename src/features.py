# src/features.py
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from .pose_extractor import NAME2IDX  # ใช้ index ของ BlazePose 33 จุด

# -------------------- I/O helpers --------------------

def _pivot_landmarks(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert long-form CSV (frame,id,x,y,z,vis) -> arrays:
      xyz: [T, 33, 3], vis: [T, 33]
    """
    T = int(df["frame"].max() + 1) if len(df) else 0
    xyz = np.full((T, 33, 3), np.nan, dtype=np.float32)
    vis = np.zeros((T, 33), dtype=np.float32)
    for _, r in df.iterrows():
        t = int(r.frame); j = int(r.id)
        xyz[t, j, 0] = r.x; xyz[t, j, 1] = r.y; xyz[t, j, 2] = r.z
        vis[t, j] = r.vis
    return xyz, vis

# -------------------- geometry --------------------

def _joint_angle_xy(a, b, c):
    """
    Angle at joint b from a-b-c in 2D (x,y); returns radians [0,pi].
    Shapes: a,b,c = [T,2]
    """
    ab = a - b
    cb = c - b
    num = (ab * cb).sum(axis=-1)
    den = (np.linalg.norm(ab, axis=-1) * np.linalg.norm(cb, axis=-1) + 1e-8)
    cs = np.clip(num / den, -1, 1)
    return np.arccos(cs)

def _center_rotate_scale(xyz: np.ndarray, cfg: Dict) -> np.ndarray:
    """
    translate -> rotate -> scale ตาม config:
      center: mid_hip | nose | none
      rotate: shoulders_to_x | none
      scale : shoulder_width | shoulder_hip_diag | none
    ใช้ 2D (x,y) สำหรับ transform
    """
    out = xyz.copy()

    # -------- translate --------
    pre = (cfg.get("preproc") or {})
    center_mode = pre.get("center", "mid_hip")

    LHIP, RHIP = NAME2IDX["LEFT_HIP"], NAME2IDX["RIGHT_HIP"]
    LSH,  RSH  = NAME2IDX["LEFT_SHOULDER"], NAME2IDX["RIGHT_SHOULDER"]

    if center_mode == "mid_hip":
        mid = (out[:, LHIP, :2] + out[:, RHIP, :2]) / 2.0
        out[:, :, 0] -= mid[:, None, 0]
        out[:, :, 1] -= mid[:, None, 1]
    elif center_mode == "nose":
        nose = out[:, NAME2IDX["NOSE"], :2]
        out[:, :, 0] -= nose[:, None, 0]
        out[:, :, 1] -= nose[:, None, 1]
    # else: none

    # -------- rotate --------
    if pre.get("rotate", "shoulders_to_x") == "shoulders_to_x":
        sh_vec = out[:, RSH, :2] - out[:, LSH, :2]
        theta = np.arctan2(sh_vec[:, 1], sh_vec[:, 0])  # angle vs +x
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.stack(
            [np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1
        )  # [T,2,2]
        xy = out[:, :, :2]  # [T,33,2]
        out[:, :, :2] = np.einsum("tij,tkj->tki", R, xy)  # batched matmul

    # -------- scale --------
    scale_mode = pre.get("scale", "shoulder_width")
    if scale_mode == "shoulder_width":
        s = np.linalg.norm(out[:, RSH, :2] - out[:, LSH, :2], axis=1)
    elif scale_mode == "shoulder_hip_diag":
        sh_mid = (out[:, RSH, :2] + out[:, LSH, :2]) / 2
        hip_mid = (out[:, RHIP, :2] + out[:, LHIP, :2]) / 2
        s = np.linalg.norm(sh_mid - hip_mid, axis=1)
    else:
        s = np.ones(out.shape[0], dtype=np.float32)
    # robust fallback
    if np.any(s < 1e-6):
        fallback = np.nanmedian(s[s > 0]) if np.any(s > 0) else 1.0
        s[s < 1e-6] = fallback
    out[:, :, 0] /= s[:, None]
    out[:, :, 1] /= s[:, None]
    return out

# -------------------- smoothing & scaling --------------------

def _smooth_1d(x: np.ndarray, cfg: Dict) -> np.ndarray:
    sm = (cfg.get("smoothing") or {}).get("method", "savgol")
    if sm == "savgol":
        try:
            from scipy.signal import savgol_filter
            win = int((cfg["smoothing"].get("window") or 9))
            if win % 2 == 0:
                win += 1
            win = max(3, win)
            poly = int((cfg["smoothing"].get("polyorder") or 2))
            if len(x) >= win:
                return savgol_filter(x, win, poly, mode="interp")
        except Exception:
            pass  # fallback to movavg
        sm = "movavg"

    if sm == "movavg":
        w = int((cfg["smoothing"].get("mov_window") or 5))
        w = max(1, w)
        k = min(w, len(x))
        if k > 1:
            kernel = np.ones(k, dtype=np.float32) / k
            return np.convolve(x, kernel, mode="same")
    return x  # none / fallback

def _robust_scale_feat(X: np.ndarray, ref_stats: Optional[Dict] = None):
    """
    Robust z-score per-dimension: (x - median) / (1.4826 * MAD).
    ถ้า ref_stats ไม่ None ให้ใช้ median/MAD จากโค้ชเพื่อสเกลฝั่งผู้เรียน
    """
    if ref_stats is not None:
        med = ref_stats["median"]
        mad = ref_stats["mad"]
    else:
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med[None, :]), axis=0) + 1e-8
    Z = (X - med[None, :]) / (1.4826 * mad[None, :])
    return Z, {"median": med, "mad": mad}

# -------------------- main: angle features --------------------

def compute_features_angle(
    df: pd.DataFrame,
    fps: float,
    cfg: Dict,
    ref_stats: Optional[Dict] = None
):
    """
    1) translate -> rotate -> scale (preproc)
    2) angle features (L/R elbow/shoulder/hip/knee/ankle ที่จำเป็น)
    3) smooth สัญญาณทั้งหมด
    4) robust z-score (อิงโค้ชถ้ามี) -> return per-segment + global
    """
    xyz_raw, vis = _pivot_landmarks(df)
    T = xyz_raw.shape[0]
    if T == 0:
        # ป้องกัน empty input
        feats = {"punch": np.zeros((0,2)), "core": np.zeros((0,2)), "feet": np.zeros((0,3)), "global": np.zeros((0,7)), "aux": {"wrist_shoulder_dist": np.array([])}}
        return feats, {"median": np.array([]), "mad": np.array([])}

    xyz = _center_rotate_scale(xyz_raw, cfg)

    L = NAME2IDX
    # left side (สำคัญต่อ Jab ซ้าย)
    ang_L_elbow    = _joint_angle_xy(xyz[:, L["LEFT_SHOULDER"], :2], xyz[:, L["LEFT_ELBOW"], :2],  xyz[:, L["LEFT_WRIST"], :2])
    ang_L_shoulder = _joint_angle_xy(xyz[:, L["LEFT_HIP"], :2],      xyz[:, L["LEFT_SHOULDER"], :2], xyz[:, L["LEFT_ELBOW"], :2])
    ang_L_hip      = _joint_angle_xy(xyz[:, L["LEFT_SHOULDER"], :2], xyz[:, L["LEFT_HIP"], :2],     xyz[:, L["LEFT_KNEE"], :2])
    ang_L_knee     = _joint_angle_xy(xyz[:, L["LEFT_HIP"], :2],      xyz[:, L["LEFT_KNEE"], :2],    xyz[:, L["LEFT_ANKLE"], :2])
    ang_L_ankle    = _joint_angle_xy(xyz[:, L["LEFT_KNEE"], :2],     xyz[:, L["LEFT_ANKLE"], :2],   xyz[:, L["LEFT_FOOT_INDEX"], :2])

    # right side (core/stance balance)
    ang_R_hip      = _joint_angle_xy(xyz[:, L["RIGHT_SHOULDER"], :2], xyz[:, L["RIGHT_HIP"], :2],  xyz[:, L["RIGHT_KNEE"], :2])
    ang_R_knee     = _joint_angle_xy(xyz[:, L["RIGHT_HIP"], :2],      xyz[:, L["RIGHT_KNEE"], :2], xyz[:, L["RIGHT_ANKLE"], :2])

    # distance for impact detection
    dist_wrist_sh  = np.linalg.norm(xyz[:, L["LEFT_WRIST"], :2] - xyz[:, L["LEFT_SHOULDER"], :2], axis=1)

    # stack features
    feats_raw = np.stack([
        ang_L_elbow, ang_L_shoulder, ang_L_hip, ang_L_knee, ang_L_ankle,
        ang_R_hip,   ang_R_knee
    ], axis=1)  # [T, D=7]

    # smooth per-dimension
    for d in range(feats_raw.shape[1]):
        feats_raw[:, d] = _smooth_1d(feats_raw[:, d], cfg)
    dist_wrist_sh = _smooth_1d(dist_wrist_sh, cfg)

    # robust z-score using coach stats if provided
    feats_z, stats = _robust_scale_feat(feats_raw, ref_stats)

    # segment mapping
    punch = feats_z[:, [0, 1]]      # elbow, shoulder
    core  = feats_z[:, [2, 5]]      # L_hip, R_hip
    feet  = feats_z[:, [3, 4, 6]]   # L_knee, L_ankle, R_knee
    global_feat = np.concatenate([punch, core, feet], axis=1)

    aux = {"wrist_shoulder_dist": dist_wrist_sh}

    return {"punch": punch, "core": core, "feet": feet, "global": global_feat, "aux": aux}, stats
