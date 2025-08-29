# src/pipeline.py
import os
import json
import yaml
import pandas as pd
from typing import Dict, Tuple, List
from pathlib import Path

from .pose_extractor import extract as extract_pose
from .features import compute_features_angle, _pivot_landmarks
from .dtw_utils import dtw
from .scoring import impact_from_aux, map_impact_to_student, score_segments
from .renderer import render_side_by_side

# ---------------- config helpers ----------------

def _load_config(cfg_path: str) -> Dict:
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"[pipeline] Config not found: {cfg_file.resolve()}")
    try:
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"[pipeline] Failed to parse YAML: {cfg_file}") from e
    if not isinstance(cfg, dict) or not cfg:
        raise RuntimeError(f"[pipeline] Config is empty or invalid: {cfg_file}")

    # required sections
    for key in ("pose", "segments", "scoring", "colors", "render"):
        if key not in cfg:
            raise KeyError(f"[pipeline] Missing '{key}' in config: {cfg_file}")

    # defaults
    cfg.setdefault("pose", {})
    cfg["pose"].setdefault("model_complexity", 1)
    cfg["pose"].setdefault("min_detection_confidence", 0.5)
    cfg["pose"].setdefault("min_tracking_confidence", 0.5)

    cfg.setdefault("preproc", {})
    cfg["preproc"].setdefault("center", "mid_hip")
    cfg["preproc"].setdefault("rotate", "shoulders_to_x")
    cfg["preproc"].setdefault("scale",  "shoulder_width")

    cfg.setdefault("smoothing", {})
    cfg["smoothing"].setdefault("method", "savgol")
    cfg["smoothing"].setdefault("window", 9)
    cfg["smoothing"].setdefault("polyorder", 2)
    cfg["smoothing"].setdefault("mov_window", 5)

    cfg.setdefault("dtw", {})
    cfg["dtw"].setdefault("window", None)        # Sakoe-Chiba band
    cfg["dtw"].setdefault("impact_window", 0)    # 0 = full sequence

    cfg.setdefault("scoring", {})
    cfg["scoring"].setdefault("weights", {"punch": 0.34, "core": 0.33, "feet": 0.33})
    cfg["scoring"].setdefault("tolerances", {"punch": 0.35, "core": 0.25, "feet": 0.25})
    cfg["scoring"].setdefault("mapping", "logistic")
    cfg["scoring"].setdefault("logistic", {"m": 0.40, "k": 6.0})
    cfg["scoring"].setdefault("linear", {"tol_at_zero": 0.35})

    cfg.setdefault("render", {})
    cfg["render"].setdefault("out_fps", 30)
    cfg["render"].setdefault("side_by_side", True)
    cfg["render"].setdefault("font_scale", 0.7)
    cfg["render"].setdefault("thickness", 2)

    return cfg

def _ensure_outputs_dir(outputs_dir: str) -> None:
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

def _extract_both_poses(
    coach_video: str,
    student_video: str,
    outputs_dir: str,
    model_complexity: int,
    min_det: float,
    min_track: float,
) -> Tuple[str, str, str, str, Dict, Dict]:
    coach_csv = os.path.join(outputs_dir, "coach_landmarks.csv")
    student_csv = os.path.join(outputs_dir, "student_landmarks.csv")
    coach_meta = os.path.join(outputs_dir, "coach_meta.json")
    student_meta = os.path.join(outputs_dir, "student_meta.json")

    extract_pose(coach_video, coach_csv, coach_meta, model_complexity, min_det, min_track)
    extract_pose(student_video, student_csv, student_meta, model_complexity, min_det, min_track)

    coach_meta_d = json.loads(Path(coach_meta).read_text(encoding="utf-8"))
    student_meta_d = json.loads(Path(student_meta).read_text(encoding="utf-8"))
    return coach_csv, student_csv, coach_meta, student_meta, coach_meta_d, student_meta_d

# ---------------- main pipeline ----------------

def run_pipeline(
    coach_video: str,
    student_video: str,
    cfg_path: str,
    outputs_dir: str = "outputs",
) -> Dict:
    # validate inputs
    if not Path(coach_video).exists():
        raise FileNotFoundError(f"[pipeline] Coach video not found: {Path(coach_video).resolve()}")
    if not Path(student_video).exists():
        raise FileNotFoundError(f"[pipeline] Student video not found: {Path(student_video).resolve()}")

    cfg = _load_config(cfg_path)
    _ensure_outputs_dir(outputs_dir)

    pose_cfg = cfg["pose"]
    model_complexity = int(pose_cfg.get("model_complexity", 1))
    min_det  = float(pose_cfg.get("min_detection_confidence", 0.5))
    min_trk  = float(pose_cfg.get("min_tracking_confidence", 0.5))
    dtw_win  = cfg.get("dtw", {}).get("window", None)
    if dtw_win is not None:
        try:
            dtw_win = int(dtw_win)
        except Exception:
            dtw_win = None
    impact_W = int(cfg.get("dtw", {}).get("impact_window", 0))

    # 1) Extract landmarks
    coach_csv, student_csv, coach_meta, student_meta, coach_meta_d, student_meta_d = _extract_both_poses(
        coach_video, student_video, outputs_dir, model_complexity, min_det, min_trk
    )

    # 2) Load CSVs
    coach_df = pd.read_csv(coach_csv)
    student_df = pd.read_csv(student_csv)

    # 3) Angle features + smoothing; robust scale by coach stats
    coach_feats, coach_stats = compute_features_angle(coach_df, coach_meta_d.get("fps", 30.0), cfg, ref_stats=None)
    student_feats, _ = compute_features_angle(student_df, student_meta_d.get("fps", 30.0), cfg, ref_stats=coach_stats)

    # 4) Initial impact (before DTW)
    coach_imp = impact_from_aux(coach_feats["aux"])
    # optional: student rough impact
    student_imp = impact_from_aux(student_feats["aux"]) if len(student_feats["aux"]["wrist_shoulder_dist"]) > 0 else 0

    # 5) Crop around impact then DTW
    def make_slice(T: int, center: int, W: int) -> Tuple[int, int]:
        if W <= 0:
            return 0, T
        return max(0, center - W), min(T, center + W + 1)

    Ta = coach_feats["global"].shape[0]
    Tb = student_feats["global"].shape[0]
    sa0, sa1 = make_slice(Ta, coach_imp, impact_W)
    sb0, sb1 = make_slice(Tb, student_imp, impact_W)

    cost, local_path = dtw(coach_feats["global"][sa0:sa1], student_feats["global"][sb0:sb1], window=dtw_win)
    path: List[Tuple[int,int]] = [(i + sa0, j + sb0) for (i, j) in local_path]

    # 6) Map coach impact -> student
    student_imp_map = map_impact_to_student(path, coach_imp)

    # 7) Scores with selected mapping (logistic/linear)
    scores = score_segments(
        coach_feats, student_feats, path,
        tolerances=cfg["scoring"]["tolerances"],
        weights=cfg["scoring"]["weights"],
        mapping_cfg=cfg["scoring"],
    )

    # 8) Render side-by-side (horizontal) with scores
    c_xyz, _ = _pivot_landmarks(coach_df)
    s_xyz, _ = _pivot_landmarks(student_df)
    out_video = os.path.join(outputs_dir, "side_by_side_analysis.mp4")
    render_error = None
    try:
        render_side_by_side(
            coach_meta_d["video_path"],
            student_meta_d["video_path"],
            c_xyz, s_xyz, path, cfg["colors"], out_video,
            out_fps=int(cfg["render"]["out_fps"]),
            overlay_text={"coach": f"Impact@{coach_imp}", "student": f"Impact@{student_imp_map}"},
            impact_coach=coach_imp, impact_student=student_imp_map,
            overlay_scores=scores,
            layout="horizontal"
        )
    except Exception as e:
        render_error = str(e)
        out_video = None

    # 9) Save results
    res = {
        "dtw_cost": cost,
        "path_len": len(path),
        "impact": {"coach": coach_imp, "student": student_imp_map},
        "scores": scores,
        "path": [[int(i), int(j)] for (i, j) in path],   # เพื่อ JSON/Viewer
        "outputs": {"video": out_video},
        "errors": {"render": render_error} if render_error else {}
    }
    Path(os.path.join(outputs_dir, "analysis.json")).write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return res

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MuayThai Jab DWT pipeline (angle features, tempo-invariant)")
    ap.add_argument("--coach", required=True, help="Path to coach video")
    ap.add_argument("--student", required=True, help="Path to student video")
    ap.add_argument("--cfg", default=str(Path(__file__).parent.parent / "configs" / "jab_left.yaml"),
                    help="Path to YAML config")
    ap.add_argument("--out", default="outputs", help="Outputs directory")
    args = ap.parse_args()

    results = run_pipeline(args.coach, args.student, args.cfg, args.out)
    print(json.dumps(results, indent=2, ensure_ascii=False))
