# src/pipeline.py
import os
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path

from .pose_extractor import extract as extract_pose
from .features import compute_features, _pivot_landmarks
from .dtw_utils import dtw
from .scoring import impact_from_aux, map_impact_to_student, score_segments
from .renderer import render_side_by_side


def _load_config(cfg_path: str) -> Dict:
    """Load YAML config with basic validation."""
    cfg_file = Path(cfg_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"[pipeline] Config not found: {cfg_file.resolve()}")
    try:
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"[pipeline] Failed to parse YAML: {cfg_file}") from e
    if not isinstance(cfg, dict) or not cfg:
        raise RuntimeError(f"[pipeline] Config is empty or invalid: {cfg_file}")

    # Minimal required sections (pose, segments, scoring, colors, render)
    for key in ("pose", "segments", "scoring", "colors", "render"):
        if key not in cfg:
            raise KeyError(f"[pipeline] Missing '{key}' section in config: {cfg_file}")

    # Fill defaults for pose/render/scoring to be robust
    cfg.setdefault("pose", {})
    cfg["pose"].setdefault("model_complexity", 1)
    cfg["pose"].setdefault("min_detection_confidence", 0.5)
    cfg["pose"].setdefault("min_tracking_confidence", 0.5)

    cfg.setdefault("render", {})
    cfg["render"].setdefault("out_fps", 30)
    cfg["render"].setdefault("side_by_side", True)
    cfg["render"].setdefault("font_scale", 0.7)
    cfg["render"].setdefault("thickness", 2)

    cfg.setdefault("scoring", {})
    cfg["scoring"].setdefault("weights", {"punch": 0.34, "core": 0.33, "feet": 0.33})
    cfg["scoring"].setdefault("tolerances", {"punch": 0.35, "core": 0.25, "feet": 0.25})

    # Optional DTW window (None = full)
    if "dtw" not in cfg:
        cfg["dtw"] = {}
    cfg["dtw"].setdefault("window", None)  # e.g., 15 for Sakoe-Chiba band

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
    """Run BlazePose extraction for both videos and return file paths + metadata dicts."""
    coach_csv = os.path.join(outputs_dir, "coach_landmarks.csv")
    student_csv = os.path.join(outputs_dir, "student_landmarks.csv")
    coach_meta = os.path.join(outputs_dir, "coach_meta.json")
    student_meta = os.path.join(outputs_dir, "student_meta.json")

    # Extract (will overwrite if already exists)
    extract_pose(coach_video, coach_csv, coach_meta, model_complexity, min_det, min_track)
    extract_pose(student_video, student_csv, student_meta, model_complexity, min_det, min_track)

    # Load metadata JSONs (must exist now)
    try:
        coach_meta_d = json.loads(Path(coach_meta).read_text(encoding="utf-8"))
        student_meta_d = json.loads(Path(student_meta).read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError("[pipeline] Failed to read meta JSON after extraction") from e

    return coach_csv, student_csv, coach_meta, student_meta, coach_meta_d, student_meta_d


def run_pipeline(
    coach_video: str,
    student_video: str,
    cfg_path: str,
    outputs_dir: str = "outputs",
) -> Dict:
    """
    Main pipeline:
      1) Load config
      2) Extract BlazePose landmarks for both videos
      3) Build tempo-invariant features (no speed), using coach stats to scale both sides
      4) DTW alignment on global features
      5) Impact detection (max shoulder-wrist distance) & mapping
      6) Segment scores + weighted final
      7) Render side-by-side video along DTW path
      8) Save analysis.json + return results dict
    """
    # Validate inputs
    if not Path(coach_video).exists():
        raise FileNotFoundError(f"[pipeline] Coach video not found: {Path(coach_video).resolve()}")
    if not Path(student_video).exists():
        raise FileNotFoundError(f"[pipeline] Student video not found: {Path(student_video).resolve()}")

    cfg = _load_config(cfg_path)
    _ensure_outputs_dir(outputs_dir)

    pose_cfg = cfg["pose"]
    model_complexity = int(pose_cfg.get("model_complexity", 1))
    min_det = float(pose_cfg.get("min_detection_confidence", 0.5))
    min_track = float(pose_cfg.get("min_tracking_confidence", 0.5))
    dtw_window = cfg.get("dtw", {}).get("window", None)

    # 1) Pose extraction for both sides
    coach_csv, student_csv, coach_meta, student_meta, coach_meta_d, student_meta_d = _extract_both_poses(
        coach_video, student_video, outputs_dir, model_complexity, min_det, min_track
    )

    # 2) Load landmark CSVs
    coach_df = pd.read_csv(coach_csv)
    student_df = pd.read_csv(student_csv)

    # 3) Build tempo-invariant features (no velocity). Use coach stats to scale both sides.
    coach_feats, coach_stats = compute_features(coach_df, coach_meta_d.get("fps", 30.0), ref_stats=None)
    student_feats, _ = compute_features(student_df, student_meta_d.get("fps", 30.0), ref_stats=coach_stats)

    # 4) DTW alignment on global features
    cost, path = dtw(coach_feats["global"], student_feats["global"], window=dtw_window)

    # 5) Impact on coach side (max shoulder-wrist distance), map to student via DTW path
    coach_imp = impact_from_aux(coach_feats["aux"])
    student_imp = map_impact_to_student(path, coach_imp)

    # 6) Scores per segment + weighted final
    scores = score_segments(
        coach_feats,
        student_feats,
        path,
        tolerances=cfg["scoring"]["tolerances"],
        weights=cfg["scoring"]["weights"],
    )

    # 7) Render side-by-side video along DTW path
    c_xyz, _ = _pivot_landmarks(coach_df)
    s_xyz, _ = _pivot_landmarks(student_df)
    out_video = os.path.join(outputs_dir, "side_by_side_analysis.mp4")

    try:
        render_side_by_side(
            coach_meta_d["video_path"],
            student_meta_d["video_path"],
            c_xyz,
            s_xyz,
            path,
            cfg["colors"],
            out_video,
            out_fps=int(cfg["render"]["out_fps"]),
            overlay_text={"coach": f"Impact@{coach_imp}", "student": f"Impact@{student_imp}"},
            impact_coach=coach_imp,
            impact_student=student_imp,
        )
    except Exception as e:
        # ไม่ให้ทั้งงานล่มถ้า render ผิดพลาด — แต่แจ้งเตือนในผลลัพธ์
        out_video = None
        render_error = str(e)
    else:
        render_error = None

    # 8) Save results
    res = {
        "dtw_cost": cost,
        "path_len": len(path),
        "impact": {"coach": coach_imp, "student": student_imp},
        "scores": scores,
        "outputs": {"video": out_video},
        "errors": {"render": render_error} if render_error else {},
    }

    Path(os.path.join(outputs_dir, "analysis.json")).write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MuayThai Jab DWT pipeline (tempo-invariant)")
    ap.add_argument("--coach", required=True, help="Path to coach video")
    ap.add_argument("--student", required=True, help="Path to student video")
    ap.add_argument("--cfg", default=str(Path(__file__).parent.parent / "configs" / "jab_left.yaml"),
                    help="Path to YAML config")
    ap.add_argument("--out", default="outputs", help="Outputs directory")
    args = ap.parse_args()

    results = run_pipeline(args.coach, args.student, args.cfg, args.out)
    print(json.dumps(results, indent=2, ensure_ascii=False))
