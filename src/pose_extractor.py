import os, json, cv2, numpy as np, pandas as pd
from tqdm import tqdm
import mediapipe as mp


LANDMARK_NAMES = [
'NOSE','LEFT_EYE_INNER','LEFT_EYE','LEFT_EYE_OUTER','RIGHT_EYE_INNER','RIGHT_EYE','RIGHT_EYE_OUTER',
'LEFT_EAR','RIGHT_EAR','MOUTH_LEFT','MOUTH_RIGHT','LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW','RIGHT_ELBOW',
'LEFT_WRIST','RIGHT_WRIST','LEFT_PINKY','RIGHT_PINKY','LEFT_INDEX','RIGHT_INDEX','LEFT_THUMB','RIGHT_THUMB',
'LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE','LEFT_HEEL','RIGHT_HEEL','LEFT_FOOT_INDEX','RIGHT_FOOT_INDEX'
]


NAME2IDX = {n:i for i,n in enumerate(LANDMARK_NAMES)}




def extract(video_path: str, out_csv: str, out_meta_json: str,
        model_complexity: int = 1,
        min_det: float = 0.5, min_track: float = 0.5) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")


    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
    model_complexity=model_complexity,
    enable_segmentation=False,
    min_detection_confidence=min_det,
    min_tracking_confidence=min_track)


    rows = []
    f_idx = 0
    with pose:
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None,
            desc=f"Pose {os.path.basename(video_path)}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                for lid, lm in enumerate(res.pose_landmarks.landmark):
                    rows.append([f_idx, lid, lm.x, lm.y, lm.z, lm.visibility])
            f_idx += 1
            pbar.update(1)
        pbar.close()
    cap.release()


    df = pd.DataFrame(rows, columns=["frame","id","x","y","z","vis"])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


    meta = {"video_path": video_path, "fps": float(fps), "width": w, "height": h, "n_frames": int(f_idx)}
    with open(out_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)




if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_csv")
    ap.add_argument("out_meta")
    ap.add_argument("--model_complexity", type=int, default=1)
    ap.add_argument("--min_det", type=float, default=0.5)
    ap.add_argument("--min_track", type=float, default=0.5)
    args = ap.parse_args()
    extract(args.video, args.out_csv, args.out_meta,
        args.model_complexity, args.min_det, args.min_track)