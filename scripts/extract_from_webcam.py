"""
Extract face biometric time series from webcam or video file.

Usage:
    python scripts/extract_from_webcam.py --output data/private/face_headpose_private.csv
    python scripts/extract_from_webcam.py --video path/to/video.mp4 --output data/private/session1.csv

Requires: opencv-python, mediapipe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit("Install mediapipe: pip install mediapipe opencv-python") from exc


def _eye_aspect_ratio(landmarks, idx_left, idx_right):
    def dist(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    left = [(landmarks[i].x, landmarks[i].y) for i in idx_left]
    right = [(landmarks[i].x, landmarks[i].y) for i in idx_right]
    ear_l = (dist(left[1], left[5]) + dist(left[2], left[4])) / (2 * dist(left[0], left[3]) + 1e-6)
    ear_r = (dist(right[1], right[5]) + dist(right[2], right[4])) / (2 * dist(right[0], right[3]) + 1e-6)
    return (ear_l + ear_r) / 2


def extract_from_capture(cap: cv2.VideoCapture, target_fps: float = 5.0, max_seconds: int = 120):
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
    )

    rows = []
    frame_interval = 1.0 / target_fps
    last_ts = -frame_interval
    frame_idx = 0

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [13, 14, 78, 308]

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_idx / max(cap.get(cv2.CAP_PROP_FPS), 1)
        if ts - last_ts < frame_interval:
            frame_idx += 1
            continue
        last_ts = ts

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ambient_light = float(gray.mean()) / 255.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_face.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            nose = lm[1]
            left_ear_pt = lm[234]
            right_ear_pt = lm[454]
            dx = right_ear_pt.x - left_ear_pt.x
            dy = right_ear_pt.y - left_ear_pt.y
            head_yaw = float(np.degrees(np.arctan2(dy, dx)))

            ear = _eye_aspect_ratio(lm, LEFT_EYE, RIGHT_EYE)
            mouth_open = abs(lm[MOUTH[0]].y - lm[MOUTH[1]].y)

            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=ts),
                    "session_id": 0,
                    "head_yaw": head_yaw,
                    "ambient_light": ambient_light,
                    "eye_aspect_ratio": ear,
                    "mouth_open_ratio": mouth_open,
                }
            )

        frame_idx += 1
        if ts >= max_seconds:
            break

    cap.release()
    mp_face.close()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract face time series for TimeXer")
    parser.add_argument("--output", default="data/private/face_headpose_private.csv")
    parser.add_argument("--video", default=None, help="Video file path (default: webcam 0)")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()

    source = args.video if args.video else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {source}")

    df = extract_from_capture(cap, target_fps=args.fps, max_seconds=args.seconds)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    meta = {
        "source": args.video or "webcam",
        "rows": len(df),
        "endo_variable": "head_yaw",
        "exo_variables": ["ambient_light", "eye_aspect_ratio", "mouth_open_ratio"],
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
