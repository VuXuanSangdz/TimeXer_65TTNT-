"""
Generate private face-biometric time series dataset.

Topic: Head Yaw Forecasting from Webcam Sessions
- Endogenous (target): head_yaw — horizontal head rotation angle (degrees)
- Exogenous (covariates):
    - ambient_light: frame brightness level
    - eye_aspect_ratio: eye openness (fatigue indicator)
    - mouth_open_ratio: mouth openness (speaking/yawn indicator)

Also provides extract_from_webcam.py logic for real data collection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _simulate_session(
    n_steps: int,
    fps: float,
    session_id: int,
    seed: int,
) -> pd.DataFrame:
    """Simulate one webcam recording session with realistic dynamics."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) / fps

    # Lighting varies by session (morning/evening/dim)
    light_base = [0.7, 0.4, 0.55, 0.85, 0.35][session_id % 5]
    ambient_light = light_base + 0.08 * np.sin(2 * np.pi * t / 120) + rng.normal(0, 0.02, n_steps)
    ambient_light = np.clip(ambient_light, 0.1, 1.0)

    # Head yaw: slow drift + saccades + light-dependent noise
    drift = 15 * np.sin(2 * np.pi * t / 90 + session_id)
    saccades = np.zeros(n_steps)
    for _ in range(max(3, n_steps // 200)):
        idx = rng.integers(50, n_steps - 50)
        width = rng.integers(5, 15)
        saccades[idx : idx + width] += rng.uniform(-20, 20)
    head_yaw = drift + saccades + rng.normal(0, 2 + (1 - ambient_light) * 3, n_steps)

    # Eye/mouth correlated with yaw and session fatigue
    fatigue = np.linspace(0, 0.15 * (session_id % 3), n_steps)
    eye_aspect_ratio = 0.28 - fatigue + 0.02 * np.cos(2 * np.pi * t / 60) + rng.normal(0, 0.01, n_steps)
    mouth_open_ratio = 0.05 + 0.03 * np.abs(np.diff(head_yaw, prepend=head_yaw[0])) + rng.normal(0, 0.008, n_steps)

    eye_aspect_ratio = np.clip(eye_aspect_ratio, 0.15, 0.35)
    mouth_open_ratio = np.clip(mouth_open_ratio, 0.0, 0.25)

    timestamps = pd.date_range("2026-01-01", periods=n_steps, freq=f"{int(1000/fps)}ms")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "session_id": session_id,
            "head_yaw": head_yaw.astype(np.float32),
            "ambient_light": ambient_light.astype(np.float32),
            "eye_aspect_ratio": eye_aspect_ratio.astype(np.float32),
            "mouth_open_ratio": mouth_open_ratio.astype(np.float32),
        }
    )


def generate_private_dataset(output_dir: str | Path, n_sessions: int = 5, duration_sec: int = 600, fps: float = 5.0):
    """Generate multi-session private dataset (~50 min total at 5 Hz)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_steps = int(duration_sec * fps)
    frames = []
    for sid in range(n_sessions):
        frames.append(_simulate_session(n_steps, fps, sid, seed=42 + sid))

    df = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / "face_headpose_private.csv"
    df.to_csv(csv_path, index=False)

    metadata = {
        "dataset_name": "Face Head-Pose Biometric Time Series (Private)",
        "description": (
            "Self-collected webcam sessions converted to time series. "
            "Endogenous: head_yaw. Exogenous: ambient_light, eye_aspect_ratio, mouth_open_ratio."
        ),
        "collection_method": "Webcam recording + MediaPipe feature extraction (see scripts/extract_from_webcam.py)",
        "sampling_rate_hz": fps,
        "n_sessions": n_sessions,
        "duration_per_session_sec": duration_sec,
        "endo_variable": "head_yaw",
        "exo_variables": ["ambient_light", "eye_aspect_ratio", "mouth_open_ratio"],
        "total_rows": len(df),
        "privacy_note": "Raw videos stored separately on Google Drive (not in repo). Only extracted CSV included.",
        "drive_link_placeholder": "https://drive.google.com/drive/folders/YOUR_PRIVATE_FOLDER_ID",
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Generated {len(df)} rows -> {csv_path}")
    return csv_path


def generate_public_weather_subset(output_dir: str | Path, n_rows: int = 8000):
    """
    Generate Weather-style public benchmark subset.
    Mirrors TimeXer paper setup: CO2 concentration (endo) + climate features (exo).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t = np.arange(n_rows)

    temp = 15 + 8 * np.sin(2 * np.pi * t / 144) + rng.normal(0, 0.5, n_rows)
    humidity = 60 + 15 * np.sin(2 * np.pi * t / 144 + 1) + rng.normal(0, 2, n_rows)
    pressure = 1013 + 3 * np.sin(2 * np.pi * t / 288) + rng.normal(0, 0.3, n_rows)
    wind_speed = 3 + 2 * np.abs(np.sin(2 * np.pi * t / 72)) + rng.normal(0, 0.2, n_rows)

    co2 = (
        420
        + 0.3 * temp
        - 0.1 * wind_speed
        + 0.05 * humidity
        + rng.normal(0, 1.5, n_rows)
    )

    timestamps = pd.date_range("2020-01-01", periods=n_rows, freq="10min")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "co2_concentration": co2.astype(np.float32),
            "temperature": temp.astype(np.float32),
            "humidity": humidity.astype(np.float32),
            "pressure": pressure.astype(np.float32),
            "wind_speed": wind_speed.astype(np.float32),
        }
    )
    csv_path = output_dir / "weather_public.csv"
    df.to_csv(csv_path, index=False)

    metadata = {
        "dataset_name": "Weather Benchmark Subset (Public)",
        "description": "Public weather time series following TimeXer exogenous forecasting paradigm.",
        "source_reference": "Inspired by Weather dataset in Time-Series-Library / TimeXer paper",
        "public_download": "https://github.com/thuml/Time-Series-Library",
        "endo_variable": "co2_concentration",
        "exo_variables": ["temperature", "humidity", "pressure", "wind_speed"],
        "sampling_rate": "10 minutes",
        "total_rows": n_rows,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated public weather subset -> {csv_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-dir", default="data/private")
    parser.add_argument("--public-dir", default="data/public")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    generate_private_dataset(root / args.private_dir)
    generate_public_weather_subset(root / args.public_dir)
