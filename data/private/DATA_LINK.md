# Private Dataset — Data Provenance

## Status

| Field | Value |
|-------|-------|
| **Source type** | `simulated` |
| **File in repo** | `face_headpose_private.csv` |
| **Generator** | `src/data/generate_datasets.py` |
| **Purpose** | Pipeline validation before real webcam collection |

## Variables

- **Endogenous:** `head_yaw` (degrees)
- **Exogenous:** `ambient_light`, `eye_aspect_ratio`, `mouth_open_ratio`

## Real data collection (recommended for final submission)

```bash
python scripts/extract_from_webcam.py --output data/private/face_headpose_real.csv --seconds 300
```

Update `experiments/config.py` CSV path or replace `face_headpose_private.csv` after collection.

## Raw video (Google Drive)

Upload original `.mp4` sessions to Google Drive and replace placeholder in README:

```
https://drive.google.com/drive/folders/YOUR_PRIVATE_FOLDER_ID
```

**Do not commit** raw video to GitHub (privacy + size).

## Metadata

See `metadata.json` for sampling rate, session count, and seed information.
