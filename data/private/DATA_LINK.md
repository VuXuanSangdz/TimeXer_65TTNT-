# Private Dataset

## Trong repository (CSV đã trích xuất)

- `face_headpose_private.csv` — 15,000 dòng, 5 session, 5 Hz
- `metadata.json` — mô tả biến và cách thu thập

## Raw video (Google Drive — private)

Upload video gốc lên Google Drive và cập nhật link trong README:

```
https://drive.google.com/drive/folders/YOUR_PRIVATE_FOLDER_ID
```

**Không commit** file `.mp4` vào GitHub (privacy + dung lượng).

## Thu thập dữ liệu thật

```bash
python scripts/extract_from_webcam.py --output data/private/face_headpose_real.csv --seconds 300
```
