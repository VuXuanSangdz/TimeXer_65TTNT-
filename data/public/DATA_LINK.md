# Public Dataset — Data Provenance

## Status

| Field | Value |
|-------|-------|
| **Source type** | `simulated` |
| **File in repo** | `weather_public.csv` |
| **Generator** | `src/data/generate_datasets.py` |
| **Schema reference** | Weather benchmark in TimeXer / Time-Series-Library |

## Variables

- **Endogenous:** `co2_concentration`
- **Exogenous:** `temperature`, `humidity`, `pressure`, `wind_speed`

## Official benchmark (for paper-level comparison)

Download from [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library/tree/main/dataset) and preprocess to match exogenous forecasting format.

This repo uses a **synthetic subset** for reproducible course experiments, not the official Weather files.

## Metadata

See `metadata.json` for row count, sampling interval, and generation seed.
