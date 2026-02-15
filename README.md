# Flood Insurance Analysis

This project builds a structured data processing pipeline for county-level flood insurance analysis.

## Structure

- `clean.py` – general data cleaning utilities
- `filter_policy.py` – policy-level filtering, used for the NFIP policy data
- `aggregate.py` – aggregation the policy data to county-year panel, used for the NFIP policy data
- `standardize.py` – log and z-score transformations
- `main.py` – pipeline entry point

## Data

Raw datasets are not included in this repository.

Required datasets:

- NFIP Policy Data (FEMA Open Data Portal)
- FEMA National Risk Index (NRI)
- CDC Social Vulnerability Index (SVI)
- NFIP Penetration Rates

After downloading, place them in:

```
data/raw_data/
```

## Reproducibility

Run:

```bash
python code/data_process/main.py
```

to generate processed datasets.
