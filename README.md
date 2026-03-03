# ADNI fMRI Benchmark Curation

This repository contains the curation pipeline for generating a benchmark split CSV from ADNI resting-state fMRI data. It matches imaging sessions to clinical data, balances diagnostic groups, and filters to TR ≈ 3s acquisitions.

## Output

Running the pipeline produces:

```
metadata/adni_fmri_benchmark_split.csv
```

Each row is one fMRI session matched to a clinical visit, with columns:

| Column | Description |
|---|---|
| `RID` | ADNI participant roster ID |
| `PTID` | Participant ID (e.g. `002_S_0295`) |
| `VISCODE_Clinical` | Clinical visit code (e.g. `bl`, `m12`) |
| `EXAMDATE_Clinical` | Date of matched clinical visit |
| `SCANDATE_Imaging` | Date of fMRI scan |
| `Days_Diff` | Days between scan and clinical visit |
| `Split` | `Train`, `Test`, or `Val` |
| `Current_DX` | Diagnosis at scan time: `CN`, `MCI`, or `Dementia` |
| `Label_Diag_AD_vs_CN` | Binary label for CN vs Dementia classification |
| `Label_Prog_pMCI_vs_sMCI` | MCI prognosis: `pMCI` (converter) or `sMCI` (stable) |
| `Score_CDRSB` | Clinical Dementia Rating Sum of Boxes |
| `Score_MMSE` | Mini-Mental State Examination score |
| `AGE` | Age at clinical visit |
| `PTGENDER` | Sex |

## Required Files

### 1. `adni_fmri_sessions.json` *(provided in this repository)*

Located at the project root. Contains the list of valid fMRI sessions with imaging metadata:

```json
{
  "subjects": {
    "002S0295": [
      {
        "session_id": "20110602",
        "TR": 3.001,
        "timesteps": 140
      }
    ]
  }
}
```

This file is **publicly available** in this repository. It was generated from preprocessed ADNI fMRI data and contains only acquisition metadata — no clinical or identifiable information.

### 2. `metadata/ADNIMERGE_14May2025.csv` *(requires ADNI access)*

The ADNIMERGE table is ADNI's merged longitudinal clinical dataset. It is not included in this repository and must be downloaded directly from the ADNI data portal.

Columns used: `RID`, `PTID`, `VISCODE`, `EXAMDATE`, `DX`, `CDRSB`, `MMSE`, `AGE`, `PTGENDER`, `COLPROT`.

## Requesting ADNI Data Access

ADNI data is available to qualified researchers. Apply for access at [https://adni.loni.usc.edu](https://adni.loni.usc.edu). Once approved, download the **ADNIMERGE** CSV from the [LONI IDA](https://ida.loni.usc.edu) and place it in the `metadata/` directory.

> **Note:** The filename in `adni_curation.py` is hardcoded to `ADNIMERGE_14May2025.csv`. If you download a newer version, update line 57 in `adni_curation.py` to match.

## Usage

### Install dependencies

```bash
uv sync
```

### Run the pipeline

```bash
uv run python adni_curation.py
```

The script will:

1. Load imaging sessions from `adni_fmri_sessions.json`
2. Match each session to the nearest clinical visit in ADNIMERGE (within 90 days)
3. Compute MCI prognosis labels (pMCI / sMCI) based on longitudinal follow-up
4. Balance diagnostic groups (CN / MCI / Dementia) by session count
5. Split into Train / Test / Val with stratification by age and sex
6. Filter to TR ≈ 3s sessions only (2.9–3.1s)
7. Save to `metadata/adni_fmri_benchmark_split.csv`
