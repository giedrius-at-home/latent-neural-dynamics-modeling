# Directory Map

## Top-level layout (2026-04-13)

```
latent-neural-dynamics-modeling/
├── data/                        # Raw/processed ECoG + behavioral data (not in git)
├── results/                     # All model outputs — see results_layout.md
├── training/
│   └── setups/                  # YAML configs for PSID/DPAD/VARMA training
│       ├── psid/narrow_band_200Hz/{both,off,on}/   # Full PSID training configs
│       ├── dpad/narrow_band_200Hz/{both,off,on}/   # DPAD training configs
│       └── varma/narrow_band_200Hz/{both,off,on}/  # VARMA configs (top5 channels)
├── scripts/                     # Pipeline scripts
│   ├── pipeline_psid.py         # 6-phase: gs → classify → train → cross-eval → classify → thesis
│   ├── pipeline_dpad.py         # 5-phase: train → test → cross-eval → classify → thesis
│   └── pipeline_varma.py        # VARMA pipeline
├── classification/
│   ├── compute.py               # LDA classifier (called by pipelines)
│   └── setups/                  # Temp classification YAML configs (auto-generated)
├── notebooks/
│   ├── thesis_sec1_data_verification.ipynb   # Gold standard for figure patterns
│   ├── thesis_sec2_model_validation.ipynb    # 43 figures, model comparison
│   └── thesis_style.py          # Paper-style constants (NOT dashboard/thesis/constants.py)
├── dashboard/thesis/            # Streamlit dashboard modules
│   ├── specs.py                 # Canonical AlignedTriplet definitions
│   ├── loaders.py               # Data loading (parquet + legacy support)
│   ├── constants.py             # Dashboard-style constants (larger fonts, dark/light theme)
│   └── *.py                     # Various figure builders
├── thesis_figures/
│   ├── sec1/                    # Section 1 output PNGs
│   └── sec2/                    # Section 2 output PNGs (fig_007 through fig_043)
├── logs/chain/                  # Pipeline execution logs
├── pipeline_runs.md             # Manual record of all pipeline runs with results
└── utils/
    └── classification.py        # load_precomputed_results, result loading utilities
```

## Key paths

- **PSID grid search results:** `results/psid_gs_{PID}_S{SESS}_200Hz_narrow_band/`
- **PSID full training:** `results/psid_behavioral_{PID}_{SESS}_nx_{NX}_n2_i{I}_dbs_{COND}_200Hz_narrow_band/`
- **Classification (grid search):** `results/classification/gs_200Hz/{run_dir_name}/`
- **Classification (full):** `results/classification/{variant_name}/`
- **DPAD training:** `results/dpad_behavioral_{PID}_{SESS}_nx_{NX}_n2_e3000_top5_dbs_{COND}_200Hz_narrow_band/`
- **VARMA training:** `results/varma_behavioral_{PID}_{SESS}_nx_{NX}_n2_e3000_top5_dbs_{COND}_200Hz_narrow_band/`
