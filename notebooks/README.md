# notebooks

The thesis figures. Each notebook reads the `results/` tree written by the
training pipeline and writes PNGs to `thesis_figures/`, grouped by thesis section.

Run them on the compute host — their first cell does
`os.chdir("/home/bobby/repos/latent-neural-dynamics-modeling")` and they read
`results/`, which only exists there:

```bash
ssh bobby@neuro
cd ~/repos/latent-neural-dynamics-modeling
~/miniconda3/envs/neuro/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/thesis_sec5_classification.ipynb
```

`_runner.py` executes an exported notebook as a plain script with `fig.show()`
suppressed, for headless runs.

No timestamps need to be passed in. `modules/loaders.py` and
`utils/thesis_result_timestamps.py` discover the newest run per variant on disk,
so a notebook always plots the latest results.

| Notebook | Thesis section |
|---|---|
| `thesis_sec1_data_verification.ipynb` | data checks, PSDs, channel/order selection, DPAD training curves |
| `thesis_sec2c_neural_recon_group.ipynb` | neural reconstruction, group level |
| `thesis_sec2d_neural_forecast_group.ipynb` | neural forecast, group level |
| `thesis_sec2e_exemplars.ipynb` | exemplar trials |
| `thesis_sec5_classification.ipynb` | DBS decoding |
| `thesis_sec5b_group_permutation.ipynb` | group-level permutation test |
| `thesis_sec7_subspace_dynamics.ipynb` | latent subspace geometry |
| `thesis_sec7b_matrix_dynamics.ipynb` | state-transition matrix analysis |
| `thesis_sec_appendix.ipynb` | appendix |

`data_analysis/` holds exploratory notebooks that produce no thesis figures.

Shared code lives in `modules/`: `loaders.py` (run discovery and result loading),
`style.py` (fonts, colours, panel labels — every figure calls
`apply_thesis_style()`), and `sec*_common.py` (per-section helpers). Put anything
reused across notebooks there rather than copying cells between them.
