# Stage 6 — Thesis figures

## Job

The thesis notebooks (`notebooks/thesis_sec1 … thesis_sec8`) consume artifacts
from stages 1-5 and emit one PNG per panel under `thesis_figures/sec{N}/`.

## Notebooks → stages

| Notebook | Main stage(s) consumed | Output dir |
|---|---|---|
| `thesis_sec1_data_verification` | 1 (data integrity) | `thesis_figures/sec1/` |
| `thesis_sec2a_diagnostics` | 1 (elbow), 3 (improved vs vanilla) | `thesis_figures/sec2/` figs 37-38 |
| `thesis_sec2b_behavioral` | 4 (behavior reconstruction) | `thesis_figures/sec2/` figs 7-17 |
| `thesis_sec2c_neural_recon_group` | 4 (ECoG + LFP reconstruction) | `thesis_figures/sec2/` figs 18-23, 39-49 |
| `thesis_sec2d_neural_forecast_group` | 4 (neural forecast) | `thesis_figures/sec2/` figs 23-36, 56-59 |
| `thesis_sec2e_exemplars` | 4 (per-trial exemplar) | `thesis_figures/sec2/` figs 51-55 |
| `thesis_sec2_model_validation` | 4 (validation panels) | `thesis_figures/sec2/` |
| `thesis_sec5_classification` | 5 (LDA BA, confusion) | `thesis_figures/sec5/` |
| `thesis_sec7_subspace_dynamics` | 3-4 (latent subspace, state matrices) | `thesis_figures/sec7/` |
| `thesis_sec8_per_trial_heatmaps` | 4 (per-trial heatmaps) | `thesis_figures/sec8/` |

## Producer

Each notebook is paired (jupytext) with a `.py` file. To rerun:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1200 \
    notebooks/thesis_sec2d_neural_forecast_group.ipynb
```

Or batch (they're independent, run in parallel):

```bash
for nb in notebooks/thesis_sec*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 "$nb" &
done
wait
```

## Current inventory

Run this for the live count:

```bash
for s in sec1 sec2 sec5 sec7 sec8; do
    printf "  %-8s %s PNGs\n" "$s" "$(ls thesis_figures/$s/*.png 2>/dev/null | wc -l)"
done
```

At last inspection (2026-04-22): **sec2 has 133 PNGs across 51 unique figure numbers**.

## Figure styling rules (condensed from memory)

- Import from `notebooks/thesis_style.py`, never from `dashboard/thesis/constants.py`
- Use `panel_label()` helper + `apply_paper_style()` at the end
- Compact height, plain `-` panel labels, metric-only y-axis
- Max 2 colors, minimal text, identical per-session style
- Raw session names (`PDI1_S2`), not `PDI1 S2`
- Save with `scale=2` (retina)
- IQR-zoomed y-ranges for RMSE boxplots (Tukey window)

See `reports/pipeline_results/README.md` for the full list of refined
figure conventions.

## Known gaps (as of 2026-04-22)

- Pooled LFP prediction (laplacian equivalent of fig 39) — missing
- Pooled LFP forecast (laplacian equivalent of fig 24) — missing
- Behavior forecast (pooled + per-session) — completely absent; `forecast_target="Z"` on behavioral triplets not yet wired
- Fig number clash: `fig_023_neural_forecast_rmse.png` (sec2d) and `fig_023_lfp_band_*.png` (sec2c) both use fig 23

## Symlinks

- `thesis_figures/` → `../../../thesis_figures/`
- `notebooks/` → `../../../notebooks/`
