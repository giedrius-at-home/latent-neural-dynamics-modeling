# PSID latent-dimension elbow selection

Method for picking `n1` (behaviour-relevant stage) and `nx` (total state) for
every (session × feature-family) cell before running the production PSID
pipeline. Picks live in `configs/diagnostic/elbow_choices.yaml`.

## Inputs

A vanilla PSID diagnostic fit (no A-matrix eigenvalue clipping, no RTS
smoother) produced by:

```
scripts/run_all_psid_diagnostics.sh         # drives pipeline_psid_diagnostic.py
```

For every session × family this writes:

```
results/diagnostic/{P}_{S}_psid_spectra/psid/{ecog,laplacian}/
    spectra.parquet      columns: index, ZHat_S, YHat_S
    matrices.npz         A, Cy, Cz, Q, R, S
```

* `ZHat_S` — stage-1 singular values of the behaviour-relevant
  cross-covariance. Rank tells you how many latent dimensions actually carry
  kinematic-target information.
* `YHat_S` — stage-2 singular values of the full past-future neural
  covariance. Rank tells you how many additional dimensions describe neural
  dynamics independent of behaviour.

Runs are restricted to the *train* blocks (`--train-only --train-frac 0.5`)
so elbows never see test data.

## Rules

Let `z = ZHat_S`, `y = YHat_S`, and let `thr = thresholds.n1_relative_to_peak`,
`E = thresholds.nx_cumulative_energy` from the YAML.

**n1** — smallest index `k` such that `z[k] / z[0] < thr`. Defaults to the
length of `z` when no singular value drops below the threshold.

```python
n1 = first k where  z[k] / max(z) < thr
```

**nx** — smallest index `k` such that the cumulative energy up to `k`
captures fraction `E` of the total stage-2 spectrum.

```python
nx = first k where  sum(y[:k]) / sum(y) >= E
```

`n2 = nx - n1` is what gets reported alongside (neural-only dim budget).

Both rules are monotone in their thresholds — lowering `thr` or raising `E`
increases `n1` / `nx`, and vice-versa.

## Operating points used so far

| Setting | `thr` (n1) | `E` (nx) | Comment |
|---|---|---|---|
| conservative | 0.15 | 0.85 | trims tails aggressively |
| default     | 0.10 | 0.90 | middle of the road |
| permissive  | 0.05 | 0.95 | keeps almost every ranked dim |

Pick per-session picks with the default operating point unless a spectrum
summary PNG shows an obviously different knee. Iterate on individual cells by
editing `configs/diagnostic/elbow_choices.yaml` directly.

## Regenerating picks

```bash
# Refit diagnostics from scratch (needed when balanced splits change or new
# sessions are added).
bash scripts/run_all_psid_diagnostics.sh

# Apply current thresholds to the latest spectra and rewrite the YAML.
python - <<'PY'
from pathlib import Path
import numpy as np, polars as pl, yaml

ROOT = Path("results/diagnostic")
SESSIONS = [("PDI1","2"), ("PDI1","4"), ("PDI4","2"), ("PDI4","3")]
FAMILIES = ["ecog", "laplacian"]
THR = 0.10      # edit to change operating point
E   = 0.90

cfg_path = Path("configs/diagnostic/elbow_choices.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

for p, s in SESSIONS:
    tag = f"{p}_{s}"; label = f"{p}_S{s}"
    for fam in FAMILIES:
        df = pl.read_parquet(ROOT / f"{tag}_psid_spectra" / "psid" / fam / "spectra.parquet")
        z, y = df["ZHat_S"].to_numpy(), df["YHat_S"].to_numpy()
        n1 = int(np.argmax(z / z[0] < THR)) if (z / z[0] < THR).any() else len(z)
        nx = int(np.searchsorted(np.cumsum(y) / y.sum(), E) + 1)
        prev_tk = cfg["sessions"].get(label, {}).get(fam, {}).get("target_K", 8)
        cfg.setdefault("sessions", {}).setdefault(label, {})[fam] = {
            "n1": n1, "nx": nx, "target_K": prev_tk
        }

cfg.setdefault("thresholds", {})
cfg["thresholds"]["n1_relative_to_peak"] = THR
cfg["thresholds"]["nx_cumulative_energy"] = E
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

## Downstream consumption

| Consumer | Field |
|---|---|
| `notebooks/thesis_sec2a_diagnostics.py` — Fig 36 (PSID scree) | `sessions[label][family].{n1,nx}` |
| `scripts/pipeline_psid.py` via `--best-nx`/`--best-n1` | per-cell picks passed in by the batch runner |
| `scripts/run_all_sessions.py` (batch runner) | via `_build_configs_from_elbows()` |
| `scripts/generate_reduced_training_configs.py` | YAML emission for mRMR-reduced variants |

All four read the same YAML, so editing `elbow_choices.yaml` (or the two
`thresholds:` values) is the single lever.
