#!/usr/bin/env bash
# Pull forecast results from all 8 Kaggle kernels.
# Downloads only the checkpoint tar.gz files and extracts to local results/.
# Usage: bash pull_forecast_results.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS="$REPO/results/dpad"
TMPBASE="/tmp/kaggle_forecast_pull_$$"

SESSIONS=(PDI1_S2 PDI1_S4 PDI4_S2 PDI4_S3)
MODES=(z-as-behavior z-as-neural)

slugify() {
  local session="${1,,}"   # lowercase
  session="${session//_/-}"
  local mode="$2"          # already hyphenated: z-as-behavior or z-as-neural
  echo "${session}-${mode}"
}

echo "=== Downloading kernel outputs ==="
pids=()
for session in "${SESSIONS[@]}"; do
  for mode in "${MODES[@]}"; do
    kid="giedriusmirklys/dpad-forecast-$(slugify "$session" "$mode")"
    tmpdir="$TMPBASE/${session}_${mode}"
    mkdir -p "$tmpdir"
    echo "  pulling $kid -> $tmpdir"
    (kaggle kernels output "$kid" -p "$tmpdir" --quiet 2>&1 | tail -3) &
    pids+=($!)
  done
done

echo "Waiting for ${#pids[@]} downloads..."
for pid in "${pids[@]}"; do wait "$pid" || true; done

echo ""
echo "=== Extracting forecast tars ==="
extracted=0
for session in "${SESSIONS[@]}"; do
  for mode in "${MODES[@]}"; do
    tmpdir="$TMPBASE/${session}_${mode}"
    for tar in "$tmpdir"/forecast_*.tar.gz; do
      [[ -f "$tar" ]] || continue
      echo "  extracting $(basename "$tar")"
      tar -xzf "$tar" -C "$REPO"
      ((extracted++))
    done
  done
done

echo ""
echo "=== Coverage after pull ==="
python3 - <<'EOF'
import sys
from pathlib import Path
from collections import defaultdict

base = Path(sys.argv[1])
coverage = defaultdict(set)

for p in base.rglob('test_results_*.parquet'):
    parts = p.parts
    if 'forecast' not in parts: continue
    fi = parts.index('forecast')
    run_dir = parts[fi-1]
    horizon = parts[fi+1]
    split   = parts[fi+2]
    if split == 'test' and 'dbs_both' in run_dir:
        for sess in ['PDI1_S2','PDI1_S4','PDI4_S2','PDI4_S3']:
            if sess in run_dir:
                variant = 'beh' if 'z-as-behavior' in run_dir else 'neu'
                coverage[f'{variant}_{sess}'].add(horizon)

for k in sorted(coverage):
    hs = sorted(coverage[k], key=lambda x: float(x[1:]))
    print(f'  {k}: {" ".join(hs)}')
EOF
"$RESULTS"

rm -rf "$TMPBASE"
echo ""
echo "Extracted $extracted tars. Temp dir cleaned."
