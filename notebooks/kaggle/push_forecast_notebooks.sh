#!/usr/bin/env bash
# Push all 8 per-(session,mode) forecast notebooks to Kaggle sequentially.
# Usage: bash push_forecast_notebooks.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TMPBASE="/tmp/kaggle_forecast_push"
rm -rf "$TMPBASE"

SESSIONS=(PDI1_S2 PDI1_S4 PDI4_S2 PDI4_S3)
MODES=(z-as-behavior z-as-neural)

ok=0; fail=0

for session in "${SESSIONS[@]}"; do
  for mode in "${MODES[@]}"; do
    nb="$HERE/dpad_forecast_${session}_${mode}.ipynb"
    meta="$HERE/kernel-metadata-forecast-${session}-${mode}.json"

    if [[ ! -f "$nb" || ! -f "$meta" ]]; then
      echo "SKIP missing: $nb or $meta"
      continue
    fi

    tmpdir="$TMPBASE/${session}_${mode}"
    mkdir -p "$tmpdir"
    cp "$nb"   "$tmpdir/$(basename "$nb")"
    cp "$meta" "$tmpdir/kernel-metadata.json"

    echo "Pushing ${session} ${mode}..."
    if kaggle kernels push -p "$tmpdir" 2>&1; then
      echo "  OK"
      ((ok++))
    else
      echo "  FAILED"
      ((fail++))
    fi
    sleep 3
  done
done

rm -rf "$TMPBASE"
echo ""
echo "Done. $ok OK, $fail failed."
