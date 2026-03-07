#!/usr/bin/env bash
set -e

echo "=== VARMA narrow_band PDI4_S2 (stabilized) ==="
python -m training.train --config training/setups/varma/narrow_band/on/varma_PDI4_S2_dbs_on_narrow_band.yaml
python -m training.train --config training/setups/varma/narrow_band/off/varma_PDI4_S2_dbs_off_narrow_band.yaml
python -m training.train --config training/setups/varma/narrow_band/both/varma_PDI4_S2_dbs_both_narrow_band.yaml

echo "=== VARMA narrow_band PDI4_S3 (stabilized) ==="
python -m training.train --config training/setups/varma/narrow_band/on/varma_PDI4_S3_dbs_on_narrow_band.yaml
python -m training.train --config training/setups/varma/narrow_band/off/varma_PDI4_S3_dbs_off_narrow_band.yaml
python -m training.train --config training/setups/varma/narrow_band/both/varma_PDI4_S3_dbs_both_narrow_band.yaml

echo "Done."
