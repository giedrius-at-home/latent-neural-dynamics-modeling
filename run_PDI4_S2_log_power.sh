#!/usr/bin/env bash
# Run PSID and VARMA (no grid search) for PDI4 session 2 log_power. VARMA configs use stabilize_roots.
set -e
cd "$(dirname "$0")"

echo "=== PSID log_power PDI4_S2 ==="
python -m training.train --config training/setups/psid/log_power/on/psid_behavioral_PDI4_2_nx_20_n16_i80_dbs_on_log_power.yaml
python -m training.train --config training/setups/psid/log_power/off/psid_behavioral_PDI4_2_nx_20_n16_i80_dbs_off_log_power.yaml
python -m training.train --config training/setups/psid/log_power/both/psid_behavioral_PDI4_2_nx_20_n16_i80_dbs_both_log_power.yaml

echo "=== VARMA log_power PDI4_S2 (stabilized) ==="
python -m training.train --config training/setups/varma/log_power/on/varma_PDI4_S2_dbs_on_log_power.yaml
python -m training.train --config training/setups/varma/log_power/off/varma_PDI4_S2_dbs_off_log_power.yaml
python -m training.train --config training/setups/varma/log_power/both/varma_PDI4_S2_dbs_both_log_power.yaml

echo "Done."
