#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== PSID log_power PDI4_S2 ==="
python -m training.train --config training/setups/psid/log_power/on/psid_behavioral_PDI4_2_nx_20_n16_i80_dbs_on_log_power.yaml
python -m training.train --config training/setups/psid/log_power/off/psid_behavioral_PDI4_2_nx_20_n16_i80_dbs_off_log_power.yaml
python -m training.train --config training/setups/psid/log_power/both/psid_behavioral_PDI4_2_nx_20_n16_i80_dbs_both_log_power.yaml

echo "=== PSID log_power PDI4_S3 ==="
python -m training.train --config training/setups/psid/log_power/on/psid_behavioral_PDI4_3_nx_15_n8_i80_dbs_on_log_power.yaml
python -m training.train --config training/setups/psid/log_power/off/psid_behavioral_PDI4_3_nx_15_n8_i80_dbs_off_log_power.yaml
python -m training.train --config training/setups/psid/log_power/both/psid_behavioral_PDI4_3_nx_15_n8_i80_dbs_both_log_power.yaml

echo "=== VARMA log_power PDI4_S2 (stabilized) ==="
python -m training.train --config training/setups/varma/log_power/on/varma_PDI4_S2_dbs_on_log_power.yaml
python -m training.train --config training/setups/varma/log_power/off/varma_PDI4_S2_dbs_off_log_power.yaml
python -m training.train --config training/setups/varma/log_power/both/varma_PDI4_S2_dbs_both_log_power.yaml

echo "=== VARMA log_power PDI4_S3 (stabilized) ==="
python -m training.train --config training/setups/varma/log_power/on/varma_PDI4_S3_dbs_on_log_power.yaml
python -m training.train --config training/setups/varma/log_power/off/varma_PDI4_S3_dbs_off_log_power.yaml
python -m training.train --config training/setups/varma/log_power/both/varma_PDI4_S3_dbs_both_log_power.yaml

echo "Done."
