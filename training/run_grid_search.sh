#!/bin/bash
# Grid search over PSID parameters nx and n1 for all psid_SEARCH configs
# Usage: ./run_grid_search.sh [config_pattern]
# Example: ./run_grid_search.sh psid_SEARCH  (runs all psid_SEARCH_* configs)
# Example: ./run_grid_search.sh psid_SEARCH_alpha  (runs only alpha)

set -e

CONFIG_PATTERN="${1:-psid_SEARCH}"
CONFIG_DIR="training/setups"

# Find all matching configs
CONFIGS=($(ls ${CONFIG_DIR}/${CONFIG_PATTERN}*.yaml 2>/dev/null))

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Error: No config files matching '${CONFIG_PATTERN}*.yaml' found in ${CONFIG_DIR}"
    exit 1
fi

# Grid parameters - intervals [5, 25] for nx and [3, 15] for n1 with step=1
NX_VALUES=(2 5 10 15 20)
N1_VALUES=(2 5 10 15 20)

echo "=== PSID Grid Search ==="
echo "Config pattern: ${CONFIG_PATTERN}"
echo "Found ${#CONFIGS[@]} config(s): ${CONFIGS[*]}"
echo "nx values: ${NX_VALUES[*]}"
echo "n1 values: ${N1_VALUES[*]}"
echo ""

for CONFIG_PATH in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG_PATH" .yaml)
    echo ""
    echo "=========================================="
    echo "Processing config: $CONFIG_NAME"
    echo "=========================================="
    
    # Backup original config
    cp "$CONFIG_PATH" "${CONFIG_PATH}.backup"

    for nx in "${NX_VALUES[@]}"; do
        for n1 in "${N1_VALUES[@]}"; do
            # Ensure n1 <= nx (PSID constraint)
            if [ "$n1" -gt "$nx" ]; then
                continue
            fi
            
            echo "[$CONFIG_NAME] Running with nx=$nx, n1=$n1"
            
            # Update config values using sed
            sed -i "s/^  nx: .*/  nx: $nx/" "$CONFIG_PATH"
            sed -i "s/^  n1: .*/  n1: $n1/" "$CONFIG_PATH"
            
            # Run training
            python -m training.train --config "$CONFIG_PATH"
            
            echo "[$CONFIG_NAME] Completed nx=$nx, n1=$n1"
        done
    done

    # Restore original config
    mv "${CONFIG_PATH}.backup" "$CONFIG_PATH"
    echo "Restored original config: $CONFIG_PATH"
done

echo ""
echo "=== Grid search complete for all configs ==="
