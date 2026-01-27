#!/bin/bash
# Grid search for PSID parameters on beta_ecog config
# FIXED parameters: nx=15, n1=4, rescale_states=true, max_eigenvalue=0.995
# TUNED parameters: alpha_Q, alpha_R, backward_kalman

# NOTE: No 'set -e' - we want to continue on failures

CONFIG_PATH="training/setups/psid_SEARCH_beta_ecog.yaml"
FAILED_LOG="training/grid_search_failed_combinations.log"

# FIXED parameters
NX=15
N1=4
RESCALE_STATES=true
MAX_EIGENVALUE=0.995

# TUNED parameters
ALPHA_Q_VALUES=(1e-8 1e-6 1e-5 1e-4 1e-3 1e-2 0.1)
ALPHA_R_VALUES=(1e-8 1e-6 1e-5 1e-4 1e-3 1e-2 0.05)
BACKWARD_KALMAN_VALUES=(true false)

# Calculate total combinations
total=$((${#ALPHA_Q_VALUES[@]} * ${#ALPHA_R_VALUES[@]} * ${#BACKWARD_KALMAN_VALUES[@]}))

echo "=== PSID Grid Search (Fine-tuning) ==="
echo "Config: $CONFIG_PATH"
echo ""
echo "FIXED parameters:"
echo "  nx: $NX"
echo "  n1: $N1"
echo "  rescale_states: $RESCALE_STATES"
echo "  max_eigenvalue: $MAX_EIGENVALUE"
echo ""
echo "TUNED parameters:"
echo "  alpha_Q: ${ALPHA_Q_VALUES[*]}"
echo "  alpha_R: ${ALPHA_R_VALUES[*]}"
echo "  backward_kalman: ${BACKWARD_KALMAN_VALUES[*]}"
echo ""
echo "Total combinations: $total"
echo ""

# Backup original config
cp "$CONFIG_PATH" "${CONFIG_PATH}.backup"
echo "Created backup: ${CONFIG_PATH}.backup"

# Initialize failed log
echo "=== Failed Combinations Log ===" > "$FAILED_LOG"
echo "Started: $(date)" >> "$FAILED_LOG"
echo "Fixed: nx=$NX, n1=$N1, rescale_states=$RESCALE_STATES, max_eigenvalue=$MAX_EIGENVALUE" >> "$FAILED_LOG"
echo "" >> "$FAILED_LOG"

run_count=0
fail_count=0
success_count=0

for alpha_Q in "${ALPHA_Q_VALUES[@]}"; do
    for alpha_R in "${ALPHA_R_VALUES[@]}"; do
        for backward_kalman in "${BACKWARD_KALMAN_VALUES[@]}"; do
            ((run_count++))
            echo "[$run_count/$total] alpha_Q=$alpha_Q alpha_R=$alpha_R backward_kalman=$backward_kalman"
            
            # Update config with fixed and tuned values
            sed -i "s/^  nx: .*/  nx: $NX/" "$CONFIG_PATH"
            sed -i "s/^  n1: .*/  n1: $N1/" "$CONFIG_PATH"
            sed -i "s/^  alpha_Q: .*/  alpha_Q: $alpha_Q/" "$CONFIG_PATH"
            sed -i "s/^  alpha_R: .*/  alpha_R: $alpha_R/" "$CONFIG_PATH"
            sed -i "s/^  backward_kalman: .*/  backward_kalman: $backward_kalman/" "$CONFIG_PATH"
            sed -i "s/^  rescale_states: .*/  rescale_states: $RESCALE_STATES/" "$CONFIG_PATH"
            sed -i "s/^  max_eigenvalue: .*/  max_eigenvalue: $MAX_EIGENVALUE/" "$CONFIG_PATH"
            
            # Run training with error handling
            if python -m training.train --config "$CONFIG_PATH"; then
                ((success_count++))
                echo "[$run_count/$total] Completed successfully"
            else
                ((fail_count++))
                echo "[$run_count/$total] FAILED"
                echo "[$run_count] alpha_Q=$alpha_Q alpha_R=$alpha_R backward_kalman=$backward_kalman" >> "$FAILED_LOG"
            fi
        done
    done
done

# Restore original config
mv "${CONFIG_PATH}.backup" "$CONFIG_PATH"

# Finalize failed log
echo "" >> "$FAILED_LOG"
echo "Finished: $(date)" >> "$FAILED_LOG"
echo "Total failed: $fail_count" >> "$FAILED_LOG"

echo ""
echo "=== Grid search complete ==="
echo "Total runs: $run_count"
echo "Successful: $success_count"
echo "Failed: $fail_count"
if [ $fail_count -gt 0 ]; then
    echo "See failed combinations in: $FAILED_LOG"
fi
echo "Restored original config: $CONFIG_PATH"
echo ""
echo "Run analysis with: python training/analyze_grid_search.py"
