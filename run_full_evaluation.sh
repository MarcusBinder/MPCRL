#!/bin/bash
# Complete evaluation pipeline for MPC+RL wind farm control
# This script runs all baselines and RL evaluations, then generates plots

set -e  # Exit on error

echo "========================================================================"
echo "    MPC+RL WIND FARM CONTROL - COMPLETE EVALUATION PIPELINE"
echo "========================================================================"
echo ""

# Configuration
RL_MODELS=("testrun7")  # Add more models here for multiple seeds
NUM_WORKERS=4

echo "Configuration:"
echo "  RL Models: ${RL_MODELS[@]}"
echo "  Parallel Workers: $NUM_WORKERS"
echo ""

# Create evals directory if it doesn't exist
mkdir -p evals

echo "========================================================================"
echo "STEP 1: Evaluating Greedy Baseline (No Control)"
echo "========================================================================"
python eval_greedy_baseline.py --num_workers $NUM_WORKERS
echo ""

echo "========================================================================"
echo "STEP 2: Evaluating MPC Baselines"
echo "========================================================================"

echo "→ Oracle MPC (Upper Bound)..."
python eval_mpc_baselines.py --agent_type oracle --num_workers $NUM_WORKERS
echo ""

echo "→ Front Turbine MPC (Sensor-Based)..."
python eval_mpc_baselines.py --agent_type front_turbine --num_workers $NUM_WORKERS
echo ""

echo "→ Simple Estimator MPC..."
python eval_mpc_baselines.py --agent_type simple_estimator --num_workers $NUM_WORKERS
echo ""

echo "========================================================================"
echo "STEP 3: Evaluating RL+MPC Models"
echo "========================================================================"

# Note: You need to manually configure eval_sac_mpc.py to evaluate specific models
# Or modify this section to pass model names as arguments

for model in "${RL_MODELS[@]}"; do
    echo "→ Evaluating $model..."
    # You'll need to modify eval_sac_mpc.py to accept model name as argument
    # or manually update the model_folder in eval_sac_mpc.py
    echo "  [Skipping - please run manually: python eval_sac_mpc.py]"
    echo "  [Update model_folder in eval_sac_mpc.py to: $model]"
done
echo ""

echo "========================================================================"
echo "STEP 4: Generating Plots"
echo "========================================================================"

# Check if all required files exist
required_files=(
    "evals/greedy_baseline.nc"
    "evals/mpc_oracle.nc"
    "evals/mpc_front_turbine.nc"
)

all_exist=true
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠ Warning: $file not found"
        all_exist=false
    fi
done

if [ "$all_exist" = true ]; then
    echo "→ Generating plots..."
    python plot_evaluation_results.py --rl_models "${RL_MODELS[@]}" --output_dir plots
    echo ""

    echo "========================================================================"
    echo "✓ EVALUATION COMPLETE!"
    echo "========================================================================"
    echo ""
    echo "Results saved to:"
    echo "  - Evaluation data: evals/*.nc"
    echo "  - Plots: plots/*.png"
    echo "  - Summary: plots/performance_summary.csv"
    echo ""
    echo "Next steps:"
    echo "  1. View plots in the 'plots' directory"
    echo "  2. Check performance_summary.csv for quantitative results"
    echo "  3. Analyze time series data for detailed insights"
    echo ""
else
    echo "⚠ Some evaluation files are missing. Please run evaluations manually."
    echo "  See EVAL_README.md for instructions."
fi
