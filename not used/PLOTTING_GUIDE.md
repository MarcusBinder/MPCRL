# Plotting Guide for Evaluation Results

This guide explains how to visualize and analyze your MPC+RL evaluation results.

## Quick Start

### Basic Usage

After running evaluations, generate all plots:

```bash
python plot_evaluation_results.py --rl_models testrun7
```

### Multiple Seeds

If you trained with multiple random seeds:

```bash
python plot_evaluation_results.py --rl_models testrun7 testrun8 testrun9
```

This will compute mean ± std across seeds for RL+MPC results.

### Custom Output Directory

```bash
python plot_evaluation_results.py --rl_models testrun7 --output_dir figures/
```

## Generated Plots

The script generates 5 types of visualizations:

### 1. Time Series Comparison (`time_series_ws9_wd270_ti0.05.png`)

Three subplots showing evolution over time:
- **Farm Power**: Total power output for all methods
- **Power Increase**: Percentage increase vs greedy baseline
- **Wind Direction Estimation**: How well each method estimates WD

**Shows:**
- RL+MPC with shaded std dev band (if multiple seeds)
- Temporal dynamics and stability
- Estimation quality over time

### 2. Aggregated Performance (`aggregated_performance.png`)

Bar charts comparing all methods:
- **Mean Farm Power**: Absolute power output (MW)
- **Power Increase**: Percentage improvement vs greedy

**Shows:**
- Error bars for RL+MPC (if multiple seeds)
- Clear ranking of methods
- Statistical significance

### 3. Estimation Accuracy (`estimation_accuracy.png`)

Six subplots (2 rows × 3 columns):
- **Top row**: RMSE for WD, WS, TI estimation
- **Bottom row**: Bias for WD, WS, TI estimation

**Shows:**
- How accurately each method estimates wind conditions
- Whether methods over/under-estimate (bias)
- Comparison between Oracle, Sensor, Simple Estimator, and RL+MPC

### 4. Performance Heatmap (`performance_heatmap.png`)

Color maps showing power increase across wind conditions:
- One panel per method (Sensor MPC, RL+MPC, Oracle MPC)
- X-axis: Wind direction
- Y-axis: Wind speed
- Color: Power increase (%)

**Shows:**
- Which wind conditions benefit most from control
- Robustness across operating conditions
- Where RL+MPC excels vs baselines

**Note:** Only generated if you evaluated multiple wind conditions.

### 5. Performance Summary (`performance_summary.csv` and `.txt`)

Table with key metrics:
- Method name
- Mean power output (MW)
- Power increase vs greedy (%)
- Standard deviation (for RL+MPC)

**Example:**
```
===============================================================================
                         Method  Mean Power (MW) Power Increase (%) Std Dev (%)
                Greedy (No Control)           8.45                0.00           -
                     Sensor MPC           8.92                5.56           -
          Simple Estimator MPC           8.95                5.92           -
        Oracle MPC (Upper Bound)           9.12                7.93           -
            RL+MPC (Ours, n=3)      9.05 ± 0.03                7.10        0.25
===============================================================================
```

## Command Line Options

```bash
python plot_evaluation_results.py --help
```

### Available Options

- `--rl_models MODEL1 MODEL2 ...`: RL model names to include
- `--output_dir DIR`: Output directory (default: `plots/`)
- `--ws SPEED`: Wind speed for time series (default: 9.0)
- `--wd DIRECTION`: Wind direction for time series (default: 270.0)
- `--ti INTENSITY`: Turbulence intensity for time series (default: 0.05)

## Example Workflows

### Single RL Model

```bash
# 1. Run all evaluations
python eval_greedy_baseline.py
python eval_mpc_baselines.py --agent_type front_turbine
python eval_mpc_baselines.py --agent_type oracle
python eval_sac_mpc.py

# 2. Generate plots
python plot_evaluation_results.py --rl_models testrun7
```

### Multiple Seeds for Statistical Significance

```bash
# 1. Train with different seeds
python sac_MPC_local.py --seed 1 --exp_name testrun_seed1
python sac_MPC_local.py --seed 2 --exp_name testrun_seed2
python sac_MPC_local.py --seed 3 --exp_name testrun_seed3

# 2. Evaluate each seed
python eval_sac_mpc.py  # Modify to evaluate testrun_seed1
python eval_sac_mpc.py  # Modify to evaluate testrun_seed2
python eval_sac_mpc.py  # Modify to evaluate testrun_seed3

# 3. Generate plots with confidence intervals
python plot_evaluation_results.py --rl_models testrun_seed1 testrun_seed2 testrun_seed3
```

### Custom Time Series Conditions

Plot time series for a different wind condition:

```bash
python plot_evaluation_results.py --rl_models testrun7 \
    --ws 10 --wd 265 --ti 0.10
```

This generates `time_series_ws10_wd265_ti0.1.png`.

## Tips for Paper Figures

### High-Quality Figures

The script saves at 300 DPI by default. For publications:

1. **Vector graphics**: Modify the script to save as `.pdf` instead of `.png`
2. **Font sizes**: Adjust `plt.rcParams['font.size']` for readability
3. **Colors**: Colorblind-friendly palette is used by default

### Multi-Seed Reporting

For publication, always use multiple seeds (≥3):

```bash
# Train 5 seeds
for seed in 1 2 3 4 5; do
    python sac_MPC_local.py --seed $seed --exp_name run_seed${seed}
done

# Evaluate all
for seed in 1 2 3 4 5; do
    # Update eval script to evaluate run_seed${seed}
    python eval_sac_mpc.py
done

# Plot with error bars
python plot_evaluation_results.py --rl_models run_seed1 run_seed2 run_seed3 run_seed4 run_seed5
```

### Customizing Plots

To customize plots, edit `plot_evaluation_results.py`:

```python
# Change figure size
fig, axes = plt.subplots(3, 1, figsize=(16, 12))  # Larger

# Change colors
colors = {
    'Greedy': '#your_color',
    'RL+MPC (mean)': '#your_color',
}

# Add annotations
ax.annotate('Important point', xy=(x, y), ...)
```

## Interpreting Results

### Good RL+MPC Performance

✅ RL+MPC power increase is between Sensor MPC and Oracle MPC
✅ Low standard deviation across seeds (<1%)
✅ Low estimation RMSE (close to Oracle)
✅ Positive power increase across all wind conditions

### Potential Issues

⚠️ **RL+MPC worse than Sensor MPC**: May need more training or reward tuning
⚠️ **High std dev across seeds**: Training instability, try different hyperparameters
⚠️ **High estimation bias**: RL agent systematically over/under-estimates
⚠️ **Negative power in some conditions**: Control strategy may be suboptimal

## Troubleshooting

### "No evaluation data found"

Make sure you've run evaluations first:
```bash
ls evals/
# Should show: greedy_baseline.nc, mpc_*.nc, testrun7.nc, etc.
```

### "Skipping heatmap"

Heatmap requires multiple wind conditions. Evaluate with:
```bash
# In eval_sac_mpc.py or eval_mpc_baselines.py, set:
wdirs = [265, 270, 275]
wss = [8, 9, 10]
TIs = [0.05, 0.10]
```

### Missing data in plots

Check that all baselines were evaluated:
```bash
ls evals/
# Should have: greedy_baseline.nc, mpc_oracle.nc, mpc_front_turbine.nc, etc.
```

### Dimension mismatch errors

All evaluations must use the same:
- Time simulation length (`t_sim=1000`)
- Environment parameters (`dt_sim`, `dt_env`, etc.)
- Wind condition coordinates

## Advanced: Creating Custom Plots

You can load the datasets yourself for custom analysis:

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load data
ds_rl = xr.open_dataset('evals/testrun7.nc')
ds_oracle = xr.open_dataset('evals/mpc_oracle.nc')

# Custom plot: RL estimation error over time
wd_error = ds_rl.sel(ws=9, wd=270, TI=0.05).estimated_wd.squeeze() - 270

plt.figure(figsize=(10, 4))
plt.plot(ds_rl.time, wd_error)
plt.xlabel('Time (s)')
plt.ylabel('Wind Direction Error (°)')
plt.title('RL Agent WD Estimation Error')
plt.grid(True, alpha=0.3)
plt.savefig('custom_plot.png', dpi=300)
```

## Next Steps

After generating plots:

1. **Analyze results**: Look for patterns in estimation accuracy and performance
2. **Statistical tests**: Use multiple seeds for t-tests or ANOVA
3. **Sensitivity analysis**: Vary wind conditions to test robustness
4. **Paper writing**: Use summary table and aggregated plots for comparison
5. **Debugging**: If performance is poor, check time series for insights
