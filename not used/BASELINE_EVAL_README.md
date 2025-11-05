# MPC Baseline Evaluation Guide

This guide explains how to evaluate the MPC controller with different wind condition estimation strategies (without the RL agent).

## Overview

The baseline evaluation allows you to test MPC performance with different approaches and compare against standard operation:

### MPC Baselines (Different Estimation Strategies)

1. **Oracle Agent** (`oracle`): Perfect ground truth wind conditions
   - Upper bound on MPC performance
   - Represents perfect forecasting scenario
   - Uses actual wind conditions directly from the environment

2. **Front Turbine Agent** (`front_turbine`): Estimates from front turbine measurements
   - Practical baseline using available sensor data
   - Measures wind at the upwind-most turbine
   - Applies smoothing to reduce measurement noise
   - Represents real-world scenario with local measurements

3. **Simple Estimator Agent** (`simple_estimator`): Uses farm-level observations
   - Extracts estimates from the observation vector
   - Uses farm-level measurements available to the controller
   - Middle ground between oracle and sensor-based estimation

### No Control Baseline

4. **Greedy Agent** (`greedy`): No wake steering control
   - Zero yaw offset (turbines face the wind)
   - Standard wind turbine operation
   - Lower bound - shows performance without any control
   - Uses `eval_greedy_baseline.py` (separate script)

## Quick Start

### Evaluate Oracle MPC (Perfect Information)

```bash
python eval_mpc_baselines.py --agent_type oracle
```

This will:
- Evaluate MPC with perfect wind condition knowledge
- Save results to `evals/mpc_oracle.nc`
- Show the theoretical upper bound of MPC performance

### Evaluate Front Turbine MPC (Sensor-Based)

```bash
python eval_mpc_baselines.py --agent_type front_turbine --smoothing_window 5
```

This will:
- Estimate wind conditions from the front turbine measurements
- Apply 5-timestep moving average smoothing
- Save results to `evals/mpc_front_turbine.nc`
- Represent practical MPC performance with real sensors

### Evaluate Simple Estimator MPC

```bash
python eval_mpc_baselines.py --agent_type simple_estimator
```

### Evaluate Greedy Agent (No Control)

```bash
python eval_greedy_baseline.py
```

This represents standard turbine operation with no wake steering control.

## Comparing All Approaches

To get a complete picture, evaluate all baselines:

```bash
# 1. No control baseline (lower bound)
python eval_greedy_baseline.py

# 2. Sensor-based MPC
python eval_mpc_baselines.py --agent_type front_turbine

# 3. Simple estimator MPC
python eval_mpc_baselines.py --agent_type simple_estimator

# 4. Your RL + MPC approach
python eval_sac_mpc.py

# 5. Oracle MPC (upper bound)
python eval_mpc_baselines.py --agent_type oracle

# 6. Compare results
python compare_agents.py  # (you'll need to create this)
```

This gives you the full spectrum:
- **Greedy**: Lower bound (no control)
- **Sensor/Simple MPC**: Practical baselines
- **RL + MPC**: Your approach
- **Oracle MPC**: Upper bound (perfect info)

## Command Line Options

```bash
python eval_mpc_baselines.py --help
```

### Available Options

- `--agent_type`: Choose baseline agent (`oracle`, `front_turbine`, `simple_estimator`)
- `--output_name`: Custom output filename (default: `mpc_{agent_type}.nc`)
- `--num_workers`: Number of parallel evaluation workers (default: 4)
- `--smoothing_window`: Smoothing window for front_turbine agent (default: 3)

## Customizing Evaluation Conditions

To evaluate on custom wind conditions, edit the script:

```python
# In eval_mpc_baselines.py, modify:
wdirs = [260, 265, 270, 275, 280]  # More wind directions
wss = [8, 9, 10, 11]  # Multiple wind speeds
TIs = [0.05, 0.10, 0.15]  # Multiple turbulence intensities
```

## Output Format

All baseline evaluations produce the same xarray dataset format as `eval_sac_mpc.py`:

- **powerF_a**: Total farm power
- **powerT_a**: Per-turbine power
- **yaw_a**: Yaw angles
- **ws_a**: Wind speeds at turbines
- **reward**: Step rewards
- **estimated_wd**: Estimated wind direction fed to MPC
- **estimated_ws**: Estimated wind speed fed to MPC
- **estimated_ti**: Estimated turbulence intensity fed to MPC
- **powerF_b**: Baseline power (if enabled)
- **pct_inc**: Percentage power increase

## Analysis Example

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load all results
ds_greedy = xr.open_dataset("evals/greedy_baseline.nc")
ds_sensor = xr.open_dataset("evals/mpc_front_turbine.nc")
ds_rl = xr.open_dataset("evals/testrun7.nc")
ds_oracle = xr.open_dataset("evals/mpc_oracle.nc")

# Compare mean power increase
# Note: Greedy uses powerF_a directly (no pct_inc since baseline=greedy)
greedy_power = ds_greedy.sel(ws=9, wd=270, TI=0.05).powerF_a.mean().values
sensor_power_inc = ds_sensor.sel(ws=9, wd=270, TI=0.05).pct_inc.mean().values
rl_power_inc = ds_rl.sel(ws=9, wd=270, TI=0.05).pct_inc.mean().values
oracle_power_inc = ds_oracle.sel(ws=9, wd=270, TI=0.05).pct_inc.mean().values

print("=== Performance Comparison ===")
print(f"Greedy (no control): baseline - {greedy_power:.2f} W")
print(f"Sensor MPC: +{sensor_power_inc:.2f}% vs baseline")
print(f"RL + MPC: +{rl_power_inc:.2f}% vs baseline")
print(f"Oracle MPC (upper bound): +{oracle_power_inc:.2f}% vs baseline")

# Plot estimation errors
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for i, (ds, name) in enumerate([(ds_oracle, 'Oracle'),
                                  (ds_sensor, 'Sensor'),
                                  (ds_rl, 'RL')]):
    # Wind direction error
    wd_error = ds.sel(ws=9, wd=270, TI=0.05).estimated_wd.squeeze() - 270
    wd_error.plot(ax=axes[0, i])
    axes[0, i].set_title(f'{name} - WD Error')

    # Wind speed error
    ws_error = ds.sel(ws=9, wd=270, TI=0.05).estimated_ws.squeeze() - 9
    ws_error.plot(ax=axes[1, i])
    axes[1, i].set_title(f'{name} - WS Error')

    # TI error
    ti_error = ds.sel(ws=9, wd=270, TI=0.05).estimated_ti.squeeze() - 0.05
    ti_error.plot(ax=axes[2, i])
    axes[2, i].set_title(f'{name} - TI Error')

plt.tight_layout()
plt.savefig('estimation_comparison.png')
```

## Understanding the Results

### Oracle MPC Performance
- Represents **maximum achievable** power gains with perfect forecasting
- Any performance gap between Oracle and RL+MPC shows estimation losses
- Useful for understanding the ceiling of your approach

### Front Turbine Performance
- Represents **practical baseline** using existing sensors
- Shows what's achievable without ML/RL
- Performance gap vs RL+MPC demonstrates the value of learned estimation

### Key Metrics to Compare

1. **Mean Power Increase**: How much each approach improves over baseline
2. **Estimation Accuracy**: RMSE of wind condition estimates
3. **Variance**: Stability of power output
4. **Computational Cost**: Oracle/Sensor are much faster than RL inference

## Tips

### Adjusting Front Turbine Agent

The `smoothing_window` parameter affects estimation quality:
- **Smaller window (1-3)**: More responsive but noisier
- **Larger window (5-10)**: Smoother but slower to adapt

Test different values:
```bash
python eval_mpc_baselines.py --agent_type front_turbine --smoothing_window 1
python eval_mpc_baselines.py --agent_type front_turbine --smoothing_window 10
```

### Expected Results

Typical performance ranking (best to worst power increase):
1. **Oracle MPC**: Best (perfect info, upper bound)
2. **RL + MPC**: Good (learned estimation)
3. **Sensor MPC**: Baseline (simple estimation)
4. **Simple Estimator**: Similar to Sensor
5. **Greedy**: Worst (no control, lower bound - 0% increase)

If RL+MPC performs worse than Sensor MPC, the RL agent may need more training or the reward function may need tuning.

## Implementation Details

### Oracle Agent
- Directly reads `env.wd`, `env.ws`, `env.ti`
- No estimation error
- Represents perfect nowcast/forecast

### Front Turbine Agent
- Measures wind at turbine 0 (assumes it's upwind)
- Computes wind direction from wind vector
- Uses moving average for smoothing
- TI estimation is simplified (uses env.ti as proxy)

### Simple Estimator Agent
- Uses `farm_measurements` from the environment
- Extracts farm-level wind measurements
- Similar to what's available in the observation space

### Greedy Agent
- No control strategy - always outputs zero yaw offset
- Turbines face directly into the wind (greedy behavior)
- Uses `FarmEval` environment (not `MPCenv`)
- Evaluated with separate script (`eval_greedy_baseline.py`)
- Represents industry standard operation

## Next Steps

After running baseline evaluations, you can:

1. **Compare all approaches** to quantify RL agent's value
2. **Analyze estimation errors** to understand where RL helps most
3. **Create visualizations** showing performance across conditions
4. **Write paper results** with clear baseline comparisons
