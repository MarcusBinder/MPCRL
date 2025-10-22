# Evaluation Guide for MPC+RL Agents

This guide explains how to evaluate your trained MPC+RL agents using the `eval_sac_mpc.py` script.

## Overview

The evaluation script (`eval_sac_mpc.py`) evaluates trained SAC agents on specific wind conditions and saves detailed time series data for analysis. It supports:

- Parallel evaluation across multiple scenarios
- Integration with wandb for automatic config extraction
- Evaluation of multiple checkpoints
- Customizable wind conditions (speed, direction, turbulence)

## Quick Start

### Basic Usage

To evaluate the model in `runs/testrun7`:

```bash
python eval_sac_mpc.py
```

This will:
- Load the model from `runs/testrun7`
- Try to fetch configuration from wandb (project: MPC_RL)
- Evaluate on default conditions: wind directions [265, 270, 275°], wind speed 9 m/s, TI 0.05
- Save results to `evals/testrun7.nc`

### Custom Evaluation Scenarios

You can customize the evaluation by modifying the script or creating a wrapper:

```python
from eval_sac_mpc import EvalArgs

args = EvalArgs(
    model_folder="testrun7",
    wandb_project="MPC_RL",
    wandb_entity="your-entity",
    num_workers=8,
    wdirs=[260, 270, 280],  # Custom wind directions
    wss=[8, 9, 10],  # Multiple wind speeds
    TIs=[0.05, 0.10],  # Multiple turbulence intensities
    deterministic=True,  # Use deterministic policy
    eval_all_checkpoints=True  # Evaluate all saved checkpoints
)
```

## Configuration

### Automatic Config from wandb

If your model was logged to wandb, the script will automatically fetch:
- `dt_sim`: Simulation timestep
- `dt_env`: Environment timestep
- `yaw_step`: Yaw step size
- `turbtype`: Wind turbine type (DTU10MW or V80)
- `TI_type`: Turbulence model type
- `max_eps`: Maximum number of inflow passes
- `NetComplexity`: Network architecture size

### Manual Configuration

If wandb is not available, default values are used:
- dt_sim = 10
- dt_env = 30
- yaw_step = 0.3
- turbtype = "DTU10MW"
- TI_type = "None"
- max_eps = 30
- NetComplexity = "default"

## Output

The script saves results to `evals/{model_name}.nc` as an xarray dataset containing:

- **powerF_a**: Total farm power (agent)
- **powerT_a**: Per-turbine power (agent)
- **yaw_a**: Yaw angles (agent)
- **ws_a**: Wind speeds at turbines (agent)
- **reward**: Step rewards
- **estimated_wd**: Wind direction estimated by RL agent (fed to MPC)
- **estimated_ws**: Wind speed estimated by RL agent (fed to MPC)
- **estimated_ti**: Turbulence intensity estimated by RL agent (fed to MPC)
- **powerF_b**: Baseline farm power (if Baseline_comp=True)
- **powerT_b**: Baseline per-turbine power
- **yaw_b**: Baseline yaw angles
- **ws_b**: Baseline wind speeds
- **pct_inc**: Percentage power increase over baseline

### Dimensions

The dataset has the following dimensions:
- `time`: Simulation timesteps
- `turb`: Turbine index
- `ws`: Wind speed
- `wd`: Wind direction
- `TI`: Turbulence intensity
- `turbbox`: Turbulence box identifier
- `model_step`: Training step of the checkpoint
- `deterministic`: Whether deterministic policy was used

## Analysis Example

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load evaluation results
ds = xr.open_dataset("evals/testrun7.nc")

# Get mean power increase for a specific condition
power_inc = ds.sel(ws=9, wd=270, TI=0.05).pct_inc.mean()
print(f"Mean power increase: {power_inc.values:.2f}%")

# Plot farm power over time
ds.sel(ws=9, wd=270, TI=0.05).powerF_a.plot()

# Analyze RL agent's wind condition estimates
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Compare estimated vs actual wind direction
ds.sel(ws=9, wd=270, TI=0.05).estimated_wd.squeeze().plot(ax=axes[0], label='Estimated')
axes[0].axhline(y=270, color='r', linestyle='--', label='Actual')
axes[0].set_title('Wind Direction Estimate')
axes[0].legend()

# Compare estimated vs actual wind speed
ds.sel(ws=9, wd=270, TI=0.05).estimated_ws.squeeze().plot(ax=axes[1], label='Estimated')
axes[1].axhline(y=9, color='r', linestyle='--', label='Actual')
axes[1].set_title('Wind Speed Estimate')
axes[1].legend()

# Compare estimated vs actual turbulence intensity
ds.sel(ws=9, wd=270, TI=0.05).estimated_ti.squeeze().plot(ax=axes[2], label='Estimated')
axes[2].axhline(y=0.05, color='r', linestyle='--', label='Actual')
axes[2].set_title('Turbulence Intensity Estimate')
axes[2].legend()

plt.tight_layout()
plt.show()
```

## Troubleshooting

### Model not found error
- Ensure the model folder exists in `runs/`
- Check that `.pt` checkpoint files are present

### wandb connection issues
- The script will fall back to default config if wandb is unavailable
- Set `wandb_project=None` to skip wandb entirely

### Memory issues
- Reduce `num_workers` if you run out of memory
- Evaluate fewer checkpoints with `eval_all_checkpoints=False`

## Baseline Comparison

To understand the value of the RL agent, you can compare against several baselines:

### Greedy Agent (No Control - Lower Bound)
Evaluate standard turbine operation with no wake steering:
```bash
python eval_greedy_baseline.py
```

### Sensor-Based MPC (Practical Baseline)
Evaluate MPC with front turbine measurements:
```bash
python eval_mpc_baselines.py --agent_type front_turbine
```

### Oracle MPC (Upper Bound)
Evaluate MPC with perfect wind condition knowledge:
```bash
python eval_mpc_baselines.py --agent_type oracle
```

See [BASELINE_EVAL_README.md](BASELINE_EVAL_README.md) for detailed baseline evaluation instructions.

**Complete Evaluation Sequence:**
```bash
python eval_greedy_baseline.py                           # Lower bound
python eval_mpc_baselines.py --agent_type front_turbine  # Practical baseline
python eval_sac_mpc.py                                   # Your RL+MPC approach
python eval_mpc_baselines.py --agent_type oracle         # Upper bound
```

## Comparison with eval_sac_old.py

Key differences:
1. **Environment**: Uses `MPCenvEval` instead of `FarmEval`
2. **MPC Integration**: Evaluates the full MPC+RL pipeline
3. **Config**: Automatically extracts MPC-specific parameters from wandb
4. **Network Architecture**: Supports different network sizes (small, medium, large, etc.)

## Notes

- The script uses `AgentEvalFast` from windgym for the actual evaluation
- Parallel evaluation uses `pathos.pools.ProcessPool` for multi-core processing
- Results are compatible with the same analysis tools as `eval_sac_old.py`
