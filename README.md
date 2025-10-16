# MPCRL - Model Predictive Control for Wind Farm Optimization

A wind farm control optimization system that combines Model Predictive Control (MPC) with Reinforcement Learning to maximize power output through intelligent wake steering.

## Overview

MPCRL optimizes wind turbine yaw angles (the direction turbines face) to redirect turbulent wakes away from downstream turbines, improving overall wind farm efficiency. By strategically "leaning" upstream turbines, the system ensures downstream turbines receive cleaner, more energetic wind flow.

The project implements a sophisticated "back-to-front" optimization strategy that accounts for wake propagation delays and uses physics-based simulation to find optimal control trajectories.

## Key Features

- **Physics-Based Wake Modeling**: Uses PyWake library with Gaussian wake models, Crespo-Hernandez turbulence, and Jimenez wake deflection
- **Sequential Optimization**: Back-to-front turbine optimization accounting for wake propagation delays
- **Parameterized Yaw Trajectories**: Smooth, realistic yaw angle changes using basis functions
- **Intelligent Caching**: LRU cache with quantization for efficient repeated power calculations
- **RL Integration**: Gymnasium environment for reinforcement learning agent training
- **Time-Delayed Cost Functions**: Models realistic wake propagation effects

## How It Works

1. **Wind Estimation**: An RL agent (or manual input) provides estimates of wind conditions (speed, direction, turbulence intensity)
2. **MPC Optimization**: The system runs dual annealing optimization from downstream to upstream turbines
3. **Trajectory Generation**: Each turbine receives parameterized yaw trajectories (controlled by 2 parameters per turbine)
4. **Wake Physics**: PyWake calculates power output accounting for wake steering effects
5. **Delay Modeling**: Wake propagation delays between turbines are modeled based on wind speed
6. **Action Execution**: Optimized yaw angles are applied to the wind farm simulation

## Installation

### Dependencies

```bash
pip install numpy scipy matplotlib
pip install py_wake gymnasium
pip install windgym  # For RL environment integration
```

### Data Requirements

The project uses wind field datasets in NetCDF format (e.g., `Hipersim_mann_l5.0...nc`).

## Usage

### Basic MPC Optimization

```python
from MPC import WindFarmModel, optimize_farm_back2front
import numpy as np

# Create wind farm model
model = WindFarmModel()

# Define wind conditions
wind_direction = 270.0  # degrees
wind_speed = 9.0        # m/s
turbulence_intensity = 0.06

# Run optimization
result = optimize_farm_back2front(
    model=model,
    wind_direction=wind_direction,
    wind_speed=wind_speed,
    turbulence_intensity=turbulence_intensity
)

print(f"Optimal yaw parameters: {result.x}")
print(f"Farm energy: {-result.fun}")  # Negative because we minimized -energy
```

### Using the RL Environment

```python
from MPCenv import MPCenv
from utils import make_config

# Create environment
config = make_config()
env = MPCenv(config)

# Run episode
obs, info = env.reset()
for _ in range(100):
    # Action: [wind_direction, wind_speed, turbulence_intensity]
    action = [270.0, 9.0, 0.06]
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
```

### Running Tests

```python
from test_mpc import *

# Run all visual validation tests
test_saturation()
test_psi_function()
test_yaw_trajectory()
test_cache_quantization()
test_turbine_sorting()
test_wake_delays()
test_power_calculation()
test_optimization()
test_energy_integration()
```

## Project Structure

```
mpcrl/
├── MPC.py              # Core wind farm model, optimization, and caching
├── MPCenv.py           # Gymnasium environment wrapper
├── utils.py            # Configuration helpers
├── test_mpc.py         # Comprehensive test suite with visualization
├── example_*.ipynb     # Example notebooks
└── data/               # Wind field datasets (NetCDF)
```

## Core Components

### MPC.py

- **Basis Functions**: `psi()` and `yaw_traj()` for smooth trajectory generation
- **YawCache**: Efficient caching system with quantization and LRU eviction
- **WindFarmModel**: Physics-based power calculations with wake steering
- **Optimization**: `optimize_farm_back2front()` using scipy's dual annealing

### MPCenv.py

- **MPCenv Class**: Gymnasium-compatible environment
- Integrates MPC optimization within RL training loop
- Configurable prediction horizons and action penalties

### Configuration Options

Key parameters (via `utils.make_config()`):
- Wind speed range: 8-10 m/s
- Wind direction range: 260-280 degrees
- Yaw limits: -45 to +45 degrees
- Prediction horizon: 100 seconds
- Cache size: 64,000 entries

## Technical Details

### Wake Steering Physics

The system uses power-yaw relationships from wind turbine aerodynamics:
- Yaw-induced power loss follows cos³(γ) law
- Wake deflection modeled with Jimenez model
- Turbulence effects via Crespo-Hernandez model

### Optimization Strategy

1. **Back-to-Front Ordering**: Optimizes from downstream to upstream
2. **Dual Annealing**: Global optimization algorithm for non-convex problems
3. **Warm Starting**: Uses previous solutions to accelerate convergence
4. **Time-Shifted Costs**: Accounts for delayed wake effects

### Parameterization

Each turbine's yaw trajectory uses 2 parameters:
- **o1**: Direction and magnitude of yaw change
- **o2**: Timing and transition behavior

Combined via: `yaw(t) = psi(t, o1, o2)`

## Example Notebooks

- `example_import_env.ipynb`: Environment setup and basic usage
- `example_loop.ipynb`: Full training loop examples

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

[Add contact information here]