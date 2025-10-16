# MPCRL Examples

This directory contains example Jupyter notebooks demonstrating the use of the MPCRL package.

## Notebooks

### 01_environment_setup.ipynb
**Main RL Training Script**

This notebook demonstrates the complete reinforcement learning training loop for wind farm control:
- Environment initialization with turbulence models
- MPC-based control strategy
- Real-time parameter estimation (wind direction, wind speed, turbulence intensity)
- Power optimization and yaw control
- Visualization of results including power output, yaw angles, and estimated vs true wind conditions

### 02_basic_usage.ipynb
Basic usage examples and getting started guide.

### 03_training_loop.ipynb
Detailed training loop implementations and configurations.

### 04_full_examples.ipynb
Complete end-to-end examples with various scenarios.

## Running the Examples

Make sure you have the MPCRL package installed and all dependencies available. Each notebook is self-contained and can be run independently.

## Running the Notebooks

Make sure you have installed the package first:

```bash
# From the repository root
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

Then start Jupyter:

```bash
jupyter notebook
```

## Note

These notebooks assume the MPCRL package is properly installed. If you see import errors, make sure you've run the installation command above.
