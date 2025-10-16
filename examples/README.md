# Examples

This directory contains Jupyter notebooks demonstrating how to use the MPCRL package.

## Notebooks

1. **01_environment_setup.ipynb** - Environment setup and basic usage
   - Shows how to create an MPCenv environment
   - Demonstrates basic configuration options
   - Runs simple episodes with random actions

2. **02_basic_usage.ipynb** - Basic MPC usage examples
   - Simple wind farm setups
   - Basic optimization examples
   - Visualization of results

3. **03_training_loop.ipynb** - Training loop examples
   - Full RL training loop implementation
   - Reward tracking and analysis
   - Performance evaluation

4. **04_full_examples.ipynb** - Comprehensive examples
   - Advanced usage patterns
   - Integration with different RL algorithms
   - Performance optimization tips

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
