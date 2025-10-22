# MPC+RL Evaluation Framework - Complete Summary

This document provides a complete overview of the evaluation framework for your MPC+RL wind farm control project.

## 📋 Overview

Your evaluation framework allows you to compare:
1. **Greedy Baseline** - Standard operation (no control)
2. **MPC with Sensor Estimation** - Practical baseline
3. **MPC with Simple Estimator** - Observation-based
4. **RL+MPC** - Your approach (learned estimation)
5. **Oracle MPC** - Upper bound (perfect info)

All evaluations produce the same data format for easy comparison.

## 🚀 Quick Start

### Option 1: Use the Automated Script

```bash
./run_full_evaluation.sh
```

### Option 2: Run Manually

```bash
# 1. Baselines
python eval_greedy_baseline.py
python eval_mpc_baselines.py --agent_type front_turbine
python eval_mpc_baselines.py --agent_type simple_estimator
python eval_mpc_baselines.py --agent_type oracle

# 2. Your RL+MPC model
python eval_sac_mpc.py

# 3. Generate plots
python plot_evaluation_results.py --rl_models testrun7
```

## 📁 File Structure

```
mpcrl/
├── eval_sac_mpc.py              # Evaluate RL+MPC agents
├── eval_mpc_baselines.py        # Evaluate MPC baselines (oracle, sensor, simple)
├── eval_greedy_baseline.py      # Evaluate greedy (no control)
├── plot_evaluation_results.py   # Generate all plots
├── run_full_evaluation.sh       # Automated pipeline
│
├── mpc_baseline_agents.py       # Baseline agent implementations
├── mpcrl/
│   ├── environment_eval.py      # MPCenvEval class
│   └── ...
│
├── evals/                       # Generated evaluation data
│   ├── greedy_baseline.nc
│   ├── mpc_oracle.nc
│   ├── mpc_front_turbine.nc
│   ├── mpc_simple_estimator.nc
│   └── testrun7.nc              # Your RL models
│
├── plots/                       # Generated plots
│   ├── time_series_*.png
│   ├── aggregated_performance.png
│   ├── estimation_accuracy.png
│   ├── performance_heatmap.png
│   └── performance_summary.csv
│
└── Documentation/
    ├── EVAL_README.md           # Main evaluation guide
    ├── BASELINE_EVAL_README.md  # Baseline evaluation details
    ├── PLOTTING_GUIDE.md        # Plotting instructions
    └── EVALUATION_SUMMARY.md    # This file
```

## 📊 Generated Outputs

### Evaluation Data Files (.nc)

Each evaluation produces an xarray Dataset containing:
- **Time series**: Power, yaw angles, wind speeds
- **Estimates**: Estimated wind conditions (wd, ws, TI)
- **Baseline comparison**: Power increase vs greedy
- **Metadata**: Wind conditions, simulation parameters

### Plots

1. **time_series_*.png**: Power and estimation over time
2. **aggregated_performance.png**: Bar charts comparing all methods
3. **estimation_accuracy.png**: RMSE and bias for wind estimates
4. **performance_heatmap.png**: Performance across wind conditions
5. **performance_summary.csv**: Quantitative results table

## 🔬 Research Workflow

### For a Single Experiment

```bash
# 1. Train your model
python sac_MPC_local.py --exp_name testrun7

# 2. Evaluate all baselines
python eval_greedy_baseline.py
python eval_mpc_baselines.py --agent_type front_turbine
python eval_mpc_baselines.py --agent_type oracle

# 3. Evaluate your model
python eval_sac_mpc.py  # Set model_folder="testrun7"

# 4. Generate plots
python plot_evaluation_results.py --rl_models testrun7

# 5. Analyze results
cat plots/performance_summary.txt
```

### For Multiple Seeds (Publication Quality)

```bash
# 1. Train multiple seeds
for seed in 1 2 3 4 5; do
    python sac_MPC_local.py --seed $seed --exp_name run_seed${seed}
done

# 2. Evaluate all baselines once
python eval_greedy_baseline.py
python eval_mpc_baselines.py --agent_type front_turbine
python eval_mpc_baselines.py --agent_type oracle

# 3. Evaluate each seed
for seed in 1 2 3 4 5; do
    # Update model_folder in eval_sac_mpc.py to run_seed${seed}
    python eval_sac_mpc.py
done

# 4. Generate plots with statistics
python plot_evaluation_results.py --rl_models run_seed1 run_seed2 run_seed3 run_seed4 run_seed5

# 5. Results now include mean ± std for RL+MPC
cat plots/performance_summary.txt
```

## 📈 Key Metrics to Report

### 1. Mean Power Increase
- How much more power vs greedy baseline (%)
- **Expected**: Oracle > RL+MPC > Sensor > Greedy

### 2. Estimation Accuracy
- RMSE for wind direction, speed, and TI
- **Goal**: RL+MPC closer to Oracle than Sensor

### 3. Robustness
- Standard deviation across seeds
- Performance across different wind conditions
- **Target**: Low std dev (<1%), positive increase in all conditions

### 4. Computational Cost
- Inference time per step
- Training time
- **Trade-off**: More computation for better performance?

## 🎯 Interpreting Results

### Success Criteria

✅ **RL+MPC > Sensor MPC**: Your approach beats simple baseline
✅ **RL+MPC < Oracle MPC**: There's room for improvement (realistic)
✅ **Low variance**: Consistent performance across seeds
✅ **Positive everywhere**: Gains in all wind conditions

### Red Flags

⚠️ **RL+MPC < Sensor MPC**: Need more training or better reward
⚠️ **High variance**: Unstable training, tune hyperparameters
⚠️ **Negative in some conditions**: Control strategy issues
⚠️ **High estimation bias**: Systematic errors in RL estimates

## 📝 For Your Paper

### Methods Section

```
We evaluate our RL+MPC approach against four baselines:
1. Greedy (no control) - lower bound
2. MPC with sensor-based estimation - practical baseline
3. MPC with simple estimator - observation-based baseline
4. Oracle MPC (perfect information) - upper bound

All methods use the same MPC controller; only the wind condition
estimation strategy differs. We evaluate on N={number} random seeds
with M={number} different wind conditions.
```

### Results Section

Use these figures:
- **Fig 1**: Aggregated performance (bar chart)
- **Fig 2**: Estimation accuracy (RMSE comparison)
- **Fig 3**: Time series example
- **Table 1**: Performance summary with statistics

Sample text:
```
Our RL+MPC approach achieves X.XX% ± Y.YY% power increase over the
greedy baseline, compared to Z.ZZ% for the sensor-based MPC baseline
and W.WW% for the oracle upper bound. This demonstrates that the learned
estimation strategy captures [X-Y]% of the gap between sensor-based
estimation and perfect information.
```

## 🔧 Customization

### Different Wind Conditions

Edit any evaluation script:
```python
wdirs = [260, 265, 270, 275, 280]
wss = [8, 9, 10, 11]
TIs = [0.05, 0.10, 0.15]
```

### Different Simulation Length

```python
T_SIM = 2000  # 2000 seconds instead of 1000
```

### Different Turbine Layouts

Modify the `generate_square_grid` call:
```python
x_pos, y_pos = generate_square_grid(turbine=turbine,
                                    nx=4, ny=2,  # 4x2 grid instead of 3x1
                                    xDist=5, yDist=5)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Model not found"**: Check `runs/{model_name}/` has `.pt` files
2. **"Cannot access wandb"**: Set `wandb_project=None` or provide credentials
3. **Dimension mismatch**: Ensure all evals use same `dt_sim`, `dt_env`, `t_sim`
4. **NaN in results**: Check TI measurements enabled in config
5. **Empty plots**: Verify all evaluation files exist in `evals/`

### Getting Help

1. Check the specific README files:
   - `EVAL_README.md` - RL+MPC evaluation
   - `BASELINE_EVAL_README.md` - Baseline evaluation
   - `PLOTTING_GUIDE.md` - Plotting details

2. Check evaluation outputs:
   ```bash
   ls evals/
   python -c "import xarray as xr; ds = xr.open_dataset('evals/testrun7.nc'); print(ds)"
   ```

3. Test plotting with existing data:
   ```bash
   python plot_evaluation_results.py --rl_models testrun7
   ```

## 📚 Additional Resources

- **WindGym Documentation**: For environment details
- **CleanRL Documentation**: For SAC implementation details
- **xarray Documentation**: For working with evaluation data
- **MPC Paper**: For controller details

## 🎓 Citation

If you use this evaluation framework, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## ✅ Checklist for Complete Evaluation

- [ ] Train RL+MPC model(s)
- [ ] Run greedy baseline evaluation
- [ ] Run sensor MPC baseline
- [ ] Run oracle MPC baseline
- [ ] Run RL+MPC evaluation
- [ ] Generate all plots
- [ ] Check performance summary table
- [ ] Verify results make sense (Oracle > RL > Sensor > Greedy)
- [ ] If using multiple seeds, check low variance
- [ ] Document any issues or anomalies

## 🚀 Next Steps

1. **Collect results** from all evaluations
2. **Generate plots** and analyze patterns
3. **Statistical analysis** if using multiple seeds
4. **Write up results** for your paper
5. **Share findings** with collaborators

Good luck with your research! 🎉
