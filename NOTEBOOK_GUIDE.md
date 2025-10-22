# Evaluation Analysis Notebook Guide

## Quick Start

```bash
# Launch Jupyter
jupyter notebook evaluation_analysis.ipynb
```

## Notebook Structure

The notebook is organized into 7 main sections:

### 1. **Setup & Data Loading**
- Imports all necessary libraries
- Loads all evaluation datasets (greedy, baselines, RL+MPC)
- Defines helper function for extracting 1D time series
- **Action**: Modify the `eval_files` dict to point to your model names

### 2. **Quick Overview**
- Performance summary table showing all methods
- Quick comparison of mean power and power increase

### 3. **Time Series Analysis**
- **3.1**: Select wind condition (ws, wd, ti)
- **3.2**: Farm power over time for all methods
- **3.3**: Power increase vs greedy baseline
- **Action**: Change `ws_select`, `wd_select`, `ti_select` to explore different conditions

### 4. **Individual Turbine Analysis**
- **4.1**: Per-turbine power output
- **4.2**: Turbine yaw angles
- **4.3**: Compare yaw angles across methods
- **Action**: Change `method_to_analyze` and `turbine_idx` to focus on specific turbines/methods

### 5. **Wind Estimation Analysis**
- **5.1**: Wind direction estimates vs true
- **5.2**: Wind direction estimation error (with RMSE/bias stats)
- **5.3**: Wind speed estimates vs true
- **5.4**: Turbulence intensity estimates vs true
- **Shows**: How well each method estimates wind conditions

### 6. **Performance Comparison**
- **6.1**: Mean power comparison (bar chart)
- **6.2**: Power increase vs greedy (bar chart)
- **6.3**: Estimation accuracy summary table
- **Shows**: Aggregate statistics across all methods

### 7. **Custom Analysis**
- Empty cells for your own explorations
- Example: Calculate gap between RL+MPC and Oracle
- **Use**: Add your own analysis and plots here

## Key Variables to Modify

### Data Loading
```python
eval_files = {
    'RL+MPC': 'evals/testrun7.nc',  # ← Change to your model
}
```

### Wind Condition Selection
```python
ws_select = 9.0    # Wind speed
wd_select = 270.0  # Wind direction
ti_select = 0.05   # Turbulence intensity
```

### Method/Turbine Selection
```python
method_to_analyze = 'RL+MPC'  # Which method to analyze
turbine_idx = 0               # Which turbine to focus on
```

## Tips for Interactive Use

### 1. **Run All Cells First**
- Use `Cell → Run All` to generate all plots initially
- Then go back and modify parameters to explore

### 2. **Modify and Re-run**
- Change parameters in a cell
- Press `Shift+Enter` to re-run just that cell
- Plot will update immediately

### 3. **Compare Multiple Conditions**
- Run a cell with one condition
- Change parameters and run again
- Jupyter keeps both plots so you can scroll and compare

### 4. **Save Your Findings**
- Use `File → Download as → HTML` to save a static report
- Or `File → Download as → PDF` for presentations

### 5. **Add Your Own Cells**
- Click `+` button to add new cells
- Use for custom analysis, notes, or experiments

## Common Use Cases

### Compare Specific Turbines
```python
# In section 4.3, modify:
turbine_idx = 0  # Front turbine
# Run cell, then change to:
turbine_idx = 2  # Rear turbine
# Run again to compare
```

### Analyze Different Wind Conditions
```python
# In section 3.1, try:
ws_select = 8.0   # Low wind
# vs
ws_select = 11.0  # High wind
```

### Extract Specific Data
```python
# In a custom cell:
ds = datasets['RL+MPC']
data_slice = ds.sel(ws=9, wd=270, TI=0.05, method='nearest')
power = extract_1d_timeseries(data_slice.powerF_a)

# Now you can do whatever you want with this data:
print(f"Mean: {power.mean()/1e6:.2f} MW")
print(f"Std: {power.std()/1e6:.2f} MW")
print(f"Max: {power.max()/1e6:.2f} MW")
```

### Create Combined Plots
```python
# In section 7 custom analysis:
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Farm power
# Top right: Yaw angles
# Bottom left: Wind estimates
# Bottom right: Power increase

# ... your plotting code ...
```

## Available Data Variables

Each dataset typically contains:
- `powerF_a`: Total farm power
- `powerT_a`: Per-turbine power
- `yaw_a`: Turbine yaw angles
- `ws_a`: Wind speeds at turbines
- `estimated_wd`: Estimated wind direction
- `estimated_ws`: Estimated wind speed
- `estimated_ti`: Estimated turbulence intensity
- `pct_inc`: Percentage power increase vs baseline
- `reward`: Step rewards (for RL methods)

Check what's available:
```python
print(datasets['RL+MPC'].data_vars)
```

## Troubleshooting

### "NameError: name 'extract_1d_timeseries' is not defined"
- Run the setup cells (Section 1) first

### "KeyError: 'RL+MPC'"
- Check that your dataset loaded correctly
- Look at the "LOADING EVALUATION DATA" output
- Make sure the file path is correct

### Plots look strange
- Check the dimensions: `print(data_slice.powerF_a.dims)`
- Use `extract_1d_timeseries()` helper function
- Verify you selected the right wind condition

### Empty plots
- Check that the wind condition exists in your dataset
- Try using `method='nearest'` in `.sel()` calls

## Advanced: Multiple RL Seeds

To compare multiple RL training runs:

```python
# Load multiple seeds
eval_files = {
    'Greedy': 'evals/greedy_baseline.nc',
    'Oracle MPC': 'evals/mpc_oracle.nc',
    'RL Seed 1': 'evals/testrun_seed1.nc',
    'RL Seed 2': 'evals/testrun_seed2.nc',
    'RL Seed 3': 'evals/testrun_seed3.nc',
}

# In plotting cells, all seeds will appear automatically
# Add custom cell to compute mean/std:
rl_datasets = {k: v for k, v in datasets.items() if 'RL Seed' in k}
rl_powers = [ds.powerF_a.mean().values for ds in rl_datasets.values()]
print(f"RL Mean Power: {np.mean(rl_powers)/1e6:.2f} ± {np.std(rl_powers)/1e6:.2f} MW")
```

## Exporting Results

### Save Individual Plots
```python
# Add to any plotting cell:
plt.savefig('my_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('my_figure.pdf', bbox_inches='tight')  # Vector graphics
```

### Export Data to CSV
```python
# In custom cell:
ds = datasets['RL+MPC']
data_slice = ds.sel(ws=9, wd=270, TI=0.05, method='nearest')

# Create DataFrame
df = pd.DataFrame({
    'time': extract_1d_timeseries(data_slice.time),
    'power': extract_1d_timeseries(data_slice.powerF_a),
    'estimated_wd': extract_1d_timeseries(data_slice.estimated_wd),
})

# Save to CSV
df.to_csv('rl_mpc_timeseries.csv', index=False)
```

## Next Steps

1. **Explore**: Run through all cells to see what's available
2. **Modify**: Change parameters to focus on your research questions
3. **Extend**: Add custom analysis in Section 7
4. **Document**: Add markdown cells with your observations
5. **Share**: Export as HTML/PDF to share with collaborators

Happy analyzing! 🎉
