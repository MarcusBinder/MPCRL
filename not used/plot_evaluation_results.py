"""
Comprehensive plotting script for MPC+RL evaluation results.

This script creates various plots to analyze and compare:
- Greedy baseline (no control)
- MPC with different estimation strategies
- RL+MPC approach (supports multiple seeds)
- Oracle MPC (upper bound)

Usage:
    python plot_evaluation_results.py --rl_models testrun7 testrun8 --output_dir plots/
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import List, Dict
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_evaluation_data(eval_files: Dict[str, str]) -> Dict[str, xr.Dataset]:
    """
    Load evaluation datasets.

    Args:
        eval_files: Dict mapping method names to file paths

    Returns:
        Dict of loaded datasets
    """
    datasets = {}
    for name, filepath in eval_files.items():
        try:
            datasets[name] = xr.open_dataset(filepath)
            print(f"✓ Loaded {name}: {filepath}")
        except FileNotFoundError:
            print(f"✗ Could not find {name}: {filepath}")

    return datasets


def extract_1d_timeseries(data_array: xr.DataArray) -> np.ndarray:
    """
    Extract 1D time series from a DataArray, properly handling extra dimensions.

    This function handles DataArrays that may have extra dimensions like
    turbbox, model_step, deterministic, etc., and extracts just the time series.

    Args:
        data_array: xarray DataArray with 'time' dimension and possibly others

    Returns:
        1D numpy array containing the time series
    """
    # Make a copy to avoid modifying the original
    result = data_array

    # Select first element of common extra dimensions if they exist
    extra_dims = ['turbbox', 'model_step', 'deterministic']
    for dim in extra_dims:
        if dim in result.dims:
            result = result.isel({dim: 0})

    # Squeeze to remove any size-1 dimensions
    result = result.squeeze()

    # Get values - should now be 1D
    values = result.values

    # If still not 1D, flatten it
    if values.ndim > 1:
        values = values.ravel()

    return values


def plot_time_series_comparison(datasets: Dict[str, xr.Dataset],
                                  ws: float, wd: float, ti: float,
                                  output_dir: Path):
    """
    Plot time series comparison of power output for all methods.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    colors = {
        'Greedy': '#d62728',  # Red
        'Sensor MPC': '#ff7f0e',  # Orange
        'Simple Est. MPC': '#bcbd22',  # Yellow-green
        'RL+MPC (mean)': '#2ca02c',  # Green
        'Oracle MPC': '#1f77b4',  # Blue
    }

    # Plot 1: Farm power
    ax = axes[0]
    for name, ds in datasets.items():
        if name.startswith('RL+MPC'):
            continue  # Handle separately for mean/std

        try:
            data_slice = ds.sel(ws=ws, wd=wd, TI=ti, method='nearest')
            power = extract_1d_timeseries(data_slice.powerF_a)
            time = extract_1d_timeseries(data_slice.time)

            display_name = name
            if name == 'mpc_oracle':
                display_name = 'Oracle MPC'
            elif name == 'mpc_front_turbine':
                display_name = 'Sensor MPC'
            elif name == 'mpc_simple_estimator':
                display_name = 'Simple Est. MPC'
            elif name == 'greedy':
                display_name = 'Greedy'

            ax.plot(time, power / 1e6, label=display_name,
                   color=colors.get(display_name, None), linewidth=1.5)
        except Exception as e:
            print(f"Warning: Could not plot {name}: {e}")
            pass

    # Handle RL+MPC with multiple seeds
    rl_datasets = {k: v for k, v in datasets.items() if k.startswith('RL+MPC')}
    if rl_datasets:
        rl_powers = []
        rl_time = None
        for name, ds in rl_datasets.items():
            try:
                # Select and extract 1D time series
                data_slice = ds.sel(ws=ws, wd=wd, TI=ti, method='nearest')
                power = extract_1d_timeseries(data_slice.powerF_a)
                rl_powers.append(power)

                if rl_time is None:
                    rl_time = extract_1d_timeseries(data_slice.time)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
                import traceback
                traceback.print_exc()

        if rl_powers:
            # Ensure all arrays have same length
            min_len = min(len(p) for p in rl_powers)
            rl_powers = [p[:min_len] for p in rl_powers]
            rl_time = rl_time[:min_len]

            rl_powers = np.array(rl_powers)

            if len(rl_powers) > 1:
                # Multiple seeds - show mean and std
                mean_power = np.mean(rl_powers, axis=0)
                std_power = np.std(rl_powers, axis=0)

                ax.plot(rl_time, mean_power / 1e6, label='RL+MPC (mean)',
                       color=colors['RL+MPC (mean)'], linewidth=2)
                ax.fill_between(rl_time,
                               (mean_power - std_power) / 1e6,
                               (mean_power + std_power) / 1e6,
                               color=colors['RL+MPC (mean)'], alpha=0.2)
            else:
                # Single model - just plot directly
                single_power = rl_powers[0]
                ax.plot(rl_time, single_power / 1e6, label='RL+MPC',
                       color=colors['RL+MPC (mean)'], linewidth=2)

    ax.set_ylabel('Farm Power (MW)')
    ax.set_title(f'Power Output Comparison (WS={ws} m/s, WD={wd}°, TI={ti})')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Percentage increase over greedy
    ax = axes[1]
    if 'greedy' in datasets:
        data_slice_greedy = datasets['greedy'].sel(ws=ws, wd=wd, TI=ti, method='nearest')
        greedy_power = extract_1d_timeseries(data_slice_greedy.powerF_a)

        for name, ds in datasets.items():
            if name == 'greedy' or name.startswith('RL+MPC'):
                continue

            try:
                data_slice = ds.sel(ws=ws, wd=wd, TI=ti, method='nearest')
                power = extract_1d_timeseries(data_slice.powerF_a)
                time = extract_1d_timeseries(data_slice.time)

                # Match length with greedy_power
                min_len = min(len(power), len(greedy_power))
                power = power[:min_len]
                time = time[:min_len]
                greedy_subset = greedy_power[:min_len]

                pct_inc = ((power - greedy_subset) / greedy_subset) * 100

                display_name = name.replace('mpc_', '').replace('_', ' ').title()
                if 'oracle' in name.lower():
                    display_name = 'Oracle MPC'
                elif 'front_turbine' in name.lower():
                    display_name = 'Sensor MPC'
                elif 'simple' in name.lower():
                    display_name = 'Simple Est. MPC'

                ax.plot(time, pct_inc, label=display_name, linewidth=1.5)
            except:
                pass

        # Handle RL+MPC
        if rl_datasets:
            rl_pct_incs = []
            for name, ds in rl_datasets.items():
                try:
                    data_slice = ds.sel(ws=ws, wd=wd, TI=ti, method='nearest')
                    power = extract_1d_timeseries(data_slice.powerF_a)
                    # Match length with greedy_power
                    min_len = min(len(power), len(greedy_power))
                    power = power[:min_len]
                    greedy_subset = greedy_power[:min_len]
                    pct_inc = ((power - greedy_subset) / greedy_subset) * 100
                    rl_pct_incs.append(pct_inc)
                except Exception as e:
                    print(f"Warning: Could not compute pct_inc for {name}: {e}")
                    pass

            if rl_pct_incs:
                # Ensure all arrays have same length
                min_len = min(len(p) for p in rl_pct_incs)
                rl_pct_incs = [p[:min_len] for p in rl_pct_incs]
                rl_time_subset = rl_time[:min_len] if rl_time is not None else None

                rl_pct_incs = np.array(rl_pct_incs)

                if len(rl_pct_incs) > 1:
                    # Multiple seeds - show mean and std
                    mean_pct = np.mean(rl_pct_incs, axis=0)
                    std_pct = np.std(rl_pct_incs, axis=0)

                    if rl_time_subset is not None:
                        ax.plot(rl_time_subset, mean_pct, label='RL+MPC (mean)',
                               color=colors['RL+MPC (mean)'], linewidth=2)
                        ax.fill_between(rl_time_subset, mean_pct - std_pct, mean_pct + std_pct,
                                       color=colors['RL+MPC (mean)'], alpha=0.2)
                else:
                    # Single model - just plot directly
                    single_pct = rl_pct_incs[0]
                    if rl_time_subset is not None:
                        ax.plot(rl_time_subset, single_pct, label='RL+MPC',
                               color=colors['RL+MPC (mean)'], linewidth=2)

    ax.set_ylabel('Power Increase (%)')
    ax.set_title('Power Increase vs Greedy Baseline')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Plot 3: Wind direction estimation (for methods that estimate)
    ax = axes[2]
    for name, ds in datasets.items():
        if 'estimated_wd' not in ds.data_vars or name == 'greedy':
            continue

        try:
            data_slice = ds.sel(ws=ws, wd=wd, TI=ti, method='nearest')
            estimated_wd = extract_1d_timeseries(data_slice.estimated_wd)
            time = extract_1d_timeseries(data_slice.time)

            display_name = name.replace('mpc_', '').replace('_', ' ').title()
            if 'rl+mpc' in name.lower():
                display_name = name  # Keep full name for RL seeds

            ax.plot(time, estimated_wd, label=display_name, linewidth=1.5, alpha=0.7)
        except Exception as e:
            print(f"Warning: Could not plot WD estimation for {name}: {e}")
            pass

    ax.axhline(y=wd, color='k', linestyle='--', linewidth=2, label='True WD', alpha=0.7)
    ax.set_ylabel('Wind Direction (°)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Wind Direction Estimation')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'time_series_ws{ws}_wd{wd}_ti{ti}.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved time series plot")


def plot_aggregated_performance(datasets: Dict[str, xr.Dataset], output_dir: Path):
    """
    Bar chart comparing mean performance across all methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collect statistics
    methods = []
    mean_power_inc = []
    std_power_inc = []
    mean_power = []
    std_power = []

    # Get greedy power for reference
    greedy_power = None
    if 'greedy' in datasets:
        greedy_power = datasets['greedy'].powerF_a.mean().values
        methods.append('Greedy\n(No Control)')
        mean_power.append(greedy_power / 1e6)
        std_power.append(0)
        mean_power_inc.append(0)
        std_power_inc.append(0)

    # Process each method
    method_order = ['mpc_front_turbine', 'mpc_simple_estimator', 'mpc_oracle']
    display_names = {
        'mpc_front_turbine': 'Sensor MPC',
        'mpc_simple_estimator': 'Simple Est.\nMPC',
        'mpc_oracle': 'Oracle MPC\n(Upper Bound)',
    }

    for method_key in method_order:
        if method_key in datasets:
            ds = datasets[method_key]
            power = ds.powerF_a.mean().values

            methods.append(display_names[method_key])
            mean_power.append(power / 1e6)
            std_power.append(0)

            if greedy_power is not None:
                pct_inc = ((power - greedy_power) / greedy_power) * 100
                mean_power_inc.append(pct_inc)
                std_power_inc.append(0)
            else:
                mean_power_inc.append(0)
                std_power_inc.append(0)

    # Handle RL+MPC with multiple seeds
    rl_datasets = {k: v for k, v in datasets.items() if k.startswith('RL+MPC')}
    if rl_datasets:
        rl_powers = []
        for name, ds in rl_datasets.items():
            rl_powers.append(ds.powerF_a.mean().values)

        rl_powers = np.array(rl_powers)
        mean_rl_power = np.mean(rl_powers)
        std_rl_power = np.std(rl_powers)

        methods.append('RL+MPC\n(Ours)')
        mean_power.append(mean_rl_power / 1e6)
        std_power.append(std_rl_power / 1e6)

        if greedy_power is not None:
            rl_pct_incs = ((rl_powers - greedy_power) / greedy_power) * 100
            mean_power_inc.append(np.mean(rl_pct_incs))
            std_power_inc.append(np.std(rl_pct_incs))
        else:
            mean_power_inc.append(0)
            std_power_inc.append(0)

    # Plot 1: Mean power output
    ax = axes[0]
    colors_list = ['#d62728', '#ff7f0e', '#bcbd22', '#1f77b4', '#2ca02c']
    x_pos = np.arange(len(methods))

    bars = ax.bar(x_pos, mean_power, yerr=std_power,
                  color=colors_list[:len(methods)], alpha=0.8, capsize=5)
    ax.set_ylabel('Mean Farm Power (MW)', fontsize=12)
    ax.set_title('Mean Power Output Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val, std_val) in enumerate(zip(bars, mean_power, std_power)):
        height = bar.get_height()
        label = f'{val:.2f}'
        if std_val > 0:
            label += f'\n±{std_val:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom', fontsize=9)

    # Plot 2: Power increase vs greedy
    ax = axes[1]
    bars = ax.bar(x_pos[1:], mean_power_inc[1:], yerr=std_power_inc[1:],
                  color=colors_list[1:len(methods)], alpha=0.8, capsize=5)
    ax.set_ylabel('Power Increase vs Greedy (%)', fontsize=12)
    ax.set_title('Performance Improvement', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos[1:])
    ax.set_xticklabels(methods[1:], fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Add value labels on bars
    for bar, val, std_val in zip(bars, mean_power_inc[1:], std_power_inc[1:]):
        height = bar.get_height()
        label = f'{val:.2f}%'
        if std_val > 0:
            label += f'\n±{std_val:.2f}%'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_performance.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved aggregated performance plot")


def plot_estimation_accuracy(datasets: Dict[str, xr.Dataset], output_dir: Path):
    """
    Plot wind condition estimation accuracy (RMSE and bias).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    metrics = {
        'wd': ('Wind Direction', '°'),
        'ws': ('Wind Speed', 'm/s'),
        'ti': ('Turbulence Intensity', ''),
    }

    methods_to_plot = []
    colors_dict = {
        'Sensor MPC': '#ff7f0e',
        'Simple Est. MPC': '#bcbd22',
        'RL+MPC': '#2ca02c',
        'Oracle MPC': '#1f77b4',
    }

    for idx, (metric, (label, unit)) in enumerate(metrics.items()):
        ax_rmse = axes[0, idx]
        ax_bias = axes[1, idx]

        method_names = []
        rmse_values = []
        bias_values = []
        rmse_stds = []
        bias_stds = []

        # Get true values from coordinates
        for name, ds in datasets.items():
            if 'estimated_' + metric not in ds.data_vars or name == 'greedy':
                continue

            display_name = name.replace('mpc_', '').replace('_', ' ').title()
            if 'oracle' in name.lower():
                display_name = 'Oracle MPC'
            elif 'front_turbine' in name.lower():
                display_name = 'Sensor MPC'
            elif 'simple' in name.lower():
                display_name = 'Simple Est. MPC'
            elif name.startswith('RL+MPC'):
                continue  # Handle separately

            try:
                estimated = ds['estimated_' + metric].values.flatten()

                # Get true value from coordinate
                if metric == 'wd':
                    true_vals = np.tile(ds.wd.values, len(ds.time))
                elif metric == 'ws':
                    true_vals = np.tile(ds.ws.values, len(ds.time))
                elif metric == 'ti':
                    true_vals = np.tile(ds.TI.values, len(ds.time))

                # Remove NaN values
                mask = ~np.isnan(estimated)
                estimated = estimated[mask]
                true_vals = np.repeat(true_vals, sum(mask) // len(true_vals))[:len(estimated)]

                errors = estimated - true_vals
                rmse = np.sqrt(np.mean(errors**2))
                bias = np.mean(errors)

                method_names.append(display_name)
                rmse_values.append(rmse)
                bias_values.append(bias)
                rmse_stds.append(0)
                bias_stds.append(0)
            except Exception as e:
                print(f"Warning: Could not process {name} for {metric}: {e}")
                pass

        # Handle RL+MPC with multiple seeds
        rl_datasets = {k: v for k, v in datasets.items() if k.startswith('RL+MPC')}
        if rl_datasets:
            rl_rmses = []
            rl_biases = []

            for name, ds in rl_datasets.items():
                try:
                    estimated = ds['estimated_' + metric].values.flatten()

                    if metric == 'wd':
                        true_vals = np.tile(ds.wd.values, len(ds.time))
                    elif metric == 'ws':
                        true_vals = np.tile(ds.ws.values, len(ds.time))
                    elif metric == 'ti':
                        true_vals = np.tile(ds.TI.values, len(ds.time))

                    mask = ~np.isnan(estimated)
                    estimated = estimated[mask]
                    true_vals = np.repeat(true_vals, sum(mask) // len(true_vals))[:len(estimated)]

                    errors = estimated - true_vals
                    rl_rmses.append(np.sqrt(np.mean(errors**2)))
                    rl_biases.append(np.mean(errors))
                except:
                    pass

            if rl_rmses:
                method_names.append('RL+MPC')
                rmse_values.append(np.mean(rl_rmses))
                bias_values.append(np.mean(rl_biases))
                rmse_stds.append(np.std(rl_rmses))
                bias_stds.append(np.std(rl_biases))

        # Plot RMSE
        x_pos = np.arange(len(method_names))
        colors = [colors_dict.get(m, '#999999') for m in method_names]

        bars = ax_rmse.bar(x_pos, rmse_values, yerr=rmse_stds,
                          color=colors, alpha=0.8, capsize=5)
        ax_rmse.set_ylabel(f'RMSE ({unit})' if unit else 'RMSE', fontsize=11)
        ax_rmse.set_title(f'{label} Estimation Error', fontsize=12, fontweight='bold')
        ax_rmse.set_xticks(x_pos)
        ax_rmse.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
        ax_rmse.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val, std_val in zip(bars, rmse_values, rmse_stds):
            height = bar.get_height()
            label_text = f'{val:.3f}'
            if std_val > 0:
                label_text += f'\n±{std_val:.3f}'
            ax_rmse.text(bar.get_x() + bar.get_width()/2., height,
                        label_text, ha='center', va='bottom', fontsize=8)

        # Plot Bias
        bars = ax_bias.bar(x_pos, bias_values, yerr=bias_stds,
                          color=colors, alpha=0.8, capsize=5)
        ax_bias.set_ylabel(f'Bias ({unit})' if unit else 'Bias', fontsize=11)
        ax_bias.set_title(f'{label} Estimation Bias', fontsize=12, fontweight='bold')
        ax_bias.set_xticks(x_pos)
        ax_bias.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
        ax_bias.grid(axis='y', alpha=0.3)
        ax_bias.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Add value labels
        for bar, val, std_val in zip(bars, bias_values, bias_stds):
            height = bar.get_height()
            label_text = f'{val:.3f}'
            if std_val > 0:
                label_text += f'\n±{std_val:.3f}'
            y_pos = height if height > 0 else height
            va = 'bottom' if height > 0 else 'top'
            ax_bias.text(bar.get_x() + bar.get_width()/2., y_pos,
                        label_text, ha='center', va=va, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'estimation_accuracy.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved estimation accuracy plot")


def plot_performance_heatmap(datasets: Dict[str, xr.Dataset], output_dir: Path):
    """
    Heatmap showing performance across different wind conditions.
    """
    # Check if we have multiple wind conditions
    if len(datasets) == 0:
        return

    first_ds = list(datasets.values())[0]
    n_ws = len(first_ds.ws)
    n_wd = len(first_ds.wd)

    if n_ws <= 1 and n_wd <= 1:
        print("⚠ Skipping heatmap - only single wind condition evaluated")
        return

    # Create subplots for each method
    methods_to_plot = ['mpc_front_turbine', 'RL+MPC', 'mpc_oracle']
    fig, axes = plt.subplots(1, len(methods_to_plot), figsize=(15, 5))

    if len(methods_to_plot) == 1:
        axes = [axes]

    for idx, method_key in enumerate(methods_to_plot):
        ax = axes[idx]
        im = None  # Initialize to avoid UnboundLocalError

        if method_key == 'RL+MPC':
            # Average over RL seeds
            rl_datasets = {k: v for k, v in datasets.items() if k.startswith('RL+MPC')}
            if not rl_datasets:
                continue

            # Get pct_inc for all seeds and average
            pct_incs = []
            for name, ds in rl_datasets.items():
                if 'pct_inc' in ds.data_vars:
                    pct_incs.append(ds.pct_inc.mean(dim='time').values.squeeze())

            if pct_incs:
                mean_pct_inc = np.mean(pct_incs, axis=0)
                im = ax.imshow(mean_pct_inc, cmap='RdYlGn', aspect='auto')
                ax.set_title('RL+MPC (Ours)', fontsize=13, fontweight='bold')
        else:
            if method_key not in datasets:
                continue

            ds = datasets[method_key]
            if 'pct_inc' in ds.data_vars:
                pct_inc_map = ds.pct_inc.mean(dim='time').values.squeeze()
                im = ax.imshow(pct_inc_map, cmap='RdYlGn', aspect='auto')

                title = method_key.replace('mpc_', '').replace('_', ' ').title()
                if 'oracle' in method_key.lower():
                    title = 'Oracle MPC (Upper Bound)'
                elif 'front_turbine' in method_key.lower():
                    title = 'Sensor MPC'

                ax.set_title(title, fontsize=13, fontweight='bold')

        # Only add labels and colorbar if we created an image
        if im is not None:
            # Set axis labels
            ax.set_xlabel('Wind Direction (°)', fontsize=11)
            ax.set_ylabel('Wind Speed (m/s)', fontsize=11)
            ax.set_xticks(np.arange(len(first_ds.wd)))
            ax.set_xticklabels([f'{wd:.0f}' for wd in first_ds.wd.values])
            ax.set_yticks(np.arange(len(first_ds.ws)))
            ax.set_yticklabels([f'{ws:.0f}' for ws in first_ds.ws.values])

            plt.colorbar(im, ax=ax, label='Power Increase (%)')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved performance heatmap")


def plot_learning_curve(datasets: Dict[str, xr.Dataset], output_dir: Path):
    """
    Plot RL agent performance vs training steps (learning curve).
    Shows how the RL agent improves during training.
    """
    # Get RL datasets
    rl_datasets = {k: v for k, v in datasets.items() if k.startswith('RL+MPC')}

    if not rl_datasets:
        print("⚠ Skipping learning curve - no RL+MPC datasets found")
        return

    # Check if any RL dataset has multiple model_steps
    has_multiple_steps = False
    for name, ds in rl_datasets.items():
        if 'model_step' in ds.dims and len(ds.model_step) > 1:
            has_multiple_steps = True
            break

    if not has_multiple_steps:
        print("⚠ Skipping learning curve - RL datasets only have single model_step")
        print("  To see learning curves, evaluate at multiple training checkpoints")
        return

    # Get greedy baseline
    greedy_power = None
    if 'greedy' in datasets:
        greedy_power = datasets['greedy'].powerF_a.mean(dim='time').values
    else:
        print("⚠ Warning: No greedy baseline found for learning curve")
        return

    # Collect performance data across training steps
    all_model_steps = set()
    performance_by_seed = {}

    for name, ds in rl_datasets.items():
        if 'model_step' not in ds.dims or len(ds.model_step) <= 1:
            continue

        model_steps = ds.model_step.values
        all_model_steps.update(model_steps)

        perf_at_steps = []
        for step in model_steps:
            ds_step = ds.sel(model_step=step)
            step_power = ds_step.powerF_a.mean(dim='time').values
            pct_inc = ((step_power - greedy_power) / greedy_power) * 100
            perf_at_steps.append((step, np.mean(pct_inc)))

        performance_by_seed[name] = perf_at_steps

    if not performance_by_seed:
        print("⚠ Skipping learning curve - no valid data")
        return

    # Sort model steps
    model_steps_sorted = sorted(all_model_steps)

    # Aggregate performance across seeds at each step
    mean_performance = []
    std_performance = []

    for step in model_steps_sorted:
        perfs_at_step = []
        for seed_name, seed_data in performance_by_seed.items():
            # Find performance at this step
            for s, p in seed_data:
                if s == step:
                    perfs_at_step.append(p)
                    break

        if perfs_at_step:
            mean_performance.append(np.mean(perfs_at_step))
            std_performance.append(np.std(perfs_at_step))
        else:
            mean_performance.append(np.nan)
            std_performance.append(np.nan)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot RL learning curve
    ax.plot(model_steps_sorted, mean_performance, 'o-', color='#2ca02c',
            linewidth=2.5, markersize=10, label='RL+MPC', zorder=3)

    # Add error bars/shaded region if we have multiple seeds
    if len(performance_by_seed) > 1:
        ax.fill_between(model_steps_sorted,
                        np.array(mean_performance) - np.array(std_performance),
                        np.array(mean_performance) + np.array(std_performance),
                        alpha=0.2, color='#2ca02c')

    # Add baseline reference lines
    if 'mpc_oracle' in datasets:
        oracle_power = datasets['mpc_oracle'].powerF_a.mean(dim='time').values
        oracle_pct_inc = np.mean(((oracle_power - greedy_power) / greedy_power) * 100)
        ax.axhline(y=oracle_pct_inc, color='#1f77b4', linestyle='--',
                  linewidth=2, label='Oracle MPC (upper bound)', alpha=0.7, zorder=1)

    if 'mpc_front_turbine' in datasets:
        sensor_power = datasets['mpc_front_turbine'].powerF_a.mean(dim='time').values
        sensor_pct_inc = np.mean(((sensor_power - greedy_power) / greedy_power) * 100)
        ax.axhline(y=sensor_pct_inc, color='#ff7f0e', linestyle='--',
                  linewidth=2, label='Sensor MPC baseline', alpha=0.7, zorder=1)

    if 'mpc_simple_estimator' in datasets:
        simple_power = datasets['mpc_simple_estimator'].powerF_a.mean(dim='time').values
        simple_pct_inc = np.mean(((simple_power - greedy_power) / greedy_power) * 100)
        ax.axhline(y=simple_pct_inc, color='#bcbd22', linestyle='--',
                  linewidth=2, label='Simple Estimator MPC', alpha=0.7, zorder=1)

    # Add value labels on points
    for step, pct, std in zip(model_steps_sorted, mean_performance, std_performance):
        if not np.isnan(pct):
            if len(performance_by_seed) > 1 and not np.isnan(std):
                label = f'{pct:.1f}%\n±{std:.1f}%'
            else:
                label = f'{pct:.1f}%'
            ax.annotate(label, (step, pct),
                       textcoords="offset points", xytext=(0,10),
                       ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Power Increase vs Greedy (%)', fontsize=13, fontweight='bold')
    ax.set_title('RL Agent Learning Progress: Performance vs Training Steps',
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(0, min(mean_performance)-5))

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curve.png', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved learning curve plot")

    # Print summary
    print("\n" + "="*70)
    print("LEARNING PROGRESS SUMMARY")
    print("="*70)
    for step, pct, std in zip(model_steps_sorted, mean_performance, std_performance):
        if not np.isnan(pct):
            if len(performance_by_seed) > 1 and not np.isnan(std):
                print(f"  After {step:6d} steps: {pct:6.2f}% ± {std:.2f}% power increase")
            else:
                print(f"  After {step:6d} steps: {pct:6.2f}% power increase")
    print("="*70)


def create_summary_table(datasets: Dict[str, xr.Dataset], output_dir: Path):
    """
    Create a summary table of performance metrics.
    """
    rows = []

    # Get greedy power for reference
    greedy_power = None
    if 'greedy' in datasets:
        greedy_power = datasets['greedy'].powerF_a.mean().values
        rows.append({
            'Method': 'Greedy (No Control)',
            'Mean Power (MW)': f'{greedy_power/1e6:.2f}',
            'Power Increase (%)': '0.00',
            'Std Dev (%)': '-',
        })

    # Process other methods
    method_order = ['mpc_front_turbine', 'mpc_simple_estimator', 'mpc_oracle']
    display_names = {
        'mpc_front_turbine': 'Sensor MPC',
        'mpc_simple_estimator': 'Simple Estimator MPC',
        'mpc_oracle': 'Oracle MPC (Upper Bound)',
    }

    for method_key in method_order:
        if method_key in datasets:
            ds = datasets[method_key]
            power = ds.powerF_a.mean().values
            pct_inc = ((power - greedy_power) / greedy_power) * 100 if greedy_power else 0

            rows.append({
                'Method': display_names[method_key],
                'Mean Power (MW)': f'{power/1e6:.2f}',
                'Power Increase (%)': f'{pct_inc:.2f}',
                'Std Dev (%)': '-',
            })

    # Handle RL+MPC with multiple seeds
    rl_datasets = {k: v for k, v in datasets.items() if k.startswith('RL+MPC')}
    if rl_datasets:
        rl_powers = []
        for name, ds in rl_datasets.items():
            rl_powers.append(ds.powerF_a.mean().values)

        rl_powers = np.array(rl_powers)
        mean_power = np.mean(rl_powers)
        std_power = np.std(rl_powers)

        pct_incs = ((rl_powers - greedy_power) / greedy_power) * 100 if greedy_power else np.zeros_like(rl_powers)
        mean_pct_inc = np.mean(pct_incs)
        std_pct_inc = np.std(pct_incs)

        rows.append({
            'Method': f'RL+MPC (Ours, n={len(rl_datasets)})',
            'Mean Power (MW)': f'{mean_power/1e6:.2f} ± {std_power/1e6:.2f}',
            'Power Increase (%)': f'{mean_pct_inc:.2f}',
            'Std Dev (%)': f'{std_pct_inc:.2f}' if len(rl_datasets) > 1 else '-',
        })

    # Create DataFrame and save
    df = pd.DataFrame(rows)

    # Save as CSV
    df.to_csv(output_dir / 'performance_summary.csv', index=False)

    # Save as formatted text
    with open(output_dir / 'performance_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Saved performance summary table")
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Plot MPC+RL evaluation results')
    parser.add_argument('--rl_models', nargs='+', default=[],
                       help='List of RL model names (e.g., testrun7 testrun8)')
    parser.add_argument('--output_dir', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--ws', type=float, default=9.0,
                       help='Wind speed for time series plots')
    parser.add_argument('--wd', type=float, default=270.0,
                       help='Wind direction for time series plots')
    parser.add_argument('--ti', type=float, default=0.05,
                       help='Turbulence intensity for time series plots')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("LOADING EVALUATION DATA")
    print("="*80)

    # Define evaluation files
    eval_files = {
        'greedy': 'evals/greedy_baseline.nc',
        'mpc_front_turbine': 'evals/mpc_front_turbine.nc',
        'mpc_simple_estimator': 'evals/mpc_simple_estimator.nc',
        'mpc_oracle': 'evals/mpc_oracle.nc',
    }

    # Add RL models
    for model_name in args.rl_models:
        eval_files[f'RL+MPC_{model_name}'] = f'evals/{model_name}.nc'

    # Load datasets
    datasets = load_evaluation_data(eval_files)

    if len(datasets) == 0:
        print("\n✗ No evaluation data found! Please run evaluations first.")
        return

    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80 + "\n")

    # Generate all plots
    plot_time_series_comparison(datasets, args.ws, args.wd, args.ti, output_dir)
    plot_aggregated_performance(datasets, output_dir)
    plot_learning_curve(datasets, output_dir)
    plot_estimation_accuracy(datasets, output_dir)
    plot_performance_heatmap(datasets, output_dir)
    create_summary_table(datasets, output_dir)

    print(f"\n{'='*80}")
    print(f"✓ All plots saved to: {output_dir.absolute()}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
