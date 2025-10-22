"""
MPC Hyperparameter Validation Framework
=======================================

Systematic testing and validation of MPC hyperparameters including:
- Action horizon (t_AH)
- Prediction horizon (T_opt)
- Optimization timestep (dt_opt)
- Optimizer budget (maxfun)
- Cache parameters

Uses simplified PyWake model for fast benchmarking.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
import time
from typing import Dict, List, Tuple
import json

# Import MPC components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from mpcrl.mpc import (
    WindFarmModel, optimize_farm_back2front,
    run_farm_delay_loop_optimized, farm_energy
)
from py_wake.examples.data.hornsrev1 import V80

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

FIGURE_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


class MPCBenchmark:
    """
    Simplified MPC benchmark environment using PyWake directly.
    Bypasses full WindGym for faster, deterministic testing.
    """

    def __init__(self, layout='line', n_turbines=4, spacing_D=10.0):
        """
        Initialize benchmark environment.

        Args:
            layout: 'line' or 'staggered'
            n_turbines: Number of turbines
            spacing_D: Spacing in rotor diameters
        """
        self.layout = layout
        self.n_turbines = n_turbines
        self.D = 80.0  # Rotor diameter
        self.spacing = spacing_D * self.D
        self.wt = V80()

        # Create layout
        if layout == 'line':
            self.x_pos = np.arange(n_turbines) * self.spacing
            self.y_pos = np.zeros(n_turbines)
        elif layout == 'staggered':
            # 2 rows, staggered pattern
            self.x_pos = []
            self.y_pos = []
            for i in range(n_turbines):
                row = i % 2
                col = i // 2
                self.x_pos.append(col * self.spacing + row * self.spacing / 2)
                self.y_pos.append(row * self.spacing * 0.5)
            self.x_pos = np.array(self.x_pos)
            self.y_pos = np.array(self.y_pos)
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def run_scenario(self, wd: float, ws: float, TI: float, mpc_params: Dict) -> Dict:
        """
        Run single MPC optimization scenario.

        Args:
            wd: Wind direction (deg)
            ws: Wind speed (m/s)
            TI: Turbulence intensity
            mpc_params: Dictionary with t_AH, T_opt, dt_opt, maxfun, etc.

        Returns:
            Dictionary with energy, time, trajectories, etc.
        """
        # Create model
        model = WindFarmModel(
            self.x_pos, self.y_pos, wt=self.wt,
            U_inf=ws, TI=TI, wd=wd,
            cache_size=mpc_params.get('cache_size', 64000),
            cache_quant=mpc_params.get('cache_quant', 0.25),
            wind_quant=mpc_params.get('wind_quant', 0.2)
        )

        # Initial conditions
        current_yaws = np.zeros(self.n_turbines)

        # Extract MPC parameters
        t_AH = mpc_params['t_AH']
        T_opt = mpc_params['T_opt']
        dt_opt = mpc_params['dt_opt']
        maxfun = mpc_params['maxfun']
        r_gamma = mpc_params.get('r_gamma', 0.3)
        use_time_shifted = mpc_params.get('use_time_shifted', False)  # Standard cost performs better
        seed = mpc_params.get('seed', 42)

        # Run optimization
        start_time = time.time()
        try:
            optimal_params = optimize_farm_back2front(
                model, current_yaws, r_gamma, t_AH, dt_opt, T_opt,
                maxfun=maxfun, seed=seed, use_time_shifted=use_time_shifted
            )

            # Generate trajectories
            t_opt, trajectories, P_opt = run_farm_delay_loop_optimized(
                model, optimal_params, current_yaws, r_gamma, t_AH, dt_opt, T_opt
            )

            # Calculate metrics
            energy = farm_energy(P_opt, t_opt) / 1e6  # MWh
            elapsed_time = time.time() - start_time

            # Cache statistics
            total_accesses = model.cache.hits + model.cache.misses
            hit_rate = (model.cache.hits / total_accesses * 100) if total_accesses > 0 else 0.0
            cache_stats = {
                'hit_rate': hit_rate,
                'hits': model.cache.hits,
                'misses': model.cache.misses
            }

            return {
                'success': True,
                'energy': energy,
                'time': elapsed_time,
                'trajectories': trajectories,
                'power': P_opt,
                'time_grid': t_opt,
                'parameters': optimal_params,
                'cache': cache_stats,
                'model': model
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time
            }

    def sweep_parameter(self,
                       param_name: str,
                       param_values: List,
                       base_params: Dict,
                       scenarios: List[Dict]) -> pd.DataFrame:
        """
        Sweep a single parameter across multiple scenarios.

        Args:
            param_name: Name of parameter to sweep
            param_values: List of values to test
            base_params: Base MPC parameters
            scenarios: List of wind scenarios (wd, ws, TI)

        Returns:
            DataFrame with results
        """
        results = []

        for scenario_idx, scenario in enumerate(scenarios):
            print(f"\n  Scenario {scenario_idx+1}/{len(scenarios)}: " +
                  f"wd={scenario['wd']}°, ws={scenario['ws']} m/s, TI={scenario['TI']}")

            for value in param_values:
                # Update parameter
                test_params = base_params.copy()
                test_params[param_name] = value

                # Run scenario
                result = self.run_scenario(**scenario, mpc_params=test_params)

                # Store results
                if result['success']:
                    results.append({
                        'scenario_idx': scenario_idx,
                        'wd': scenario['wd'],
                        'ws': scenario['ws'],
                        'TI': scenario['TI'],
                        'param_name': param_name,
                        'param_value': value,
                        'energy_MWh': result['energy'],
                        'time_s': result['time'],
                        'cache_hit_rate': result['cache']['hit_rate'],
                        'energy_per_time': result['energy'] / result['time']
                    })
                    print(f"    {param_name}={value:>6}: " +
                          f"Energy={result['energy']:.4f} MWh, " +
                          f"Time={result['time']:.2f}s")
                else:
                    print(f"    {param_name}={value:>6}: FAILED - {result.get('error', 'Unknown')}")

        return pd.DataFrame(results)


def test_action_horizon(benchmark: MPCBenchmark, scenarios: List[Dict]) -> pd.DataFrame:
    """Test different action horizon (t_AH) values."""
    print("\n" + "="*70)
    print("TEST 1: ACTION HORIZON (t_AH) SENSITIVITY")
    print("="*70)

    base_params = {
        't_AH': 100.0,  # Will be overridden
        'T_opt': 400.0,
        'dt_opt': 25.0,
        'maxfun': 20,
        'r_gamma': 0.3,
    }

    param_values = [50, 75, 100, 150, 200]

    results = benchmark.sweep_parameter('t_AH', param_values, base_params, scenarios)
    results.to_csv(DATA_DIR / 'test_t_AH.csv', index=False)

    return results


def test_prediction_horizon(benchmark: MPCBenchmark, scenarios: List[Dict]) -> pd.DataFrame:
    """Test different prediction horizon (T_opt) values."""
    print("\n" + "="*70)
    print("TEST 2: PREDICTION HORIZON (T_opt) SENSITIVITY")
    print("="*70)

    base_params = {
        't_AH': 100.0,
        'T_opt': 400.0,  # Will be overridden
        'dt_opt': 25.0,
        'maxfun': 20,
        'r_gamma': 0.3,
    }

    param_values = [200, 300, 400, 500, 600]

    results = benchmark.sweep_parameter('T_opt', param_values, base_params, scenarios)
    results.to_csv(DATA_DIR / 'test_T_opt.csv', index=False)

    return results


def test_optimization_timestep(benchmark: MPCBenchmark, scenarios: List[Dict]) -> pd.DataFrame:
    """Test different optimization timestep (dt_opt) values."""
    print("\n" + "="*70)
    print("TEST 3: OPTIMIZATION TIMESTEP (dt_opt) SENSITIVITY")
    print("="*70)

    base_params = {
        't_AH': 100.0,
        'T_opt': 400.0,
        'dt_opt': 25.0,  # Will be overridden
        'maxfun': 20,
        'r_gamma': 0.3,
    }

    param_values = [10, 15, 20, 25, 30, 40, 50]

    results = benchmark.sweep_parameter('dt_opt', param_values, base_params, scenarios)
    results.to_csv(DATA_DIR / 'test_dt_opt.csv', index=False)

    return results


def test_optimizer_budget(benchmark: MPCBenchmark, scenarios: List[Dict]) -> pd.DataFrame:
    """Test different optimizer budget (maxfun) values."""
    print("\n" + "="*70)
    print("TEST 4: OPTIMIZER BUDGET (maxfun) SENSITIVITY")
    print("="*70)

    base_params = {
        't_AH': 100.0,
        'T_opt': 400.0,
        'dt_opt': 25.0,
        'maxfun': 20,  # Will be overridden
        'r_gamma': 0.3,
    }

    param_values = [5, 10, 15, 20, 30, 50]

    results = benchmark.sweep_parameter('maxfun', param_values, base_params, scenarios)
    results.to_csv(DATA_DIR / 'test_maxfun.csv', index=False)

    return results


def test_cache_quantization(benchmark: MPCBenchmark, scenarios: List[Dict]) -> pd.DataFrame:
    """Test different cache quantization values."""
    print("\n" + "="*70)
    print("TEST 5: CACHE QUANTIZATION (cache_quant) SENSITIVITY")
    print("="*70)

    base_params = {
        't_AH': 100.0,
        'T_opt': 400.0,
        'dt_opt': 25.0,
        'maxfun': 20,
        'r_gamma': 0.3,
        'cache_quant': 0.25,  # Will be overridden
    }

    param_values = [0.1, 0.25, 0.5, 1.0, 2.0]

    results = benchmark.sweep_parameter('cache_quant', param_values, base_params, scenarios)
    results.to_csv(DATA_DIR / 'test_cache_quant.csv', index=False)

    return results


def plot_sensitivity_results(all_results: Dict[str, pd.DataFrame], save=True):
    """
    Generate comprehensive sensitivity plots for all parameters.
    """
    print("\nGenerating sensitivity plots...")

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('MPC Hyperparameter Sensitivity Analysis',
                 fontsize=16, fontweight='bold')

    plot_configs = [
        ('t_AH', 'Action Horizon (s)', 0, 0),
        ('T_opt', 'Prediction Horizon (s)', 0, 1),
        ('dt_opt', 'Optimization Timestep (s)', 0, 2),
        ('maxfun', 'Optimizer Budget (func evals)', 1, 0),
        ('cache_quant', 'Cache Quantization (deg)', 1, 1),
    ]

    for param_name, xlabel, row, col in plot_configs:
        if param_name not in all_results:
            continue

        df = all_results[param_name]

        # Plot 1: Energy vs parameter (top row)
        ax = fig.add_subplot(gs[row, col])

        # Group by parameter value, plot mean and std
        grouped = df.groupby('param_value').agg({
            'energy_MWh': ['mean', 'std'],
            'time_s': ['mean', 'std']
        }).reset_index()

        x = grouped['param_value']
        energy_mean = grouped['energy_MWh']['mean']
        energy_std = grouped['energy_MWh']['std']

        ax.plot(x, energy_mean, 'bo-', linewidth=2, markersize=8, label='Energy')
        if not energy_std.isna().all():
            ax.fill_between(x, energy_mean - energy_std, energy_mean + energy_std,
                           alpha=0.2, color='blue')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Energy (MWh)')
        ax.set_title(f'Energy vs {param_name}')
        ax.grid(True, alpha=0.3)

        # Twin axis for time
        ax2 = ax.twinx()
        time_mean = grouped['time_s']['mean']
        time_std = grouped['time_s']['std']
        ax2.plot(x, time_mean, 'ro--', linewidth=2, markersize=6, label='Time', alpha=0.7)
        ax2.set_ylabel('Computation Time (s)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

    # Bottom row: Pareto frontiers
    ax_pareto = fig.add_subplot(gs[2, :])

    colors = plt.cm.tab10(np.arange(len(all_results)))
    for idx, (param_name, df) in enumerate(all_results.items()):
        grouped = df.groupby('param_value').agg({
            'energy_MWh': 'mean',
            'time_s': 'mean'
        }).reset_index()

        ax_pareto.scatter(grouped['time_s'], grouped['energy_MWh'],
                         s=100, alpha=0.7, color=colors[idx], label=param_name,
                         edgecolors='black', linewidth=1)
        ax_pareto.plot(grouped['time_s'], grouped['energy_MWh'],
                      alpha=0.3, color=colors[idx])

    ax_pareto.set_xlabel('Computation Time (s)', fontsize=12)
    ax_pareto.set_ylabel('Energy (MWh)', fontsize=12)
    ax_pareto.set_title('Pareto Frontier: Energy vs Computation Time (All Parameters)', fontsize=13)
    ax_pareto.legend(loc='best')
    ax_pareto.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "hyperparameter_sensitivity.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "hyperparameter_sensitivity.pdf", bbox_inches='tight')
    print("  ✓ Saved sensitivity plots")

    return fig


def test_pareto_configurations(benchmark: MPCBenchmark, scenarios: List[Dict]) -> pd.DataFrame:
    """
    Test specific Pareto-optimal configurations.
    """
    print("\n" + "="*70)
    print("TEST 6: PARETO-OPTIMAL CONFIGURATIONS")
    print("="*70)

    configs = {
        'Ultra-Fast': {'t_AH': 100, 'T_opt': 200, 'dt_opt': 40, 'maxfun': 10, 'r_gamma': 0.3},
        'Fast': {'t_AH': 100, 'T_opt': 300, 'dt_opt': 20, 'maxfun': 15, 'r_gamma': 0.3},
        'Standard': {'t_AH': 100, 'T_opt': 400, 'dt_opt': 25, 'maxfun': 20, 'r_gamma': 0.3},
        'High-Quality': {'t_AH': 100, 'T_opt': 500, 'dt_opt': 15, 'maxfun': 30, 'r_gamma': 0.3},
        'Reference': {'t_AH': 100, 'T_opt': 600, 'dt_opt': 10, 'maxfun': 50, 'r_gamma': 0.3},
    }

    results = []
    for config_name, params in configs.items():
        print(f"\n  Testing {config_name} configuration...")
        print(f"    {params}")

        for scenario_idx, scenario in enumerate(scenarios):
            result = benchmark.run_scenario(**scenario, mpc_params=params)

            if result['success']:
                results.append({
                    'config': config_name,
                    'scenario_idx': scenario_idx,
                    'wd': scenario['wd'],
                    'ws': scenario['ws'],
                    'TI': scenario['TI'],
                    't_AH': params['t_AH'],
                    'T_opt': params['T_opt'],
                    'dt_opt': params['dt_opt'],
                    'maxfun': params['maxfun'],
                    'energy_MWh': result['energy'],
                    'time_s': result['time'],
                    'cache_hit_rate': result['cache']['hit_rate'],
                    'energy_per_time': result['energy'] / result['time']
                })
                print(f"    Scenario {scenario_idx+1}: " +
                      f"Energy={result['energy']:.4f} MWh, Time={result['time']:.2f}s")

    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / 'pareto_configurations.csv', index=False)

    return df


def plot_pareto_comparison(df: pd.DataFrame, save=True):
    """Plot comparison of Pareto configurations."""
    print("\nGenerating Pareto configuration comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Pareto Configuration Comparison',
                 fontsize=14, fontweight='bold')

    configs = df['config'].unique()
    colors = plt.cm.tab10(np.arange(len(configs)))

    # Plot 1: Energy comparison
    ax = axes[0]
    grouped = df.groupby('config')['energy_MWh'].agg(['mean', 'std']).reset_index()
    # Sort by mean energy
    grouped = grouped.sort_values('mean', ascending=False)

    x_pos = np.arange(len(grouped))
    ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], alpha=0.7, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped['config'], rotation=45, ha='right')
    ax.set_ylabel('Energy (MWh)')
    ax.set_title('Energy Production by Configuration')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Computation time comparison
    ax = axes[1]
    grouped = df.groupby('config')['time_s'].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values('mean')

    x_pos = np.arange(len(grouped))
    ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], alpha=0.7, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped['config'], rotation=45, ha='right')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Computation Time by Configuration')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Pareto frontier
    ax = axes[2]
    for idx, config in enumerate(configs):
        config_data = df[df['config'] == config]
        mean_time = config_data['time_s'].mean()
        mean_energy = config_data['energy_MWh'].mean()

        ax.scatter(mean_time, mean_energy, s=200, alpha=0.7,
                  color=colors[idx], label=config,
                  edgecolors='black', linewidth=2)

    ax.set_xlabel('Computation Time (s)')
    ax.set_ylabel('Energy (MWh)')
    ax.set_title('Pareto Frontier')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "pareto_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "pareto_comparison.pdf", bbox_inches='tight')
    print("  ✓ Saved Pareto comparison")

    return fig


def run_full_validation(layout='line'):
    """
    Run complete hyperparameter validation suite.
    """
    print("\n" + "="*70)
    print("MPC HYPERPARAMETER VALIDATION SUITE")
    print("="*70)
    print(f"Layout: {layout}")

    # Create benchmark
    benchmark = MPCBenchmark(layout=layout, n_turbines=4, spacing_D=10.0)

    # Define test scenarios
    scenarios = [
        {'wd': 270.0, 'ws': 8.0, 'TI': 0.06},  # Aligned flow, nominal
        {'wd': 240.0, 'ws': 8.0, 'TI': 0.06},  # Oblique flow
        {'wd': 270.0, 'ws': 10.0, 'TI': 0.06}, # Aligned, high wind
    ]

    # Run all tests
    all_results = {}

    all_results['t_AH'] = test_action_horizon(benchmark, scenarios)
    all_results['T_opt'] = test_prediction_horizon(benchmark, scenarios)
    all_results['dt_opt'] = test_optimization_timestep(benchmark, scenarios)
    all_results['maxfun'] = test_optimizer_budget(benchmark, scenarios)
    all_results['cache_quant'] = test_cache_quantization(benchmark, scenarios)

    # Generate sensitivity plots
    plot_sensitivity_results(all_results, save=True)

    # Test Pareto configurations
    pareto_results = test_pareto_configurations(benchmark, scenarios)
    plot_pareto_comparison(pareto_results, save=True)

    # Save summary
    summary = {
        'layout': layout,
        'n_turbines': benchmark.n_turbines,
        'scenarios': scenarios,
        'test_parameters': {k: df['param_value'].unique().tolist() for k, df in all_results.items()}
    }

    with open(DATA_DIR / 'validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {DATA_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")

    return all_results, pareto_results


if __name__ == "__main__":
    # Run for line layout
    results_line, pareto_line = run_full_validation(layout='line')

    plt.close('all')
    print("\n✓ All validation tests complete!")
