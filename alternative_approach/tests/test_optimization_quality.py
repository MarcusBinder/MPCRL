"""
Test the impact of optimization parameters on solution quality.

This script investigates the trade-off between computational cost and optimization quality
for different combinations of dt_opt, T_opt, and maxfun.

Goal: Find the Pareto frontier of speed vs quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import product
import pandas as pd
from py_wake.examples.data.hornsrev1 import V80
from mpcrl.mpc import WindFarmModel, optimize_farm_back2front, farm_energy, run_farm_delay_loop_optimized

def test_fixed_scenario(dt_opt, T_opt, maxfun, wind_conditions, initial_yaws,
                        eval_horizon=1000.0, seed=None, verbose=False):
    """
    Test optimization with fixed scenario and parameters.

    IMPORTANT: We separate two horizons:
    - T_opt: MPC optimization horizon (what we're testing - affects speed)
    - eval_horizon: Evaluation horizon (fixed at 1000s to capture full wake effects)

    Args:
        seed: Random seed for optimization. If None, uses a random seed.
              For reproducible tests, pass a specific seed.

    Returns:
        dict with: time, energy, yaw_angles, convergence_info
    """
    # Setup
    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])

    model = WindFarmModel(
        x_pos, y_pos, wt=V80(), D=80.0,
        U_inf=wind_conditions['ws'],
        TI=wind_conditions['ti'],
        wd=wind_conditions['wd'],
        cache_size=10000,
        cache_quant=0.25,
        wind_quant=0.25
    )

    # Sort initial yaws according to wind direction
    initial_yaws_sorted = initial_yaws[model.sorted_indices]

    # Run MPC optimization with T_opt horizon (this is what we're testing for speed)
    t_start = time.time()
    optimized_params = optimize_farm_back2front(
        model, initial_yaws_sorted,
        r_gamma=0.3,        # yaw rate
        t_AH=100.0,         # action horizon
        dt_opt=dt_opt,
        T_opt=T_opt,
        maxfun=maxfun,
        seed=seed,  # Use provided seed (or None for random)
        initial_params=None,
        use_time_shifted=True  # Use paper's best approach
    )
    t_opt = time.time() - t_start

    # Evaluate the optimized solution over LONG horizon to capture full wake effects
    t_eval, _, P_eval = run_farm_delay_loop_optimized(
        model, optimized_params, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0, dt=10.0, T=eval_horizon
    )

    total_energy = farm_energy(P_eval, t_eval)
    avg_power = total_energy / t_eval[-1]

    # Extract final yaw angles
    from mpcrl.mpc import yaw_traj
    final_yaws = []
    for i, params in enumerate(optimized_params):
        _, traj = yaw_traj(initial_yaws_sorted[i], params[0], params[1],
                          100.0, 0.3, 10.0, 100.0)
        final_yaws.append(traj[-1])
    final_yaws = np.array(final_yaws)

    if verbose:
        print(f"  dt_opt={dt_opt:>4}, T_opt={T_opt:>4}, maxfun={maxfun:>3} | "
              f"Time: {t_opt:>6.2f}s | Avg Power: {avg_power:>10,.0f} W | "
              f"Yaws: [{final_yaws[0]:>6.2f}, {final_yaws[1]:>6.2f}, {final_yaws[2]:>6.2f}]")

    return {
        'dt_opt': dt_opt,
        'T_opt': T_opt,
        'maxfun': maxfun,
        'time': t_opt,
        'total_energy': total_energy,
        'avg_power': avg_power,
        'final_yaws': final_yaws,
        'cache_hits': model.cache.hits,
        'cache_misses': model.cache.misses,
    }


def parameter_sweep(eval_horizon=1000.0, n_seeds=1, base_seed=None):
    """
    Comprehensive parameter sweep across dt_opt, T_opt, and maxfun.

    Args:
        eval_horizon: Evaluation time horizon in seconds
        n_seeds: Number of random seeds to average over per configuration
                 (1 = single seed, 3-5 = good statistical average, default=1 for speed)
        base_seed: Base seed for reproducibility. If None, uses random seeds.
                   If provided, uses base_seed, base_seed+1, base_seed+2, etc.
    """
    print("\n" + "="*80)
    print("OPTIMIZATION PARAMETER QUALITY SWEEP")
    print("="*80)

    # Fixed test scenario
    wind_conditions = {
        'ws': 8.0,
        'wd': 270.0,
        'ti': 0.06
    }
    initial_yaws = np.array([0.0, 0.0, 0.0])

    print(f"\nFixed scenario:")
    print(f"  Wind: WS={wind_conditions['ws']} m/s, WD={wind_conditions['wd']}°, TI={wind_conditions['ti']}")
    print(f"  Initial yaws: {initial_yaws}")
    print(f"  Farm layout: 3 turbines at [0, 500, 1000]m")
    print(f"  Evaluation horizon: {eval_horizon:.0f}s (ensures full wake propagation)")

    print(f"\nRandom seed strategy:")
    if base_seed is not None:
        print(f"  Using deterministic seeds starting from {base_seed}")
        print(f"  Seeds per config: {n_seeds}")
        if n_seeds > 1:
            print(f"  Will average over {n_seeds} runs per configuration")
    else:
        print(f"  Using random seeds (not reproducible)")
        if n_seeds > 1:
            print(f"  Will average over {n_seeds} runs per configuration")

    # Parameter ranges to test
    dt_opt_values = [10, 15, 20, 25, 30]
    T_opt_values = [200, 300, 400, 500]
    maxfun_values = [10, 15, 20, 30, 50]

    print(f"\nParameter ranges:")
    print(f"  dt_opt: {dt_opt_values}")
    print(f"  T_opt: {T_opt_values}")
    print(f"  maxfun: {maxfun_values}")
    print(f"  Total combinations: {len(dt_opt_values) * len(T_opt_values) * len(maxfun_values)}")
    print(f"  Total runs: {len(dt_opt_values) * len(T_opt_values) * len(maxfun_values) * n_seeds}")

    # Run sweep
    results = []
    print("\nRunning parameter sweep...")

    config_idx = 0
    for dt_opt, T_opt, maxfun in product(dt_opt_values, T_opt_values, maxfun_values):
        # Average over multiple seeds for this configuration
        config_results = []

        for seed_idx in range(n_seeds):
            if base_seed is not None:
                # Deterministic: use base_seed + unique offset for this config and seed
                seed = base_seed + config_idx * 1000 + seed_idx
            else:
                # Random
                seed = None

            result = test_fixed_scenario(dt_opt, T_opt, maxfun, wind_conditions, initial_yaws,
                                         eval_horizon=eval_horizon, seed=seed)
            config_results.append(result)

        # Average results across seeds
        avg_result = {
            'dt_opt': dt_opt,
            'T_opt': T_opt,
            'maxfun': maxfun,
            'time': np.mean([r['time'] for r in config_results]),
            'time_std': np.std([r['time'] for r in config_results]) if n_seeds > 1 else 0.0,
            'total_energy': np.mean([r['total_energy'] for r in config_results]),
            'avg_power': np.mean([r['avg_power'] for r in config_results]),
            'avg_power_std': np.std([r['avg_power'] for r in config_results]) if n_seeds > 1 else 0.0,
            'final_yaws': config_results[0]['final_yaws'],  # Just keep first seed's yaws
            'cache_hits': int(np.mean([r['cache_hits'] for r in config_results])),
            'cache_misses': int(np.mean([r['cache_misses'] for r in config_results])),
            'n_seeds': n_seeds
        }

        results.append(avg_result)
        config_idx += 1

        if config_idx % 10 == 0:
            print(f"  Progress: {config_idx}/{len(dt_opt_values) * len(T_opt_values) * len(maxfun_values)}")

    print(f"✓ Completed {len(results)} configurations ({len(results) * n_seeds} total runs)")

    return results, wind_conditions, initial_yaws


def baseline_comparison(wind_conditions, initial_yaws, eval_horizon=1000.0):
    """
    Compute baseline (no optimization) performance with delayed simulation.
    """
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80)

    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])

    model = WindFarmModel(
        x_pos, y_pos, wt=V80(), D=80.0,
        U_inf=wind_conditions['ws'],
        TI=wind_conditions['ti'],
        wd=wind_conditions['wd'],
        cache_size=5000
    )

    initial_yaws_sorted = initial_yaws[model.sorted_indices]

    # Greedy baseline: maintain 0° yaw using delayed simulation
    n_turbines = len(initial_yaws_sorted)
    greedy_params = np.array([[0.5, 0.5] for _ in range(n_turbines)])

    t_greedy, _, P_greedy = run_farm_delay_loop_optimized(
        model, greedy_params, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0, dt=10.0, T=eval_horizon
    )

    energy_greedy = farm_energy(P_greedy, t_greedy)
    baseline_power = energy_greedy / t_greedy[-1]

    print(f"Greedy baseline (hold yaws at 0°, evaluated over {eval_horizon:.0f}s):")
    print(f"  Average power: {baseline_power:,.0f} W")
    print(f"  Per turbine: {P_greedy.mean(axis=1)}")

    # Best case: high-quality optimization (reference)
    print(f"\nComputing reference solution with high-quality optimization...")
    print(f"  (averaging over 3 seeds for robust estimate)")

    ref_results = []
    for seed in [42, 43, 44]:
        result = test_fixed_scenario(
            dt_opt=10, T_opt=500, maxfun=100,
            wind_conditions=wind_conditions,
            initial_yaws=initial_yaws,
            eval_horizon=eval_horizon,
            seed=seed,
            verbose=False
        )
        ref_results.append(result)

    reference_power = np.mean([r['avg_power'] for r in ref_results])
    reference_std = np.std([r['avg_power'] for r in ref_results])

    print(f"  Reference solution power: {reference_power:,.0f} W (±{reference_std:,.0f} W)")
    print(f"  Improvement over greedy: {(reference_power - baseline_power) / baseline_power * 100:.1f}%")

    return baseline_power, reference_power


def analyze_results(results, baseline_power, reference_power):
    """
    Analyze sweep results and identify trade-offs.
    """
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    df = pd.DataFrame(results)

    # Normalize power relative to baseline and reference
    df['power_gain_vs_baseline_pct'] = (df['avg_power'] - baseline_power) / baseline_power * 100
    df['power_vs_reference_pct'] = df['avg_power'] / reference_power * 100

    # Computational cost metric (relative)
    reference_cost = df['time'].max()
    df['speedup'] = reference_cost / df['time']

    print("\nTop 10 configurations by power:")
    print(df.nlargest(10, 'avg_power')[['dt_opt', 'T_opt', 'maxfun', 'time', 'avg_power',
                                          'power_gain_vs_baseline_pct', 'speedup']])

    print("\nTop 10 fastest configurations:")
    print(df.nsmallest(10, 'time')[['dt_opt', 'T_opt', 'maxfun', 'time', 'avg_power',
                                      'power_gain_vs_baseline_pct', 'speedup']])

    # Find Pareto frontier (maximize power, minimize time)
    pareto_optimal = []
    for i, row in df.iterrows():
        is_pareto = True
        for j, other in df.iterrows():
            if i != j:
                # Check if other dominates this
                if (other['avg_power'] >= row['avg_power'] and other['time'] <= row['time'] and
                    (other['avg_power'] > row['avg_power'] or other['time'] < row['time'])):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_optimal.append(i)

    df['is_pareto'] = False
    df.loc[pareto_optimal, 'is_pareto'] = True

    print(f"\nPareto optimal configurations ({len(pareto_optimal)} found):")
    print(df[df['is_pareto']].sort_values('time')[['dt_opt', 'T_opt', 'maxfun', 'time',
                                                      'avg_power', 'power_vs_reference_pct', 'speedup']])

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Fast but good (within 1% of reference)
    good = df[df['power_vs_reference_pct'] >= 99.0]
    if len(good) > 0:
        fast_good = good.nsmallest(1, 'time').iloc[0]
        print(f"\n1. Best speed with minimal quality loss (<1%):")
        print(f"   dt_opt={fast_good['dt_opt']}, T_opt={fast_good['T_opt']}, maxfun={fast_good['maxfun']}")
        print(f"   Time: {fast_good['time']:.2f}s (speedup: {fast_good['speedup']:.1f}x)")
        print(f"   Power: {fast_good['avg_power']:,.0f} W ({fast_good['power_vs_reference_pct']:.2f}% of reference)")
    else:
        print(f"\n1. Best speed with minimal quality loss (<1%): No configs meet this threshold")

    # Balanced (within 2% of reference)
    good = df[df['power_vs_reference_pct'] >= 98.0]
    if len(good) > 0:
        balanced = good.nsmallest(1, 'time').iloc[0]
        print(f"\n2. Balanced speed/quality (<2% loss):")
        print(f"   dt_opt={balanced['dt_opt']}, T_opt={balanced['T_opt']}, maxfun={balanced['maxfun']}")
        print(f"   Time: {balanced['time']:.2f}s (speedup: {balanced['speedup']:.1f}x)")
        print(f"   Power: {balanced['avg_power']:,.0f} W ({balanced['power_vs_reference_pct']:.2f}% of reference)")
    else:
        print(f"\n2. Balanced speed/quality (<2% loss): No configs meet this threshold")

    # Very fast (from top 5 fastest)
    if len(df) >= 5:
        very_fast = df.nsmallest(5, 'time').nlargest(1, 'avg_power').iloc[0]
        print(f"\n3. Maximum speed (accept quality loss):")
        print(f"   dt_opt={very_fast['dt_opt']}, T_opt={very_fast['T_opt']}, maxfun={very_fast['maxfun']}")
        print(f"   Time: {very_fast['time']:.2f}s (speedup: {very_fast['speedup']:.1f}x)")
        print(f"   Power: {very_fast['avg_power']:,.0f} W ({very_fast['power_vs_reference_pct']:.2f}% of reference)")
    else:
        print(f"\n3. Maximum speed: Not enough configurations to analyze")

    return df


def visualize_results(df, baseline_power, reference_power):
    """
    Create comprehensive visualization of parameter sweep results.
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(16, 12))

    # 1. Pareto frontier: Time vs Power
    ax1 = plt.subplot(2, 3, 1)
    pareto = df[df['is_pareto']]
    non_pareto = df[~df['is_pareto']]

    ax1.scatter(non_pareto['time'], non_pareto['avg_power'],
               c='lightgray', alpha=0.5, s=30, label='Non-Pareto')
    ax1.scatter(pareto['time'], pareto['avg_power'],
               c='red', alpha=0.8, s=100, label='Pareto Optimal', marker='*')
    ax1.axhline(baseline_power, color='blue', linestyle='--', label='Baseline (no opt)')
    ax1.axhline(reference_power, color='green', linestyle='--', label='Reference (best)')
    ax1.set_xlabel('Optimization Time (s)')
    ax1.set_ylabel('Average Power (W)')
    ax1.set_title('Pareto Frontier: Speed vs Quality')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Impact of maxfun
    ax2 = plt.subplot(2, 3, 2)
    for maxfun in sorted(df['maxfun'].unique()):
        subset = df[(df['dt_opt'] == 20) & (df['T_opt'] == 300) & (df['maxfun'] == maxfun)]
        if len(subset) > 0:
            ax2.scatter(subset['time'], subset['avg_power'],
                       s=100, label=f'maxfun={maxfun}')
    ax2.axhline(reference_power, color='green', linestyle='--', alpha=0.5, label='Reference')
    ax2.set_xlabel('Optimization Time (s)')
    ax2.set_ylabel('Average Power (W)')
    ax2.set_title('Impact of maxfun\n(dt_opt=20, T_opt=300)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Impact of T_opt
    ax3 = plt.subplot(2, 3, 3)
    for T_opt in sorted(df['T_opt'].unique()):
        subset = df[(df['dt_opt'] == 20) & (df['T_opt'] == T_opt) & (df['maxfun'] == 20)]
        if len(subset) > 0:
            ax3.scatter(subset['time'], subset['avg_power'],
                       s=100, label=f'T_opt={T_opt}')
    ax3.axhline(reference_power, color='green', linestyle='--', alpha=0.5, label='Reference')
    ax3.set_xlabel('Optimization Time (s)')
    ax3.set_ylabel('Average Power (W)')
    ax3.set_title('Impact of T_opt\n(dt_opt=20, maxfun=20)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Impact of dt_opt
    ax4 = plt.subplot(2, 3, 4)
    for dt_opt in sorted(df['dt_opt'].unique()):
        subset = df[(df['dt_opt'] == dt_opt) & (df['T_opt'] == 300) & (df['maxfun'] == 20)]
        if len(subset) > 0:
            ax4.scatter(subset['time'], subset['avg_power'],
                       s=100, label=f'dt_opt={dt_opt}')
    ax4.axhline(reference_power, color='green', linestyle='--', alpha=0.5, label='Reference')
    ax4.set_xlabel('Optimization Time (s)')
    ax4.set_ylabel('Average Power (W)')
    ax4.set_title('Impact of dt_opt\n(T_opt=300, maxfun=20)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Heatmap: dt_opt vs T_opt (fixed maxfun=20)
    ax5 = plt.subplot(2, 3, 5)
    pivot = df[df['maxfun'] == 20].pivot_table(
        values='power_vs_reference_pct',
        index='T_opt',
        columns='dt_opt'
    )
    im = ax5.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=95, vmax=100)
    ax5.set_xticks(range(len(pivot.columns)))
    ax5.set_xticklabels(pivot.columns)
    ax5.set_yticks(range(len(pivot.index)))
    ax5.set_yticklabels(pivot.index)
    ax5.set_xlabel('dt_opt (s)')
    ax5.set_ylabel('T_opt (s)')
    ax5.set_title('Power Quality Heatmap\n(maxfun=20, % of reference)')
    plt.colorbar(im, ax=ax5, label='% of reference power')

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax5.text(j, i, f'{pivot.values[i, j]:.1f}',
                           ha="center", va="center", color="black", fontsize=8)

    # 6. Speedup vs Quality Trade-off
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(df['speedup'], df['power_vs_reference_pct'],
                         c=df['maxfun'], cmap='viridis', s=50, alpha=0.6)

    # Highlight Pareto points
    pareto_points = df[df['is_pareto']]
    ax6.scatter(pareto_points['speedup'], pareto_points['power_vs_reference_pct'],
               s=200, facecolors='none', edgecolors='red', linewidths=2,
               label='Pareto Optimal')

    ax6.axhline(99, color='orange', linestyle='--', alpha=0.5, label='99% threshold')
    ax6.axhline(98, color='red', linestyle='--', alpha=0.5, label='98% threshold')
    ax6.set_xlabel('Speedup (relative to slowest)')
    ax6.set_ylabel('Power Quality (% of reference)')
    ax6.set_title('Speedup vs Quality Trade-off')
    plt.colorbar(scatter, ax=ax6, label='maxfun')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/optimization_parameter_quality_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: tests/optimization_parameter_quality_analysis.png")
    plt.show()


if __name__ == "__main__":
    print("="*80)
    print("MPC OPTIMIZATION PARAMETER QUALITY INVESTIGATION")
    print("="*80)
    print("\nThis test systematically explores how dt_opt, T_opt, and maxfun")
    print("affect both computational cost and optimization quality.")
    print("\nObjective: Find the best speed/quality trade-off for RL training.\n")

    # Configuration
    EVAL_HORIZON = 1000.0  # seconds - must be long enough for wake effects

    # Seed strategy: Choose ONE of these options:

    # Option 1: Fast single-seed test (like original, but varies seeds across configs)
    # N_SEEDS = 1
    # BASE_SEED = 100  # Different from original seed=42 to avoid bias

    # Option 2: Robust multi-seed average (RECOMMENDED - takes 3x longer)
    N_SEEDS = 3  # Average over 3 seeds per configuration
    BASE_SEED = 100  # Start from seed 100

    # Option 3: Random seeds (not reproducible, but realistic)
    # N_SEEDS = 3
    # BASE_SEED = None

    print(f"\n{'='*80}")
    print(f"IMPORTANT: Using {N_SEEDS} seed(s) per configuration")
    if N_SEEDS == 1:
        print("  → Fast test, but results may be affected by random seed luck")
        print("  → For robust results, increase N_SEEDS to 3-5")
    else:
        print(f"  → Averaging over {N_SEEDS} seeds for statistical robustness")
        print(f"  → This will take ~{N_SEEDS}x longer than single-seed test")
    print(f"{'='*80}\n")

    # Run parameter sweep
    results, wind_conditions, initial_yaws = parameter_sweep(
        eval_horizon=EVAL_HORIZON,
        n_seeds=N_SEEDS,
        base_seed=BASE_SEED
    )

    # Compute baselines
    baseline_power, reference_power = baseline_comparison(
        wind_conditions, initial_yaws,
        eval_horizon=EVAL_HORIZON
    )

    # Analyze results
    df = analyze_results(results, baseline_power, reference_power)

    # Visualize
    visualize_results(df, baseline_power, reference_power)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Check the Pareto optimal configurations above")
    print("2. Review the visualization for detailed trade-offs")
    print("3. Choose parameters based on your speed/quality tolerance")
    print("4. For RL training, aim for 98-99% of reference quality with maximum speedup")
    print(f"\nNOTE: All tests evaluated over {EVAL_HORIZON:.0f}s to capture full wake propagation effects!")
    if N_SEEDS > 1:
        print(f"NOTE: Results averaged over {N_SEEDS} random seeds for statistical robustness")
