"""
Quick version of optimization quality test with fewer configurations.
Runs in ~1-2 minutes instead of 5-10 minutes.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import product
import pandas as pd
from py_wake.examples.data.hornsrev1 import V80
from mpcrl.mpc import WindFarmModel, optimize_farm_back2front, farm_energy, run_farm_delay_loop_optimized

def test_fixed_scenario(dt_opt, T_opt, maxfun, wind_conditions, initial_yaws,
                        eval_horizon=1000.0, verbose=False):
    """
    Test optimization with fixed scenario and parameters.

    IMPORTANT: We separate two horizons:
    - T_opt: MPC optimization horizon (what we're testing - affects speed)
    - eval_horizon: Evaluation horizon (fixed at 1000s to capture full wake effects)
    """
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

    initial_yaws_sorted = initial_yaws[model.sorted_indices]

    # Run MPC optimization with T_opt horizon (this is what we're testing for speed)
    t_start = time.time()
    optimized_params = optimize_farm_back2front(
        model, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0,
        dt_opt=dt_opt, T_opt=T_opt, maxfun=maxfun,
        seed=42, initial_params=None, use_time_shifted=True
    )
    t_opt = time.time() - t_start

    # Evaluate solution over LONG horizon to capture full wake effects
    # This is the same for all tests to ensure fair comparison
    t_eval, _, P_eval = run_farm_delay_loop_optimized(
        model, optimized_params, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0, dt=10.0, T=eval_horizon
    )

    total_energy = farm_energy(P_eval, t_eval)
    avg_power = total_energy / t_eval[-1]

    if verbose:
        print(f"  dt_opt={dt_opt:>4}, T_opt={T_opt:>4}, maxfun={maxfun:>3} | "
              f"Time: {t_opt:>6.2f}s | Avg Power: {avg_power:>10,.0f} W")

    return {
        'dt_opt': dt_opt, 'T_opt': T_opt, 'maxfun': maxfun,
        'time': t_opt, 'avg_power': avg_power
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUICK OPTIMIZATION QUALITY TEST")
    print("="*80)

    wind_conditions = {'ws': 8.0, 'wd': 270.0, 'ti': 0.06}
    initial_yaws = np.array([0.0, 0.0, 0.0])

    # Fixed evaluation horizon for ALL tests (must be long enough for wake effects)
    EVAL_HORIZON = 1000.0  # seconds

    print(f"\nScenario: WS={wind_conditions['ws']} m/s, WD={wind_conditions['wd']}°, TI={wind_conditions['ti']}")
    print(f"Evaluation horizon: {EVAL_HORIZON:.0f}s (ensures full wake propagation)")

    # Reduced parameter set (3 × 3 × 3 = 27 configs)
    dt_opt_values = [15, 20, 25]
    T_opt_values = [200, 300, 400]
    maxfun_values = [10, 15, 20]

    print(f"Testing {len(dt_opt_values) * len(T_opt_values) * len(maxfun_values)} configurations...")

    # Greedy baseline: maintain 0° yaw using delayed simulation
    print("\n[BASELINE] Greedy strategy (hold yaws at 0°)")
    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])
    model_baseline = WindFarmModel(
        x_pos, y_pos, wt=V80(), D=80.0,
        U_inf=wind_conditions['ws'], TI=wind_conditions['ti'], wd=wind_conditions['wd']
    )

    initial_yaws_sorted = initial_yaws[model_baseline.sorted_indices]

    # Create greedy params (o1=0.5, o2=0.5 → no yaw change)
    n_turbines = len(initial_yaws_sorted)
    greedy_params = np.array([[0.5, 0.5] for _ in range(n_turbines)])

    # Evaluate greedy with same long horizon
    t_greedy, _, P_greedy = run_farm_delay_loop_optimized(
        model_baseline, greedy_params, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0, dt=10.0, T=EVAL_HORIZON
    )

    energy_greedy = farm_energy(P_greedy, t_greedy)
    baseline_power = energy_greedy / t_greedy[-1]
    print(f"  Greedy power (averaged over {EVAL_HORIZON:.0f}s): {baseline_power:,.0f} W")

    # Reference (best quality)
    print("\n[REFERENCE] High-quality optimization")
    ref = test_fixed_scenario(10, 500, 50, wind_conditions, initial_yaws,
                             eval_horizon=EVAL_HORIZON, verbose=True)

    # Parameter sweep
    print("\n[SWEEP] Testing parameter combinations:")
    results = []
    for dt_opt, T_opt, maxfun in product(dt_opt_values, T_opt_values, maxfun_values):
        result = test_fixed_scenario(dt_opt, T_opt, maxfun, wind_conditions, initial_yaws,
                                     eval_horizon=EVAL_HORIZON)
        results.append(result)

    df = pd.DataFrame(results)
    df['power_pct_of_ref'] = df['avg_power'] / ref['avg_power'] * 100
    df['speedup'] = ref['time'] / df['time']
    df['power_gain_vs_baseline'] = (df['avg_power'] - baseline_power) / baseline_power * 100

    print("\n" + "="*80)
    print("TOP CONFIGURATIONS BY SPEED")
    print("="*80)
    print(df.nsmallest(5, 'time')[['dt_opt', 'T_opt', 'maxfun', 'time', 'speedup',
                                     'avg_power', 'power_pct_of_ref', 'power_gain_vs_baseline']])

    print("\n" + "="*80)
    print("TOP CONFIGURATIONS BY QUALITY")
    print("="*80)
    print(df.nlargest(5, 'avg_power')[['dt_opt', 'T_opt', 'maxfun', 'time', 'speedup',
                                         'avg_power', 'power_pct_of_ref', 'power_gain_vs_baseline']])

    # Best compromise
    print("\n" + "="*80)
    print("RECOMMENDED CONFIGURATIONS")
    print("="*80)

    # Within 1% of reference, fastest
    good = df[df['power_pct_of_ref'] >= 99.0]
    if len(good) > 0:
        best = good.nsmallest(1, 'time').iloc[0]
        print(f"\n✓ Minimal quality loss (<1%):")
        print(f"  dt_opt={best['dt_opt']}, T_opt={best['T_opt']}, maxfun={best['maxfun']}")
        print(f"  Time: {best['time']:.2f}s | Speedup: {best['speedup']:.1f}x | Quality: {best['power_pct_of_ref']:.2f}%")

    # Within 2% of reference, fastest
    good = df[df['power_pct_of_ref'] >= 98.0]
    if len(good) > 0:
        best = good.nsmallest(1, 'time').iloc[0]
        print(f"\n✓ Balanced (<2% loss):")
        print(f"  dt_opt={best['dt_opt']}, T_opt={best['T_opt']}, maxfun={best['maxfun']}")
        print(f"  Time: {best['time']:.2f}s | Speedup: {best['speedup']:.1f}x | Quality: {best['power_pct_of_ref']:.2f}%")

    # Fastest in top 5 by quality
    fast = df.nsmallest(5, 'time').nlargest(1, 'avg_power').iloc[0]
    print(f"\n✓ Maximum speed:")
    print(f"  dt_opt={fast['dt_opt']}, T_opt={fast['T_opt']}, maxfun={fast['maxfun']}")
    print(f"  Time: {fast['time']:.2f}s | Speedup: {fast['speedup']:.1f}x | Quality: {fast['power_pct_of_ref']:.2f}%")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Time vs Power with size = maxfun
    ax = axes[0]
    for maxfun in maxfun_values:
        subset = df[df['maxfun'] == maxfun]
        ax.scatter(subset['time'], subset['avg_power'], s=100, alpha=0.7, label=f'maxfun={maxfun}')
    ax.axhline(baseline_power, color='blue', linestyle='--', label='Baseline', alpha=0.5)
    ax.axhline(ref['avg_power'], color='green', linestyle='--', label='Reference', alpha=0.5)
    ax.set_xlabel('Optimization Time (s)')
    ax.set_ylabel('Average Power (W)')
    ax.set_title('Speed vs Quality Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Speedup vs Quality
    ax = axes[1]
    scatter = ax.scatter(df['speedup'], df['power_pct_of_ref'],
                        c=df['maxfun'], cmap='viridis', s=80, alpha=0.7)
    ax.axhline(99, color='orange', linestyle='--', alpha=0.5, label='99% threshold')
    ax.axhline(98, color='red', linestyle='--', alpha=0.5, label='98% threshold')
    ax.set_xlabel('Speedup vs Reference')
    ax.set_ylabel('Quality (% of reference)')
    ax.set_title('Speedup vs Quality')
    plt.colorbar(scatter, ax=ax, label='maxfun')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/optimization_quality_quick.png', dpi=150)
    print("\n✓ Saved: tests/optimization_quality_quick.png")
    plt.show()

    print("\n" + "="*80)
    print("For more detailed analysis, run: python tests/test_optimization_quality.py")
    print("="*80)
