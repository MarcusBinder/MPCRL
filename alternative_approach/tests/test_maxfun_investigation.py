"""
Investigate why lower maxfun produces better results.

This script compares specific configurations to understand the optimization dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from py_wake.examples.data.hornsrev1 import V80
from mpcrl.mpc import WindFarmModel, optimize_farm_back2front, farm_energy, run_farm_delay_loop_optimized, yaw_traj


def test_with_multiple_seeds(dt_opt, T_opt, maxfun, n_seeds=5):
    """
    Test a configuration with multiple random seeds to see if results are consistent.
    """
    print(f"\nTesting dt_opt={dt_opt}, T_opt={T_opt}, maxfun={maxfun} with {n_seeds} seeds")
    print("="*70)

    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])
    wind_conditions = {'ws': 8.0, 'wd': 270.0, 'ti': 0.06}
    initial_yaws = np.array([0.0, 0.0, 0.0])
    eval_horizon = 1000.0

    results = []

    for seed in range(n_seeds):
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

        # Optimize
        optimized_params = optimize_farm_back2front(
            model, initial_yaws_sorted,
            r_gamma=0.3, t_AH=100.0,
            dt_opt=dt_opt, T_opt=T_opt, maxfun=maxfun,
            seed=seed,  # Different seed each time
            initial_params=None,
            use_time_shifted=True
        )

        # Evaluate
        t_eval, yaw_trajs, P_eval = run_farm_delay_loop_optimized(
            model, optimized_params, initial_yaws_sorted,
            r_gamma=0.3, t_AH=100.0, dt=10.0, T=eval_horizon
        )

        total_energy = farm_energy(P_eval, t_eval)
        avg_power = total_energy / t_eval[-1]

        # Get final yaws
        final_yaws = [traj[-1] for traj in yaw_trajs]

        results.append({
            'seed': seed,
            'avg_power': avg_power,
            'final_yaws': final_yaws,
            'params': optimized_params
        })

        print(f"  Seed {seed}: {avg_power:,.0f} W | Yaws: {[f'{y:>6.2f}' for y in final_yaws]}")

    powers = [r['avg_power'] for r in results]
    print(f"\nStatistics:")
    print(f"  Mean:   {np.mean(powers):,.0f} W")
    print(f"  Std:    {np.std(powers):,.0f} W")
    print(f"  Min:    {np.min(powers):,.0f} W")
    print(f"  Max:    {np.max(powers):,.0f} W")
    print(f"  Range:  {np.max(powers) - np.min(powers):,.0f} W ({(np.max(powers) - np.min(powers))/np.mean(powers)*100:.2f}%)")

    return results


def compare_trajectories(config_low_maxfun, config_high_maxfun):
    """
    Compare the actual yaw trajectories produced by low vs high maxfun.
    """
    print("\n" + "="*80)
    print("TRAJECTORY COMPARISON: Low maxfun vs High maxfun")
    print("="*80)

    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])
    wind_conditions = {'ws': 8.0, 'wd': 270.0, 'ti': 0.06}
    initial_yaws = np.array([0.0, 0.0, 0.0])
    eval_horizon = 1000.0

    configs = [config_low_maxfun, config_high_maxfun]
    all_results = []

    for label, config in zip(['Low maxfun', 'High maxfun'], configs):
        dt_opt, T_opt, maxfun = config
        print(f"\n{label}: dt_opt={dt_opt}, T_opt={T_opt}, maxfun={maxfun}")

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

        # Optimize
        optimized_params = optimize_farm_back2front(
            model, initial_yaws_sorted,
            r_gamma=0.3, t_AH=100.0,
            dt_opt=dt_opt, T_opt=T_opt, maxfun=maxfun,
            seed=42,
            initial_params=None,
            use_time_shifted=True
        )

        # Evaluate
        t_eval, yaw_trajs, P_eval = run_farm_delay_loop_optimized(
            model, optimized_params, initial_yaws_sorted,
            r_gamma=0.3, t_AH=100.0, dt=10.0, T=eval_horizon
        )

        total_energy = farm_energy(P_eval, t_eval)
        avg_power = total_energy / t_eval[-1]

        print(f"  Average power: {avg_power:,.0f} W")
        print(f"  Optimization params:")
        for i, params in enumerate(optimized_params):
            print(f"    Turbine {i}: o1={params[0]:.4f}, o2={params[1]:.4f}")
        print(f"  Final yaws: {[f'{traj[-1]:>6.2f}°' for traj in yaw_trajs]}")
        print(f"  Per-turbine power: {[f'{P_eval[i,:].mean()/1000:.1f}kW' for i in range(3)]}")

        all_results.append({
            'label': label,
            'config': config,
            't': t_eval,
            'yaw_trajs': yaw_trajs,
            'P': P_eval,
            'avg_power': avg_power,
            'params': optimized_params
        })

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['blue', 'red']
    linestyles = ['-', '--']

    # Plot 1: Yaw trajectories
    ax = axes[0, 0]
    for res, color, ls in zip(all_results, colors, linestyles):
        for i in range(3):
            ax.plot(res['t'], res['yaw_trajs'][i], color=color, linestyle=ls,
                   label=f"{res['label']} T{i}" if i == 0 else "", alpha=0.7, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw Angle (°)')
    ax.set_title('Yaw Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Power time series per turbine
    ax = axes[0, 1]
    for res, color, ls in zip(all_results, colors, linestyles):
        for i in range(3):
            ax.plot(res['t'], res['P'][i, :], color=color, linestyle=ls,
                   alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Time Series (Per Turbine)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Total farm power
    ax = axes[1, 0]
    for res, color, ls in zip(all_results, colors, linestyles):
        total = res['P'].sum(axis=0)
        ax.plot(res['t'], total, color=color, linestyle=ls, linewidth=2,
               label=f"{res['label']}: {res['avg_power']:,.0f}W avg")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Farm Power (W)')
    ax.set_title('Total Farm Power')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Average power comparison
    ax = axes[1, 1]
    labels_short = [r['label'] for r in all_results]
    powers_avg = [r['avg_power'] for r in all_results]
    bars = ax.bar(labels_short, powers_avg, color=colors, alpha=0.7)
    ax.set_ylabel('Average Power (W)')
    ax.set_title('Average Power Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height):,}W',
               ha='center', va='bottom', fontsize=10)

    # Show difference
    diff = all_results[0]['avg_power'] - all_results[1]['avg_power']
    diff_pct = diff / all_results[1]['avg_power'] * 100
    ax.text(0.5, 0.95, f'Difference: {diff:+,.0f}W ({diff_pct:+.2f}%)',
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('tests/maxfun_investigation.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: tests/maxfun_investigation.png")
    plt.show()

    return all_results


if __name__ == "__main__":
    print("="*80)
    print("INVESTIGATION: Why does lower maxfun produce better results?")
    print("="*80)

    # Test 1: Check if results are consistent across different seeds
    print("\n" + "="*80)
    print("TEST 1: Consistency across random seeds")
    print("="*80)

    print("\n[Config 1] Best from test: dt_opt=30, T_opt=200, maxfun=10")
    results_best = test_with_multiple_seeds(dt_opt=30, T_opt=200, maxfun=10, n_seeds=5)

    print("\n[Config 2] Reference: dt_opt=10, T_opt=500, maxfun=100")
    results_ref = test_with_multiple_seeds(dt_opt=10, T_opt=500, maxfun=100, n_seeds=5)

    # Test 2: Compare trajectories
    print("\n" + "="*80)
    print("TEST 2: Trajectory comparison (seed=42)")
    print("="*80)

    traj_comparison = compare_trajectories(
        config_low_maxfun=(30, 200, 10),
        config_high_maxfun=(10, 500, 100)
    )

    # Test 3: Test if it's specifically about maxfun or about the combination
    print("\n" + "="*80)
    print("TEST 3: Isolate the effect of maxfun")
    print("="*80)

    print("\nKeeping dt_opt=30, T_opt=200 fixed, vary only maxfun:")
    for maxfun in [5, 10, 20, 50, 100]:
        results = test_with_multiple_seeds(dt_opt=30, T_opt=200, maxfun=maxfun, n_seeds=3)

    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
This investigation helps us understand whether:
1. Low maxfun consistently finds better solutions (across different seeds)
2. The combination of (dt_opt, T_opt, maxfun) matters, not just maxfun alone
3. Different configurations produce fundamentally different yaw strategies

Key insights:
- If results vary a lot across seeds: optimization is getting lucky/unlucky
- If low maxfun is consistently better: early stopping prevents local minima
- If trajectories look very different: discretization changes the problem fundamentally
    """)
