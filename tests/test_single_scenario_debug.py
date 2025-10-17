"""
Detailed diagnostic test for a single wake steering scenario.

Focus on WD=270°, WS=8 m/s, TI=0.05 with detailed breakdown.
"""

import numpy as np
import matplotlib.pyplot as plt
from py_wake.examples.data.hornsrev1 import V80
from mpcrl.mpc import WindFarmModel, optimize_farm_back2front, farm_energy, run_farm_delay_loop_optimized, yaw_traj

def detailed_single_scenario():
    """
    Test single scenario with maximum detail and diagnostics.
    """
    print("\n" + "="*80)
    print("DETAILED WAKE STEERING TEST - SINGLE SCENARIO")
    print("="*80)

    # Setup
    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])

    ws = 8.0
    wd = 270.0
    ti = 0.05  # Low TI → strong wakes → should benefit from steering

    print(f"\nConditions:")
    print(f"  Wind Speed: {ws} m/s")
    print(f"  Wind Direction: {wd}°")
    print(f"  Turbulence Intensity: {ti}")
    print(f"  Layout: 3 turbines at x=[0, 500, 1000]m, y=[0, 0, 0]m")

    model = WindFarmModel(
        x_pos, y_pos, wt=V80(), D=80.0,
        U_inf=ws, TI=ti, wd=wd,
        cache_size=5000, cache_quant=0.25, wind_quant=0.25
    )

    print(f"\n  Sorted order (upstream→downstream): {model.sorted_indices}")
    print(f"  Wake delays [s]:")
    max_delay = 0
    for i in range(len(x_pos)):
        for j in range(i):
            if model.delays[j, i] > 0:
                print(f"    Turbine {j} → {i}: {model.delays[j, i]:.1f}s")
                max_delay = max(max_delay, model.delays[j, i])

    print(f"\n  Max wake propagation delay: {max_delay:.1f}s")
    print(f"  → Need to evaluate for at least {max_delay * 1.5:.0f}s to see full wake steering effects!")

    initial_yaws = np.array([0.0, 0.0, 0.0])
    initial_yaws_sorted = initial_yaws[model.sorted_indices]

    # ========================================================================
    # GREEDY STRATEGY (0° yaw, no optimization)
    # ========================================================================
    print("\n" + "-"*80)
    print("GREEDY STRATEGY: Hold yaws at 0°")
    print("-"*80)

    # Params that produce zero yaw change
    n_turbines = len(initial_yaws_sorted)
    greedy_params = np.array([[0.5, 0.5] for _ in range(n_turbines)])

    # Evaluate over longer horizon to capture full wake propagation effects
    # At WS=8m/s, 500m spacing → 62.5s delay per hop
    # Need at least 150-200s to see full effects through all turbines
    eval_horizon = 200.0  # seconds

    t_greedy, traj_greedy, P_greedy = run_farm_delay_loop_optimized(
        model, greedy_params, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0, dt=10.0, T=eval_horizon
    )

    energy_greedy = farm_energy(P_greedy, t_greedy)
    avg_power_greedy = energy_greedy / t_greedy[-1]

    print(f"\nGreedy trajectories (should all be constant at 0°):")
    for i, traj in enumerate(traj_greedy):
        print(f"  Turbine {i}: [{traj[0]:.2f}, {traj[5]:.2f}, {traj[-1]:.2f}] (start, mid, end)")

    print(f"\nGreedy power per turbine (time-averaged over {eval_horizon:.0f}s):")
    for i in range(n_turbines):
        avg_turb = P_greedy[i, :].mean()
        print(f"  Turbine {i}: {avg_turb:>10,.0f} W")

    print(f"\nGreedy total:")
    print(f"  Total energy: {energy_greedy:>12,.0f} J")
    print(f"  Average power: {avg_power_greedy:>10,.0f} W")

    # ========================================================================
    # OPTIMIZED STRATEGY (wake steering)
    # ========================================================================
    print("\n" + "-"*80)
    print("OPTIMIZED STRATEGY: Wake steering with MPC")
    print("-"*80)

    print(f"\nRunning optimization...")
    print(f"  Parameters: dt_opt=20s, T_opt=300s, maxfun=20")

    optimized_params = optimize_farm_back2front(
        model, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0,
        dt_opt=20.0, T_opt=300.0, maxfun=20,
        seed=42, initial_params=None, use_time_shifted=True
    )

    print(f"\nOptimized trajectory parameters:")
    for i, params in enumerate(optimized_params):
        print(f"  Turbine {i}: o1={params[0]:.4f}, o2={params[1]:.4f}")

    # Generate and show trajectories
    print(f"\nOptimized yaw trajectories:")
    opt_trajectories = []
    for i, params in enumerate(optimized_params):
        t_traj, traj = yaw_traj(initial_yaws_sorted[i], params[0], params[1],
                                100.0, 0.3, 10.0, eval_horizon)
        opt_trajectories.append(traj)
        print(f"  Turbine {i}: [{traj[0]:.2f}°, {traj[5]:.2f}°, {traj[-1]:.2f}°] (start, mid, end)")

    # Evaluate optimized solution over same long horizon
    t_opt, traj_opt, P_opt = run_farm_delay_loop_optimized(
        model, optimized_params, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0, dt=10.0, T=eval_horizon
    )

    energy_opt = farm_energy(P_opt, t_opt)
    avg_power_opt = energy_opt / t_opt[-1]

    print(f"\nOptimized power per turbine (time-averaged over {eval_horizon:.0f}s):")
    for i in range(n_turbines):
        avg_turb_opt = P_opt[i, :].mean()
        avg_turb_greedy = P_greedy[i, :].mean()
        diff = avg_turb_opt - avg_turb_greedy
        diff_pct = (diff / avg_turb_greedy) * 100
        print(f"  Turbine {i}: {avg_turb_opt:>10,.0f} W (Δ {diff:>+8,.0f} W, {diff_pct:>+6.2f}%)")

    print(f"\nOptimized total:")
    print(f"  Total energy: {energy_opt:>12,.0f} J")
    print(f"  Average power: {avg_power_opt:>10,.0f} W")

    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    benefit = avg_power_opt - avg_power_greedy
    benefit_pct = (benefit / avg_power_greedy) * 100

    print(f"\nPower comparison:")
    print(f"  Greedy:     {avg_power_greedy:>10,.0f} W")
    print(f"  Optimized:  {avg_power_opt:>10,.0f} W")
    print(f"  Benefit:    {benefit:>10,.0f} W ({benefit_pct:>+6.2f}%)")

    if benefit > 0:
        print(f"\n✓ Wake steering provides benefit!")
    else:
        print(f"\n✗ Wake steering reduces power (greedy is better)")

    # ========================================================================
    # DETAILED VISUALIZATION
    # ========================================================================
    print("\n" + "-"*80)
    print("GENERATING DETAILED VISUALIZATION")
    print("-"*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Yaw trajectories
    ax = axes[0, 0]
    for i in range(n_turbines):
        ax.plot(t_greedy, traj_greedy[i], '--', label=f'Greedy T{i}', alpha=0.6)
        ax.plot(t_opt, traj_opt[i], '-', label=f'Optimized T{i}', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw Angle (°)')
    ax.set_title('Yaw Angle Trajectories')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Power time series per turbine
    ax = axes[0, 1]
    for i in range(n_turbines):
        ax.plot(t_greedy, P_greedy[i, :], '--', label=f'Greedy T{i}', alpha=0.6)
        ax.plot(t_opt, P_opt[i, :], '-', label=f'Optimized T{i}', linewidth=2)

    # Mark wake arrival times
    for i in range(1, n_turbines):
        delay = model.delays[0, i]
        ax.axvline(delay, color='red', linestyle=':', alpha=0.3)
        ax.text(delay, ax.get_ylim()[1]*0.9, f'Wake→T{i}',
               rotation=90, va='top', fontsize=7, color='red', alpha=0.7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Time Series (Per Turbine)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Total farm power
    ax = axes[1, 0]
    total_greedy = P_greedy.sum(axis=0)
    total_opt = P_opt.sum(axis=0)
    ax.plot(t_greedy, total_greedy, '--', label='Greedy', linewidth=2, alpha=0.7)
    ax.plot(t_opt, total_opt, '-', label='Optimized', linewidth=2)
    ax.axhline(avg_power_greedy, color='blue', linestyle=':', alpha=0.5, label=f'Greedy avg')
    ax.axhline(avg_power_opt, color='orange', linestyle=':', alpha=0.5, label=f'Opt avg')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Farm Power (W)')
    ax.set_title('Total Farm Power Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Power comparison bar chart
    ax = axes[1, 1]
    x = np.arange(n_turbines)
    width = 0.35

    greedy_avgs = [P_greedy[i, :].mean() for i in range(n_turbines)]
    opt_avgs = [P_opt[i, :].mean() for i in range(n_turbines)]

    bars1 = ax.bar(x - width/2, greedy_avgs, width, label='Greedy', alpha=0.7)
    bars2 = ax.bar(x + width/2, opt_avgs, width, label='Optimized', alpha=0.7)

    ax.set_xlabel('Turbine')
    ax.set_ylabel('Average Power (W)')
    ax.set_title('Average Power Per Turbine')
    ax.set_xticks(x)
    ax.set_xticklabels([f'T{i}' for i in range(n_turbines)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height/1000)}k',
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('tests/single_scenario_debug.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: tests/single_scenario_debug.png")
    plt.show()

    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print(f"\nPer-turbine changes:")
    for i in range(n_turbines):
        greedy_avg = P_greedy[i, :].mean()
        opt_avg = P_opt[i, :].mean()
        diff = opt_avg - greedy_avg
        diff_pct = (diff / greedy_avg) * 100

        yaw_change = traj_opt[i][-1] - traj_greedy[i][-1]

        print(f"  Turbine {i}:")
        print(f"    Yaw change: 0° → {yaw_change:+.2f}°")
        print(f"    Power change: {greedy_avg:,.0f} → {opt_avg:,.0f} W ({diff:+,.0f} W, {diff_pct:+.2f}%)")

    print(f"\nInterpretation:")
    if benefit > 0:
        print(f"  ✓ Wake steering helps! The upstream turbine(s) sacrifice some power")
        print(f"    by yawing, but the downstream turbines gain more from reduced wake effects.")
    else:
        print(f"  ✗ Wake steering doesn't help here. The power lost by yawing the upstream")
        print(f"    turbine(s) exceeds the gains from wake deflection for downstream turbines.")

    print(f"\n  This could mean:")
    print(f"    - Wake effects are already weak at these conditions")
    print(f"    - The layout/spacing doesn't benefit from wake steering")
    print(f"    - The MPC objective isn't aligned with instantaneous power")

    return {
        'greedy_power': avg_power_greedy,
        'optimized_power': avg_power_opt,
        'benefit': benefit,
        'benefit_pct': benefit_pct,
        'P_greedy': P_greedy,
        'P_opt': P_opt,
        't': t_opt,
        'traj_greedy': traj_greedy,
        'traj_opt': traj_opt
    }


if __name__ == "__main__":
    result = detailed_single_scenario()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nCheck the generated plots for detailed visualization of the comparison.")
