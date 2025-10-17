"""
Wake Delay Impact Analysis for Paper Figure 1

This script demonstrates the critical importance of evaluation horizon
by showing how measured performance changes with horizon length.

Key insight: Must evaluate for T >= max_delay + T_AH to capture full wake steering benefits

Output:
  - Figure: fig_wake_delay_impact.pdf (publication quality)
  - Data: results/data/wake_delay_analysis.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from py_wake.examples.data.hornsrev1 import V80
from mpcrl.mpc import WindFarmModel, optimize_farm_back2front, farm_energy, run_farm_delay_loop_optimized


def test_eval_horizon_impact():
    """
    Test how evaluation horizon affects measured wake steering benefit.
    """
    print("="*80)
    print("WAKE DELAY IMPACT ANALYSIS")
    print("="*80)
    print("\nThis test demonstrates why evaluation horizon must be long enough")
    print("to capture wake propagation effects.\n")

    # Farm setup
    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])
    wind_conditions = {'ws': 8.0, 'wd': 270.0, 'ti': 0.06}
    initial_yaws = np.array([0.0, 0.0, 0.0])

    # MPC configuration (use recommended params)
    mpc_config = {
        'dt_opt': 30,
        'T_opt': 300,
        'maxfun': 10,
        'r_gamma': 0.3,
        't_AH': 100.0,
        'seed': 42
    }

    print("Farm configuration:")
    print(f"  Turbines: {len(x_pos)}")
    print(f"  Spacing: 500m = {500/80:.1f}D")
    print(f"  Wind: WS={wind_conditions['ws']} m/s, WD={wind_conditions['wd']}°, TI={wind_conditions['ti']}")

    # Calculate wake delays
    model = WindFarmModel(
        x_pos, y_pos, wt=V80(), D=80.0,
        U_inf=wind_conditions['ws'],
        TI=wind_conditions['ti'],
        wd=wind_conditions['wd']
    )

    initial_yaws_sorted = initial_yaws[model.sorted_indices]

    print(f"\nWake propagation delays:")
    wake_delays = []
    for i in range(len(x_pos)):
        for j in range(i+1, len(x_pos)):
            delay = model.delays[i, j]
            if delay > 0:
                print(f"  Turbine {i} → {j}: {delay:.1f}s")
                wake_delays.append(delay)

    max_delay = max(wake_delays)
    print(f"\n  Maximum delay: {max_delay:.1f}s")
    print(f"  Action horizon: {mpc_config['t_AH']:.0f}s")
    print(f"  → Minimum recommended horizon: {max_delay + mpc_config['t_AH']:.0f}s")

    # Test different evaluation horizons
    eval_horizons = [50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 750, 1000]

    print(f"\nTesting {len(eval_horizons)} different evaluation horizons...")
    print("This will take a few minutes...\n")

    # First, optimize once with a long horizon
    print("Step 1: Optimizing yaw trajectories (using T_opt=300s)...")
    optimized_params = optimize_farm_back2front(
        model, initial_yaws_sorted,
        r_gamma=mpc_config['r_gamma'],
        t_AH=mpc_config['t_AH'],
        dt_opt=mpc_config['dt_opt'],
        T_opt=mpc_config['T_opt'],
        maxfun=mpc_config['maxfun'],
        seed=mpc_config['seed'],
        initial_params=None,
        use_time_shifted=True
    )
    print("  ✓ Optimization complete")

    # Greedy baseline (no optimization)
    n_turbines = len(initial_yaws_sorted)
    greedy_params = np.array([[0.5, 0.5] for _ in range(n_turbines)])

    # Now evaluate with different horizons
    print("\nStep 2: Evaluating with different time horizons...")

    results = []
    for i, T_eval in enumerate(eval_horizons):
        print(f"  [{i+1}/{len(eval_horizons)}] T_eval = {T_eval:>4.0f}s...", end=" ", flush=True)

        # Greedy
        t_g, _, P_g = run_farm_delay_loop_optimized(
            model, greedy_params, initial_yaws_sorted,
            r_gamma=mpc_config['r_gamma'],
            t_AH=mpc_config['t_AH'],
            dt=10.0,
            T=T_eval
        )
        greedy_power = farm_energy(P_g, t_g) / t_g[-1]

        # Optimized
        t_o, _, P_o = run_farm_delay_loop_optimized(
            model, optimized_params, initial_yaws_sorted,
            r_gamma=mpc_config['r_gamma'],
            t_AH=mpc_config['t_AH'],
            dt=10.0,
            T=T_eval
        )
        opt_power = farm_energy(P_o, t_o) / t_o[-1]

        gain = (opt_power - greedy_power) / greedy_power * 100

        results.append({
            'eval_horizon': T_eval,
            'greedy_power': greedy_power,
            'optimized_power': opt_power,
            'gain_pct': gain
        })

        print(f"Gain: {gain:>6.2f}%")

    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nMeasured benefit vs evaluation horizon:")
    print(df.to_string(index=False))

    # Find plateau
    final_gain = df.iloc[-1]['gain_pct']
    plateau_idx = np.where(df['gain_pct'] >= 0.95 * final_gain)[0][0]
    plateau_horizon = df.iloc[plateau_idx]['eval_horizon']

    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print("="*80)
    print(f"\n1. Final measured gain (T=1000s): {final_gain:.2f}%")
    print(f"2. Plateau reached at: T={plateau_horizon:.0f}s")
    print(f"3. Theoretical minimum: {max_delay + mpc_config['t_AH']:.0f}s")
    print(f"\n→ Evaluation horizon should be at least {plateau_horizon:.0f}s for this scenario")

    # Save data
    import os
    os.makedirs('results/data', exist_ok=True)
    csv_path = 'results/data/wake_delay_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Data saved to: {csv_path}")

    return df, wake_delays, max_delay, mpc_config


def create_paper_figure(df, wake_delays, max_delay, mpc_config):
    """
    Create publication-quality figure for paper.
    """
    print("\n" + "="*80)
    print("GENERATING PAPER FIGURE")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Power vs horizon
    ax = axes[0]
    ax.plot(df['eval_horizon'], df['greedy_power']/1e6, 'o-',
            label='Greedy (0° yaw)', linewidth=2, markersize=6, color='blue')
    ax.plot(df['eval_horizon'], df['optimized_power']/1e6, 's-',
            label='Wake steering (optimized)', linewidth=2, markersize=6, color='red')

    # Mark wake arrival times
    for i, delay in enumerate(wake_delays):
        ax.axvline(delay, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(delay, ax.get_ylim()[1]*0.95, f'τ{i+1}',
               ha='center', va='top', fontsize=9, color='gray')

    # Mark action horizon
    ax.axvline(mpc_config['t_AH'], color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(mpc_config['t_AH'], ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05,
           f"T_AH={mpc_config['t_AH']:.0f}s",
           ha='center', va='bottom', fontsize=9, color='green',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))

    # Mark recommended minimum
    min_rec = max_delay + mpc_config['t_AH']
    ax.axvline(min_rec, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(min_rec, ax.get_ylim()[1]*0.5, f"Recommended\nminimum\n({min_rec:.0f}s)",
           ha='left', va='center', fontsize=9, color='orange',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='orange', alpha=0.8))

    ax.set_xlabel('Evaluation Horizon (s)', fontsize=12)
    ax.set_ylabel('Average Farm Power (MW)', fontsize=12)
    ax.set_title('Impact of Evaluation Horizon on Measured Power', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, df['eval_horizon'].max() * 1.05)

    # Plot 2: Gain percentage vs horizon
    ax = axes[1]
    ax.plot(df['eval_horizon'], df['gain_pct'], 'o-',
            linewidth=2, markersize=8, color='darkgreen', label='Wake steering benefit')

    # Mark wake arrival times
    for i, delay in enumerate(wake_delays):
        ax.axvline(delay, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(delay, ax.get_ylim()[1]*0.95, f'τ{i+1}={delay:.0f}s',
               ha='center', va='top', fontsize=8, color='gray',
               rotation=90, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Mark recommended minimum
    ax.axvline(min_rec, color='orange', linestyle='--', alpha=0.7, linewidth=2)

    # Shade region before minimum (unreliable)
    ax.axvspan(0, min_rec, alpha=0.15, color='red', label='Unreliable region\n(too short)')

    # Mark plateau
    final_gain = df.iloc[-1]['gain_pct']
    ax.axhline(final_gain, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(df['eval_horizon'].max()*0.7, final_gain*1.05,
           f'Plateau: {final_gain:.2f}%',
           fontsize=9, color='blue')

    ax.set_xlabel('Evaluation Horizon (s)', fontsize=12)
    ax.set_ylabel('Power Gain vs Greedy (%)', fontsize=12)
    ax.set_title('Measured Benefit vs Evaluation Horizon', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, df['eval_horizon'].max() * 1.05)
    ax.set_ylim(min(df['gain_pct'].min()*0.9, 0), df['gain_pct'].max() * 1.15)

    plt.tight_layout()

    # Save figure
    import os
    os.makedirs('results/figures', exist_ok=True)

    # Save as PDF for paper
    pdf_path = 'results/figures/fig_wake_delay_impact.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"  ✓ PDF saved to: {pdf_path}")

    # Also save as PNG for quick viewing
    png_path = 'results/figures/fig_wake_delay_impact.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ PNG saved to: {png_path}")

    plt.show()


if __name__ == "__main__":
    # Run analysis
    df, wake_delays, max_delay, mpc_config = test_eval_horizon_impact()

    # Generate figure
    create_paper_figure(df, wake_delays, max_delay, mpc_config)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutputs:")
    print("  - Data: results/data/wake_delay_analysis.csv")
    print("  - Figure (PDF): results/figures/fig_wake_delay_impact.pdf")
    print("  - Figure (PNG): results/figures/fig_wake_delay_impact.png")
    print("\nUse the PDF for the paper, PNG for quick reference.")
    print("\nThis figure demonstrates Section 6.1.1 findings:")
    print("  - Short horizons underestimate wake steering benefit")
    print("  - Benefit plateaus once horizon > max_delay + T_AH")
    print("  - Critical for fair comparison between control strategies")
