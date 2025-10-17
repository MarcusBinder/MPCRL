"""
Test to verify when wake steering provides benefit vs greedy (0° yaw).

Tests multiple wind conditions to find scenarios where optimization helps.
"""

import numpy as np
import matplotlib.pyplot as plt
from py_wake.examples.data.hornsrev1 import V80
from mpcrl.mpc import WindFarmModel, optimize_farm_back2front, farm_energy, run_farm_delay_loop_optimized
import time

def evaluate_scenario(ws, wd, ti, optimize=True):
    """Evaluate a scenario with and without optimization."""
    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])

    model = WindFarmModel(
        x_pos, y_pos, wt=V80(), D=80.0,
        U_inf=ws, TI=ti, wd=wd,
        cache_size=5000, cache_quant=0.25
    )

    initial_yaws = np.array([0.0, 0.0, 0.0])
    initial_yaws_sorted = initial_yaws[model.sorted_indices]

    # Greedy (no optimization) - maintain current yaws (0°) for entire horizon
    # Create params that produce zero change: o1=0.5 → psi returns 0 → dgamma=0
    n_turbines = len(initial_yaws_sorted)
    greedy_params = np.array([[0.5, 0.5] for _ in range(n_turbines)])

    # Evaluate greedy with same methodology as optimized (apples-to-apples)
    t_greedy, _, P_greedy = run_farm_delay_loop_optimized(
        model, greedy_params, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0, dt=10.0, T=100.0
    )

    total_energy_greedy = farm_energy(P_greedy, t_greedy)
    avg_power_greedy = total_energy_greedy / t_greedy[-1]

    if not optimize:
        return {
            'ws': ws, 'wd': wd, 'ti': ti,
            'greedy_power': avg_power_greedy,
            'greedy_individual': P_greedy.mean(axis=1),  # Average power per turbine
            'optimized_power': None,
            'optimized_yaws': None,
            'benefit': None,
            'benefit_pct': None
        }

    # Optimized
    t_start = time.time()
    optimized_params = optimize_farm_back2front(
        model, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0,
        dt_opt=20, T_opt=300, maxfun=20,
        seed=42, initial_params=None, use_time_shifted=True
    )
    t_opt = time.time() - t_start

    # Evaluate optimized solution
    t_eval, _, P_eval = run_farm_delay_loop_optimized(
        model, optimized_params, initial_yaws_sorted,
        r_gamma=0.3, t_AH=100.0, dt=10.0, T=100.0
    )

    total_energy = farm_energy(P_eval, t_eval)
    avg_power_opt = total_energy / t_eval[-1]

    # Get final yaws
    from mpcrl.mpc import yaw_traj
    final_yaws = []
    for i, params in enumerate(optimized_params):
        _, traj = yaw_traj(initial_yaws_sorted[i], params[0], params[1],
                          100.0, 0.3, 10.0, 100.0)
        final_yaws.append(traj[-1])
    final_yaws = np.array(final_yaws)

    benefit = avg_power_opt - avg_power_greedy
    benefit_pct = (benefit / avg_power_greedy) * 100

    return {
        'ws': ws, 'wd': wd, 'ti': ti,
        'greedy_power': avg_power_greedy,
        'greedy_individual': P_greedy.mean(axis=1),
        'optimized_power': avg_power_opt,
        'optimized_yaws': final_yaws,
        'benefit': benefit,
        'benefit_pct': benefit_pct,
        'time': t_opt
    }


def test_multiple_conditions():
    """Test various wind conditions to see when wake steering helps."""

    print("\n" + "="*80)
    print("WAKE STEERING BENEFIT ANALYSIS")
    print("="*80)
    print("\nTesting when wake steering provides benefit over greedy (0° yaw) strategy\n")

    # Test different conditions
    test_cases = [
        # Original test case
        {'ws': 8.0, 'wd': 270.0, 'ti': 0.06, 'label': 'Original test (WS=8, low TI)'},

        # Higher wind speeds (more power at stake)
        {'ws': 10.0, 'wd': 270.0, 'ti': 0.06, 'label': 'Higher wind speed (WS=10)'},
        {'ws': 12.0, 'wd': 270.0, 'ti': 0.06, 'label': 'High wind speed (WS=12)'},

        # Lower turbulence (stronger wakes)
        {'ws': 8.0, 'wd': 270.0, 'ti': 0.03, 'label': 'Low turbulence (TI=0.03)'},
        {'ws': 10.0, 'wd': 270.0, 'ti': 0.03, 'label': 'WS=10, low TI'},

        # Higher turbulence (weaker wakes)
        {'ws': 8.0, 'wd': 270.0, 'ti': 0.10, 'label': 'High turbulence (TI=0.10)'},

        # Different wind directions (partial wake)
        {'ws': 8.0, 'wd': 275.0, 'ti': 0.06, 'label': 'Slight misalignment (WD=275)'},
        {'ws': 8.0, 'wd': 280.0, 'ti': 0.06, 'label': 'More misalignment (WD=280)'},

        # Combination: high WS, low TI (should show biggest benefit)
        {'ws': 12.0, 'wd': 270.0, 'ti': 0.03, 'label': 'Best case: WS=12, TI=0.03'},
    ]

    results = []

    for i, case in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {case['label']}")
        result = evaluate_scenario(case['ws'], case['wd'], case['ti'])
        results.append(result)

        print(f"  Greedy:     {result['greedy_power']:>10,.0f} W")
        print(f"  Optimized:  {result['optimized_power']:>10,.0f} W")
        print(f"  Benefit:    {result['benefit']:>10,.0f} W ({result['benefit_pct']:>+6.2f}%)")
        print(f"  Opt yaws:   [{result['optimized_yaws'][0]:>6.2f}, "
              f"{result['optimized_yaws'][1]:>6.2f}, {result['optimized_yaws'][2]:>6.2f}]")
        print()

    # Analysis
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    positive_benefits = [r for r in results if r['benefit_pct'] > 0]
    negative_benefits = [r for r in results if r['benefit_pct'] <= 0]

    print(f"\nScenarios where wake steering HELPS: {len(positive_benefits)}/{len(results)}")
    if positive_benefits:
        print("\nBest scenarios for wake steering:")
        sorted_results = sorted(positive_benefits, key=lambda x: x['benefit_pct'], reverse=True)
        for r in sorted_results[:3]:
            idx = results.index(r)
            print(f"  {test_cases[idx]['label']}")
            print(f"    Benefit: {r['benefit_pct']:+.2f}% ({r['benefit']:+,.0f} W)")

    print(f"\nScenarios where wake steering HURTS: {len(negative_benefits)}/{len(results)}")
    if negative_benefits:
        print("\nWorst scenarios (wake steering reduces power):")
        sorted_results = sorted(negative_benefits, key=lambda x: x['benefit_pct'])
        for r in sorted_results[:3]:
            idx = results.index(r)
            print(f"  {test_cases[idx]['label']}")
            print(f"    Loss: {r['benefit_pct']:.2f}% ({r['benefit']:,.0f} W)")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Benefit by wind speed
    ax = axes[0, 0]
    ws_270_ti006 = [r for r in results if r['wd'] == 270.0 and abs(r['ti'] - 0.06) < 0.01]
    if ws_270_ti006:
        ws_vals = [r['ws'] for r in ws_270_ti006]
        benefits = [r['benefit_pct'] for r in ws_270_ti006]
        ax.plot(ws_vals, benefits, 'o-', linewidth=2, markersize=10)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Benefit (%)')
        ax.set_title('Wake Steering Benefit vs Wind Speed\n(WD=270°, TI=0.06)')
        ax.grid(True, alpha=0.3)

    # Plot 2: Benefit by turbulence intensity
    ax = axes[0, 1]
    ws8_wd270 = [r for r in results if r['ws'] == 8.0 and r['wd'] == 270.0]
    if ws8_wd270:
        ti_vals = [r['ti'] for r in ws8_wd270]
        benefits = [r['benefit_pct'] for r in ws8_wd270]
        ax.plot(ti_vals, benefits, 'o-', linewidth=2, markersize=10, color='orange')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Turbulence Intensity')
        ax.set_ylabel('Benefit (%)')
        ax.set_title('Wake Steering Benefit vs Turbulence\n(WS=8 m/s, WD=270°)')
        ax.grid(True, alpha=0.3)

    # Plot 3: All scenarios comparison
    ax = axes[1, 0]
    greedy_powers = [r['greedy_power'] for r in results]
    opt_powers = [r['optimized_power'] for r in results]
    colors = ['green' if b > 0 else 'red' for b in [r['benefit_pct'] for r in results]]

    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width/2, greedy_powers, width, label='Greedy (0°)', alpha=0.7)
    ax.bar(x + width/2, opt_powers, width, label='Optimized', alpha=0.7)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Total Power (W)')
    ax.set_title('Power Comparison: Greedy vs Optimized')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(len(results))], rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Benefit distribution
    ax = axes[1, 1]
    benefits = [r['benefit_pct'] for r in results]
    colors_bar = ['green' if b > 0 else 'red' for b in benefits]
    bars = ax.bar(range(len(benefits)), benefits, color=colors_bar, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Benefit (%)')
    ax.set_title('Wake Steering Benefit by Scenario')
    ax.set_xticks(range(len(benefits)))
    ax.set_xticklabels([f'{i+1}' for i in range(len(results))], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('tests/wake_steering_benefit_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: tests/wake_steering_benefit_analysis.png")
    plt.show()

    return results


if __name__ == "__main__":
    results = test_multiple_conditions()

    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
1. Wake steering benefit depends heavily on wind conditions
2. Lower TI (stronger wakes) → more benefit from steering
3. Higher WS → more power at stake → bigger absolute benefits
4. At some conditions, greedy (0° yaw) is actually optimal!

For RL training:
- Agent needs to learn when to apply wake steering vs greedy
- Testing on varied wind conditions is crucial
- Don't expect wake steering to always help
    """)
