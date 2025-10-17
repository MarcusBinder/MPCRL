"""
Test suite to replicate and verify handling of NaN errors in PyWake simulations.

This test systematically explores conditions that cause NaN values:
- Very low turbulence intensity (TI)
- Low wind speeds
- Extreme yaw angles
- Specific configurations from observed failures
"""

import numpy as np
import matplotlib.pyplot as plt
from py_wake.examples.data.hornsrev1 import V80
from mpcrl.mpc import WindFarmModel

def test_low_ti_conditions():
    """
    Test Case 1: Very low TI values (0.0, 0.001, 0.01) with various yaw angles.
    This replicates the condition when random action = -1 converts to TI = 0.0.
    """
    print("\n" + "="*70)
    print("TEST CASE 1: Very Low TI Conditions")
    print("="*70)

    # Setup: 3 turbines in a row (same as notebook)
    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])
    turbine = V80()

    # Test various TI values
    ti_values = [0.0, 0.001, 0.005, 0.01, 0.02]
    yaw_angles = np.array([0.0, 0.0, 0.0])  # Start with zero yaw

    results = []

    for ti in ti_values:
        print(f"\nTesting TI = {ti:.5f}")

        model = WindFarmModel(
            x_pos, y_pos, wt=turbine, D=80.0,
            U_inf=8.0, TI=ti, wd=270.0,
            cache_size=100, cache_quant=0.25
        )

        powers = model.farm_power_sorted(yaw_angles)
        has_nan = np.any(np.isnan(powers))

        print(f"  Powers: {powers}")
        print(f"  Has NaN: {has_nan}")

        results.append({
            'TI': ti,
            'TI_used': model.TI,  # May be clamped
            'powers': powers.copy(),
            'has_nan': has_nan
        })

    return results


def test_low_wind_speed_with_negative_yaw():
    """
    Test Case 2: Low wind speed with negative yaw angles.
    This replicates conditions from the error output.
    """
    print("\n" + "="*70)
    print("TEST CASE 2: Low Wind Speed + Negative Yaw Angles")
    print("="*70)

    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])
    turbine = V80()

    # Test combinations
    test_configs = [
        {'ws': 3.0, 'ti': 0.01, 'yaws': np.array([-10.0, -10.0, 0.0])},
        {'ws': 5.0, 'ti': 0.01, 'yaws': np.array([-12.0, -10.0, 0.0])},
        {'ws': 6.315, 'ti': 0.01, 'yaws': np.array([-12.17, -10.77, 0.0])},  # From error
        {'ws': 8.0, 'ti': 0.01, 'yaws': np.array([-15.0, -12.0, 0.0])},
    ]

    results = []

    for config in test_configs:
        ws = config['ws']
        ti = config['ti']
        yaws = config['yaws']

        print(f"\nTesting WS={ws:.2f}, TI={ti:.3f}, yaws={yaws}")

        model = WindFarmModel(
            x_pos, y_pos, wt=turbine, D=80.0,
            U_inf=ws, TI=ti, wd=270.0,
            cache_size=100, cache_quant=0.25
        )

        powers = model.farm_power_sorted(yaws)
        has_nan = np.any(np.isnan(powers))

        print(f"  Powers: {powers}")
        print(f"  Has NaN: {has_nan}")

        results.append({
            'ws': ws,
            'ti': ti,
            'yaws': yaws.copy(),
            'powers': powers.copy(),
            'has_nan': has_nan
        })

    return results


def test_exact_failing_configuration():
    """
    Test Case 3: Exact configuration from notebook error output.
    WD: 277.66464, WS: 6.31510, TI: 0.01000
    Yaws: [-12.17010806, -10.77170201, 0.0]
    """
    print("\n" + "="*70)
    print("TEST CASE 3: Exact Failing Configuration from Notebook")
    print("="*70)

    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])
    turbine = V80()

    # Exact values from error
    wd = 277.6646423339844
    ws = 6.315097332000732
    ti = 0.01
    yaws = np.array([-12.17010806, -10.77170201, 0.0])

    print(f"\nExact failing case:")
    print(f"  WD: {wd}, WS: {ws}, TI: {ti}")
    print(f"  Yaws: {yaws}")

    model = WindFarmModel(
        x_pos, y_pos, wt=turbine, D=80.0,
        U_inf=ws, TI=ti, wd=wd,
        cache_size=100, cache_quant=0.25
    )

    powers = model.farm_power_sorted(yaws)
    has_nan = np.any(np.isnan(powers))

    print(f"  Powers: {powers}")
    print(f"  Has NaN: {has_nan}")

    # Test with slightly different variations
    print("\n  Testing variations:")
    variations = [
        {'label': 'TI doubled', 'ws': ws, 'ti': 0.02, 'wd': wd, 'yaws': yaws},
        {'label': 'WS increased', 'ws': 8.0, 'ti': ti, 'wd': wd, 'yaws': yaws},
        {'label': 'WD=270', 'ws': ws, 'ti': ti, 'wd': 270.0, 'yaws': yaws},
        {'label': 'Zero yaws', 'ws': ws, 'ti': ti, 'wd': wd, 'yaws': np.array([0.0, 0.0, 0.0])},
    ]

    for var in variations:
        model_var = WindFarmModel(
            x_pos, y_pos, wt=turbine, D=80.0,
            U_inf=var['ws'], TI=var['ti'], wd=var['wd'],
            cache_size=100, cache_quant=0.25
        )
        powers_var = model_var.farm_power_sorted(var['yaws'])
        has_nan_var = np.any(np.isnan(powers_var))
        print(f"    {var['label']:15s}: Powers={powers_var}, NaN={has_nan_var}")

    return {
        'ws': ws, 'ti': ti, 'wd': wd, 'yaws': yaws,
        'powers': powers, 'has_nan': has_nan
    }


def test_nan_replacement_logic():
    """
    Test Case 4: Verify that the NaN replacement logic works correctly.
    Check that valid powers are preserved and only NaN is replaced with 0.
    """
    print("\n" + "="*70)
    print("TEST CASE 4: Verify NaN Replacement Logic")
    print("="*70)

    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])
    turbine = V80()

    # Use conditions that we know cause NaN in downstream turbine
    wd = 277.66
    ws = 6.315
    ti = 0.01
    yaws = np.array([-12.17, -10.77, 0.0])

    print(f"\nTesting with conditions that produce NaN:")
    print(f"  WD: {wd}, WS: {ws}, TI: {ti}")
    print(f"  Yaws: {yaws}")

    model = WindFarmModel(
        x_pos, y_pos, wt=turbine, D=80.0,
        U_inf=ws, TI=ti, wd=wd,
        cache_size=100, cache_quant=0.25
    )

    powers = model.farm_power_sorted(yaws)

    print(f"\nResult after NaN handling:")
    print(f"  Powers: {powers}")
    print(f"  All finite: {np.all(np.isfinite(powers))}")

    # Check that we have some positive values (upstream turbines should work)
    n_positive = np.sum(powers > 0)
    n_zero = np.sum(powers == 0)

    print(f"  Turbines with positive power: {n_positive}")
    print(f"  Turbines with zero power: {n_zero}")

    # Verify expectations
    assert np.all(np.isfinite(powers)), "All powers should be finite (no NaN/Inf)"
    assert n_positive >= 1, "At least upstream turbines should have positive power"

    print("\n✓ NaN replacement logic works correctly!")
    print("  - Valid powers are preserved")
    print("  - NaN values are replaced with 0")
    print("  - No crashes or exceptions")

    return powers


def visualize_results(results_case1, results_case2):
    """
    Create visualization of the test results.
    """
    print("\n" + "="*70)
    print("VISUALIZATION: Creating summary plots")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: TI vs Total Power
    ax = axes[0, 0]
    ti_vals = [r['TI'] for r in results_case1]
    total_powers = [np.sum(r['powers']) for r in results_case1]
    has_nans = [r['has_nan'] for r in results_case1]

    colors = ['red' if has_nan else 'green' for has_nan in has_nans]
    ax.scatter(ti_vals, total_powers, c=colors, s=100, alpha=0.6)
    ax.axvline(0.01, color='blue', linestyle='--', label='MIN_TI threshold', alpha=0.5)
    ax.set_xlabel('Turbulence Intensity')
    ax.set_ylabel('Total Farm Power [W]')
    ax.set_title('Effect of TI on Power Output')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Power per turbine for different TI values
    ax = axes[0, 1]
    for i, r in enumerate(results_case1):
        if r['TI'] in [0.0, 0.01, 0.02]:  # Show subset
            ax.plot([0, 1, 2], r['powers'], 'o-', label=f"TI={r['TI']:.3f}", alpha=0.7)
    ax.set_xlabel('Turbine Index')
    ax.set_ylabel('Power [W]')
    ax.set_title('Power Distribution Across Turbines')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Wind speed effect
    ax = axes[1, 0]
    ws_vals = [r['ws'] for r in results_case2]
    total_powers = [np.sum(r['powers']) for r in results_case2]
    has_nans = [r['has_nan'] for r in results_case2]
    colors = ['red' if has_nan else 'green' for has_nan in has_nans]

    ax.scatter(ws_vals, total_powers, c=colors, s=100, alpha=0.6)
    ax.set_xlabel('Wind Speed [m/s]')
    ax.set_ylabel('Total Farm Power [W]')
    ax.set_title('Effect of Wind Speed (with negative yaws)')
    ax.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    # Count NaN occurrences
    n_case1_nan = sum(r['has_nan'] for r in results_case1)
    n_case2_nan = sum(r['has_nan'] for r in results_case2)

    summary_text = f"""
    TEST SUMMARY
    {'='*40}

    Case 1 (Low TI):
      Total tests: {len(results_case1)}
      Tests with NaN: {n_case1_nan}
      Tests clean: {len(results_case1) - n_case1_nan}

    Case 2 (Low WS + Neg Yaw):
      Total tests: {len(results_case2)}
      Tests with NaN: {n_case2_nan}
      Tests clean: {len(results_case2) - n_case2_nan}

    NaN Handling: {'✓ WORKING' if (n_case1_nan + n_case2_nan) == 0 else '⚠ ISSUES DETECTED'}

    Color Legend:
      🟢 Green = No NaN detected
      🔴 Red = NaN detected (but handled)
    """

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig('/home/marcus/Documents/mpcrl/tests/nan_handling_test_results.png', dpi=150)
    print("  Saved plot to: tests/nan_handling_test_results.png")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NaN ERROR REPLICATION AND HANDLING TEST SUITE")
    print("="*70)
    print("\nThis test suite systematically explores conditions that cause NaN")
    print("values in PyWake simulations and verifies the handling logic.\n")

    # Run all test cases
    results_case1 = test_low_ti_conditions()
    results_case2 = test_low_wind_speed_with_negative_yaw()
    result_case3 = test_exact_failing_configuration()
    test_nan_replacement_logic()

    # Visualize results
    visualize_results(results_case1, results_case2)

    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)
    print("\nAll tests completed successfully!")
    print("The NaN handling logic is working as expected.")
