"""
Find optimal yaw configuration for 5D spacing (Demo 1 setup).

At 5D spacing, wake interactions are strong and yaw optimization should help.
Let's find what the TRUE optimum is.
"""

import numpy as np
from nmpc_windfarm_acados_fixed import (
    build_pywake_model, pywake_farm_power
)

# Setup - 5D spacing like Demo 1
D = 178.0
spacing = 5 * D  # 5D like Demo 1
x = np.array([0.0, spacing, 2*spacing, 3*spacing])
y = np.zeros_like(x)

wind_U = 8.0
wind_theta = 0.0

print("Building PyWake model...")
wf_model, layout = build_pywake_model(x, y, D)

print("\n" + "="*70)
print("Systematic Search for Optimal Yaw (5D Spacing)")
print("="*70)
print(f"\nSetup: 4 turbines at {spacing/D}D = {spacing:.0f}m spacing")
print(f"Wind: {wind_U} m/s at {wind_theta}°")
print()

# Test many configurations
test_configs = [
    ("All aligned", np.array([0.0, 0.0, 0.0, 0.0])),

    # Upstream turbines yawed (classic wake steering)
    ("T0 +10°", np.array([10.0, 0.0, 0.0, 0.0])),
    ("T0 +20°", np.array([20.0, 0.0, 0.0, 0.0])),
    ("T0 +25°", np.array([25.0, 0.0, 0.0, 0.0])),
    ("T0,T1 +20°", np.array([20.0, 20.0, 0.0, 0.0])),
    ("T0,T1,T2 +20°", np.array([20.0, 20.0, 20.0, 0.0])),

    # Negative yaws
    ("T0 -20°", np.array([-20.0, 0.0, 0.0, 0.0])),
    ("T0,T1 -20°", np.array([-20.0, -20.0, 0.0, 0.0])),
    ("T0,T1,T2 -20°", np.array([-20.0, -20.0, -20.0, 0.0])),

    # Fine tuning around expected optimum
    ("T0,T1,T2 +15°", np.array([15.0, 15.0, 15.0, 0.0])),
    ("T0,T1,T2 +18°", np.array([18.0, 18.0, 18.0, 0.0])),
    ("T0,T1,T2 +22°", np.array([22.0, 22.0, 22.0, 0.0])),

    # All turbines yawed
    ("All +20°", np.array([20.0, 20.0, 20.0, 20.0])),

    # Mixed strategies
    ("T0 +25, rest 0", np.array([25.0, 0.0, 0.0, 0.0])),
    ("Graded", np.array([25.0, 20.0, 15.0, 0.0])),
]

# Also test what the MPC states were
demo_configs = [
    ("Demo initial", np.array([-0.8, 2.7, 1.4, 0.6])),
    ("Demo state A", np.array([6.7, -4.8, -6.1, -6.9])),
]

test_configs.extend(demo_configs)

results = []

for name, psi in test_configs:
    P = pywake_farm_power(wf_model, layout, wind_U, wind_theta, psi) / 1e6
    results.append((name, psi, P))

# Sort by power
results.sort(key=lambda x: x[2], reverse=True)

print("Configurations sorted by power (best first):\n")
print(f"{'Rank':<5} {'Configuration':<25} {'Yaw angles (°)':<35} {'Power (MW)':<12} {'vs Aligned'}")
print("="*100)

P_aligned = next(P for name, psi, P in results if name == "All aligned")

for rank, (name, psi, P) in enumerate(results, 1):
    psi_str = f"[{psi[0]:+5.1f}, {psi[1]:+5.1f}, {psi[2]:+5.1f}, {psi[3]:+5.1f}]"
    gain_kW = (P - P_aligned) * 1000
    gain_pct = (P / P_aligned - 1) * 100

    marker = ""
    if "Demo" in name:
        marker = "  ← MPC state"
    elif abs(P - P_aligned) < 1e-6:
        marker = "  ← Baseline"
    elif rank == 1:
        marker = "  ★ BEST"

    print(f"{rank:<5} {name:<25} {psi_str:<35} {P:>11.6f}   {gain_kW:+8.1f} kW ({gain_pct:+.3f}%){marker}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

best_name, best_psi, best_P = results[0]
print(f"\nBest configuration: {best_name}")
print(f"  Yaws: {np.round(best_psi, 1)}°")
print(f"  Power: {best_P:.6f} MW")
print(f"  Gain over aligned: {(best_P - P_aligned)*1000:+.1f} kW ({(best_P/P_aligned - 1)*100:+.2f}%)")

# Check if there are significant gains
if (best_P - P_aligned) / P_aligned > 0.01:  # > 1% gain
    print(f"\n✅ SIGNIFICANT GAINS AVAILABLE (>{1}%)")
    print(f"   Yaw optimization IS worthwhile for 5D spacing!")
else:
    print(f"\n⚠️  Small gains (< 1%)")

# Check where demo states rank
demo_ranks = [(name, i) for i, (name, _, _) in enumerate(results, 1) if "Demo" in name]
print(f"\nDemo MPC states:")
for name, rank in demo_ranks:
    print(f"  {name} ranks: #{rank} out of {len(results)}")

print("\n" + "="*70)
print("GRID SEARCH for fine-tuning")
print("="*70)

# Grid search around expected optimum
print("\nSearching T0,T1,T2 yaws from 10° to 25° in 2° steps...")

best_grid_P = 0
best_grid_psi = None

for yaw in range(10, 27, 2):
    psi = np.array([float(yaw), float(yaw), float(yaw), 0.0])
    P = pywake_farm_power(wf_model, layout, wind_U, wind_theta, psi) / 1e6

    gain_kW = (P - P_aligned) * 1000

    if P > best_grid_P:
        best_grid_P = P
        best_grid_psi = psi.copy()

    marker = "★" if P > best_P else ""
    print(f"  [{yaw}, {yaw}, {yaw}, 0]: {P:.6f} MW ({gain_kW:+.1f} kW) {marker}")

print(f"\nBest from grid search:")
print(f"  Yaws: {np.round(best_grid_psi, 1)}°")
print(f"  Power: {best_grid_P:.6f} MW")
print(f"  Gain: {(best_grid_P - P_aligned)*1000:+.1f} kW ({(best_grid_P/P_aligned - 1)*100:+.2f}%)")

print("\n" + "="*70)
print("CONCLUSION FOR 5D SPACING")
print("="*70)

if (best_grid_P - P_aligned) / P_aligned > 0.005:  # > 0.5%
    print(f"\n✅ Yaw optimization IS beneficial at 5D spacing")
    print(f"   Best strategy: Yaw upstream turbines, keep downstream aligned")
    print(f"   Expected gain: {(best_grid_P/P_aligned - 1)*100:.1f}%")
    print(f"\n❌ BUT: MPC is NOT finding this optimum!")
    print(f"   MPC oscillates between suboptimal states")
    print(f"\n→ Need to fix MPC to find the correct optimum")
else:
    print(f"\n⚠️  Even at 5D, gains are small (< 0.5%)")
    print(f"   Wake model or turbine spacing may not be ideal for yaw control")
