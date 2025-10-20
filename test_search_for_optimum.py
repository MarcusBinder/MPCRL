"""
Search for the true optimum by testing many configurations.

Is ψ=[0,0,0,0] actually optimal, or is there a better configuration?
"""

import numpy as np
from nmpc_windfarm_acados_fixed import (
    build_pywake_model, pywake_farm_power,
    Farm, Wind
)

# Setup
np.random.seed(42)
D = 178.0
spacing = 7 * D
x = np.array([0.0, spacing, 2*spacing, 3*spacing])
y = np.zeros_like(x)

wind = Wind(U=8.0, theta=0.0)

print("Building PyWake model...")
wf_model, layout = build_pywake_model(x, y, D)

print("\n" + "="*70)
print("Systematic Search for Optimal Yaw Configuration")
print("="*70)

# Test many different configurations
test_configs = [
    ("All aligned", np.array([0.0, 0.0, 0.0, 0.0])),
    ("Upstream yawed +10°", np.array([10.0, 0.0, 0.0, 0.0])),
    ("Upstream yawed -10°", np.array([-10.0, 0.0, 0.0, 0.0])),
    ("Upstream yawed +20°", np.array([20.0, 0.0, 0.0, 0.0])),
    ("Upstream yawed -20°", np.array([-20.0, 0.0, 0.0, 0.0])),
    ("All +5°", np.array([5.0, 5.0, 5.0, 5.0])),
    ("All -5°", np.array([-5.0, -5.0, -5.0, -5.0])),
    ("All +10°", np.array([10.0, 10.0, 10.0, 10.0])),
    ("Alternating ±10°", np.array([10.0, -10.0, 10.0, -10.0])),
    ("Random 1", np.array([5.3, -3.2, 7.1, -4.5])),
    ("Random 2", np.array([-8.2, 12.3, -5.7, 9.1])),
    ("T0 optimized", np.array([25.0, 0.0, 0.0, 0.0])),  # Maximum yaw
    ("T0 optimized neg", np.array([-25.0, 0.0, 0.0, 0.0])),
]

# From the demo logs - these are what MPC is oscillating between
demo_configs = [
    ("Demo state A", np.array([6.7, -4.8, -6.1, -6.9])),
    ("Demo state B", np.array([-0.8, 2.7, 1.4, 0.6])),
]

test_configs.extend(demo_configs)

results = []

for name, psi in test_configs:
    P = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi) / 1e6
    results.append((name, psi, P))

# Sort by power
results.sort(key=lambda x: x[2], reverse=True)

print("\nConfigurations sorted by power (best first):\n")
print(f"{'Rank':<5} {'Configuration':<25} {'Yaw angles (°)':<35} {'Power (MW)':<12} {'vs Aligned'}")
print("="*100)

P_aligned = next(P for name, psi, P in results if name == "All aligned")

for rank, (name, psi, P) in enumerate(results, 1):
    psi_str = f"[{psi[0]:+5.1f}, {psi[1]:+5.1f}, {psi[2]:+5.1f}, {psi[3]:+5.1f}]"
    gain_kW = (P - P_aligned) * 1000
    gain_pct = (P / P_aligned - 1) * 100

    marker = ""
    if name == "Demo state A" or name == "Demo state B":
        marker = "  ← MPC oscillates here"
    elif abs(P - P_aligned) < 1e-6:
        marker = "  ← Baseline"

    print(f"{rank:<5} {name:<25} {psi_str:<35} {P:>11.6f}   {gain_kW:+8.1f} kW ({gain_pct:+.3f}%){marker}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

best_name, best_psi, best_P = results[0]
print(f"\nBest configuration: {best_name}")
print(f"  Yaws: {np.round(best_psi, 2)}°")
print(f"  Power: {best_P:.6f} MW")
print(f"  Gain over aligned: {(best_P - P_aligned)*1000:+.1f} kW ({(best_P/P_aligned - 1)*100:+.2f}%)")

# Find where demo states rank
demo_A_rank = next(i for i, (name, _, _) in enumerate(results, 1) if name == "Demo state A")
demo_B_rank = next(i for i, (name, _, _) in enumerate(results, 1) if name == "Demo state B")

print(f"\nDemo oscillation states:")
print(f"  State A ranks: #{demo_A_rank} out of {len(results)}")
print(f"  State B ranks: #{demo_B_rank} out of {len(results)}")

if demo_A_rank > 5 or demo_B_rank > 5:
    print("\n❌ MPC is oscillating between SUB-OPTIMAL states!")
    print("   The aligned configuration is actually BETTER than what MPC finds.")
else:
    print("\n⚠️  MPC states are near-optimal, but still oscillating")

# Check if any configuration beats aligned by > 1%
significant_gains = [(name, psi, P) for name, psi, P in results if (P - P_aligned)/P_aligned > 0.01]

if not significant_gains:
    print("\n🔍 CRITICAL FINDING:")
    print("   NO configuration achieves > 1% gain over aligned!")
    print("   This suggests yaw optimization has LIMITED benefit for this layout.")
    print("   The oscillations are happening in a nearly FLAT region of the objective.")
elif abs(best_P - P_aligned) < 1e-3:
    print("\n🔍 CRITICAL FINDING:")
    print("   Best power ≈ aligned power (within 1 kW)")
    print("   The objective function is essentially FLAT near ψ=[0,0,0,0]")
    print("   This makes optimization extremely difficult!")

# Analyze symmetry
print("\n" + "="*70)
print("SYMMETRY ANALYSIS")
print("="*70)

# Check if +yaw and -yaw give same power (should if wake model is symmetric)
P_plus_10 = next(P for name, _, P in results if name == "Upstream yawed +10°")
P_minus_10 = next(P for name, _, P in results if name == "Upstream yawed -10°")

if abs(P_plus_10 - P_minus_10) < 1e-6:
    print(f"\n✓ System is SYMMETRIC: P(+10°) = P(-10°) = {P_plus_10:.6f} MW")
    print("  This confirms ψ=0 is a critical point due to symmetry.")
else:
    print(f"\n⚠ System is NOT symmetric: P(+10°) = {P_plus_10:.6f}, P(-10°) = {P_minus_10:.6f}")

# Check if it's a maximum or saddle
if P_plus_10 < P_aligned and P_minus_10 < P_aligned:
    print("  ψ=0 is a LOCAL MAXIMUM (both directions decrease power)")
elif P_plus_10 > P_aligned or P_minus_10 > P_aligned:
    print("  ψ=0 is a SADDLE POINT (some directions increase power)")
else:
    print("  ψ=0 is approximately FLAT (very small changes in power)")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

print("\nThe MPC problem is:")
if abs(best_P - P_aligned) < 0.01:  # Less than 10 kW difference
    print("  1. Objective is nearly FLAT - no significant gains available")
    print("  2. Small gradients + numerical noise → large control actions")
    print("  3. Oscillations happen in a flat region where all states are equally bad")
    print("\nSOLUTION: Accept that ψ=[0,0,0,0] is near-optimal!")
    print("  • Add stronger regularization to penalize movement")
    print("  • Increase move penalty λ significantly")
    print("  • Or: Don't use yaw control for this layout (not worth it!)")
else:
    print("  1. Better configurations exist!")
    print("  2. MPC is not finding them due to:")
    print("     - Poor initialization")
    print("     - Getting stuck in local optima")
    print("     - Linearization inaccuracy")
    print("\nSOLUTION: Improve optimization")
    print("  • Use multi-start optimization")
    print("  • Better initialization heuristics")
    print("  • Stronger nonlinear solver")
