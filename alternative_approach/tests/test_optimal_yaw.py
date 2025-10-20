"""
Find Optimal Yaw Angles with PyWake (Grid Search)
==================================================

Test what the ACTUAL optimal yaw angles are for comparison with MPC.
"""

import numpy as np
from itertools import product
from nmpc_windfarm_acados_fixed import (
    Farm, Wind, build_pywake_model, pywake_farm_power
)

print("=" * 70)
print("Finding Optimal Yaw Angles (Grid Search)")
print("=" * 70)

# Same setup as demo
D = 178.0
x = np.array([0.0, 5*D, 10*D, 15*D])
y = np.zeros_like(x)
wind = Wind(U=8.0, theta=270.0, TI=0.06)

wf_model, layout = build_pywake_model(x, y, D, ti=wind.TI)

# Baseline: all turbines at zero yaw
psi_zero = np.zeros(4)
P_baseline = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_zero)
print(f"\nBaseline (all at 0°): {P_baseline/1e6:.4f} MW")

# Grid search over reasonable yaw angles
print("\nSearching yaw space (this may take a minute)...")
yaw_range = np.arange(-25, 26, 5)  # -25 to +25 in 5° steps

best_power = P_baseline
best_yaw = psi_zero.copy()
n_evaluated = 0

# Only yaw first 3 turbines (last one should stay at 0)
for yaw0 in yaw_range:
    for yaw1 in yaw_range:
        for yaw2 in yaw_range:
            psi_test = np.array([yaw0, yaw1, yaw2, 0.0])
            P_test = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_test)
            n_evaluated += 1

            if P_test > best_power:
                best_power = P_test
                best_yaw = psi_test.copy()

gain_pct = (best_power / P_baseline - 1) * 100

print(f"\nEvaluated {n_evaluated} combinations")
print("\n" + "=" * 70)
print("OPTIMAL YAW ANGLES (Grid Search):")
print("=" * 70)
print(f"  Turbine 0: {best_yaw[0]:+6.1f}°")
print(f"  Turbine 1: {best_yaw[1]:+6.1f}°")
print(f"  Turbine 2: {best_yaw[2]:+6.1f}°")
print(f"  Turbine 3: {best_yaw[3]:+6.1f}°")
print(f"\n  Optimal power: {best_power/1e6:.4f} MW")
print(f"  Baseline power: {P_baseline/1e6:.4f} MW")
print(f"  Gain: {gain_pct:+.2f}%")
print("=" * 70)

# Also test a few specific cases mentioned in docs
print("\nTesting specific angles:")
test_cases = [
    ([20, 20, 20, 0], "All at +20°"),
    ([-20, -20, -20, 0], "All at -20°"),
    ([15, 15, 15, 0], "All at +15°"),
    ([25, 20, 15, 0], "Decreasing upstream to downstream"),
]

for yaw, desc in test_cases:
    P = pywake_farm_power(wf_model, layout, wind.U, wind.theta, np.array(yaw))
    gain = (P / P_baseline - 1) * 100
    print(f"  {desc:40s}: {P/1e6:.4f} MW ({gain:+.2f}%)")

print("\n" + "=" * 70)
print("COMPARISON WITH MPC:")
print("  MPC found: [-2.4°, -3.7°, -3.7°, 0°] → +0.4% gain")
print(f"  Optimal:   [{best_yaw[0]:.0f}°, {best_yaw[1]:.0f}°, {best_yaw[2]:.0f}°, {best_yaw[3]:.0f}°] → {gain_pct:+.1f}% gain")
print("  ⚠️  MPC is FAR from optimal!")
print("=" * 70)
