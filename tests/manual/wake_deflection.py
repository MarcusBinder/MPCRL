"""
Test if wake deflection is actually happening with yaw.

For a straight-line layout, yawing upstream turbines should:
1. Reduce their own power (cosine loss)
2. Deflect wake away from downstream turbines (potential gain)

Let's check if #2 is happening.
"""

import numpy as np
from nmpc_windfarm_acados_fixed import (
    build_pywake_model, pywake_farm_power
)

# Setup - 5D spacing
D = 178.0
spacing = 5 * D
x = np.array([0.0, spacing, 2*spacing, 3*spacing])
y = np.zeros_like(x)

wind_U = 8.0
wind_theta = 0.0

print("Building PyWake model...")
wf_model, layout = build_pywake_model(x, y, D)

print("\n" + "="*70)
print("Wake Deflection Analysis")
print("="*70)

# Test: Does yawing T0 affect downstream turbines?
configs = [
    ("All aligned", np.array([0.0, 0.0, 0.0, 0.0])),
    ("T0 yawed +20°", np.array([20.0, 0.0, 0.0, 0.0])),
    ("T0 yawed -20°", np.array([-20.0, 0.0, 0.0, 0.0])),
]

print("\nPower breakdown by turbine:")
print(f"{'Configuration':<20} {'T0 (MW)':<10} {'T1 (MW)':<10} {'T2 (MW)':<10} {'T3 (MW)':<10} {'Total (MW)'}")
print("="*80)

results = []
for name, psi in configs:
    # Run simulation
    sim_res = wf_model(
        x=layout["x"],
        y=layout["y"],
        wd=np.array([wind_theta]),
        ws=np.array([wind_U]),
        yaw=psi.reshape(4, 1, 1),
        tilt=0
    )

    # Get individual turbine powers
    # Debug: check shape
    if name == "All aligned":
        print(f"\nDebug: Power array shape: {sim_res.Power.values.shape}")
        print(f"Debug: Power values: {sim_res.Power.values}")

    # Flatten and convert to MW
    P_turbines = sim_res.Power.values.flatten() / 1e6
    P_total = P_turbines.sum()

    results.append((name, psi, P_turbines, P_total))

    print(f"{name:<20} {P_turbines[0]:<10.3f} {P_turbines[1]:<10.3f} {P_turbines[2]:<10.3f} {P_turbines[3]:<10.3f} {P_total:<10.3f}")

print("\n" + "="*70)
print("Analysis")
print("="*70)

aligned_name, aligned_psi, aligned_powers, aligned_total = results[0]
plus_name, plus_psi, plus_powers, plus_total = results[1]
minus_name, minus_psi, minus_powers, minus_total = results[2]

print(f"\n1. Effect of yawing T0 by +20°:")
print(f"   T0 power: {aligned_powers[0]:.3f} → {plus_powers[0]:.3f} MW ({plus_powers[0] - aligned_powers[0]:+.3f} MW)")
print(f"   T1 power: {aligned_powers[1]:.3f} → {plus_powers[1]:.3f} MW ({plus_powers[1] - aligned_powers[1]:+.3f} MW)")
print(f"   T2 power: {aligned_powers[2]:.3f} → {plus_powers[2]:.3f} MW ({plus_powers[2] - aligned_powers[2]:+.3f} MW)")
print(f"   T3 power: {aligned_powers[3]:.3f} → {plus_powers[3]:.3f} MW ({plus_powers[3] - aligned_powers[3]:+.3f} MW)")
print(f"   Total:    {aligned_total:.3f} → {plus_total:.3f} MW ({plus_total - aligned_total:+.3f} MW)")

if plus_powers[1] > aligned_powers[1] + 0.01:
    print("\n   ✓ Wake deflection IS working - T1 gains power")
elif plus_powers[1] < aligned_powers[1] - 0.01:
    print("\n   ⚠ T1 LOSES power - unexpected!")
else:
    print("\n   ⚠ T1 power unchanged - no wake deflection benefit")

print(f"\n2. Net benefit analysis:")
T0_loss = plus_powers[0] - aligned_powers[0]
downstream_gain = (plus_powers[1:].sum() - aligned_powers[1:].sum())

print(f"   T0 loss (cosine):      {T0_loss:+.3f} MW")
print(f"   Downstream gain:       {downstream_gain:+.3f} MW")
print(f"   Net effect:            {T0_loss + downstream_gain:+.3f} MW")

if downstream_gain > abs(T0_loss):
    print("\n   → Downstream gains EXCEED upstream losses (good!)")
elif downstream_gain > 0:
    print("\n   → Downstream gains exist but DON'T compensate for upstream loss")
else:
    print("\n   → NO downstream gains (wake deflection not beneficial)")

print("\n" + "="*70)
print("Root Cause")
print("="*70)

print("\nFor a STRAIGHT-LINE layout aligned with wind:")
print("  • All turbines are directly behind each other")
print("  • Zero cross-wind separation")
print("  • Wake deflection moves wake laterally BUT...")
print("  • Downstream turbines are still IN THE CENTERLINE")
print("  • They experience nearly the same wake deficit")
print("\nConclusion:")
if downstream_gain < 0.1:
    print("  ❌ Yaw control provides NO benefit for straight-line layouts")
    print("  ✓ This explains why ψ=[0,0,0,0] is optimal!")
    print("\nTo get benefits from yaw control, you need:")
    print("  1. Staggered layout (cross-wind spacing)")
    print("  2. Wind direction not aligned with layout")
    print("  3. Or both")
else:
    print("  ✓ There ARE small benefits from wake steering")
    print("  → MPC should be able to find them")

# Check expected cosine loss
print("\n" + "="*70)
print("Theoretical Cosine Loss")
print("="*70)

P_T0_aligned = aligned_powers[0]
theoretical_loss_20deg = P_T0_aligned * (1 - np.cos(np.deg2rad(20))**3)

actual_loss = aligned_powers[0] - plus_powers[0]

print(f"\nFor T0 at +20° yaw:")
print(f"  Theoretical cosine loss (P × (1 - cos³(20°))): {theoretical_loss_20deg:.3f} MW")
print(f"  Actual T0 power loss:                           {actual_loss:.3f} MW")

if abs(actual_loss - theoretical_loss_20deg) / theoretical_loss_20deg < 0.2:
    print(f"  ✓ Close match - confirms cosine loss formula")
else:
    print(f"  ⚠ Mismatch - may be additional factors")
