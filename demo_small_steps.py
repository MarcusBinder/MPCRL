"""
Test with SMALL rate limit to keep linearization valid.

The issue: Taking 7.5° steps invalidates the linearization.
Solution: Reduce rate limit to 0.05 deg/s → 0.75° per step
"""

import numpy as np
from nmpc_windfarm_acados_fixed import (
    AcadosYawMPC, Farm, Wind, Limits, MPCConfig
)

# Setup
np.random.seed(42)
D = 178.0
spacing = 7 * D
x = np.array([0.0, spacing, 2*spacing, 3*spacing])
y = np.zeros_like(x)
farm = Farm(x=x, y=y, D=D)

wind = Wind(U=8.0, theta=0.0)

print("="*70)
print("MPC with SMALL rate limit (keep linearization valid)")
print("="*70)

# SMALL rate limit
limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.05)  # 0.05 deg/s

cfg = MPCConfig(dt=15.0, N_h=20, lam_move=1.0)

print(f"\nConfiguration:")
print(f"  Rate limit: {limits.yaw_rate_max} deg/s")
print(f"  Max change per step: {limits.yaw_rate_max * cfg.dt}°")
print(f"  Move penalty: λ = {cfg.lam_move}")
print()

# Create controller
print("Building controller...")
controller = AcadosYawMPC(farm, wind, limits, cfg)

# Start from moderate yaws
psi_initial = np.array([2.0, -1.5, 1.0, -0.5])
controller.psi_current = psi_initial.copy()

print(f"Initial yaws: {np.round(psi_initial, 2)}°")
print()

# Run MPC
N_steps = 50
print(f"Running MPC for {N_steps} steps...")
print()

yaws = [psi_initial.copy()]
powers = []

for t in range(N_steps):
    info = controller.step()
    psi = info['psi']
    P = info['power'] / 1e6

    yaws.append(psi.copy())
    powers.append(P)

    if t < 10 or t % 5 == 0:
        print(f"t={t:02d}, ψ={np.round(psi, 2)}, P={P:.3f} MW")

print()
print("="*70)
print("Analysis")
print("="*70)

# Check for oscillations
print("\nLast 10 yaw configurations:")
for i in range(max(0, len(yaws) - 10), len(yaws)):
    print(f"  t={i-1:02d}: {np.round(yaws[i], 2)}")

# Check if oscillating
oscillating = False
for i in range(max(0, len(yaws) - 8), len(yaws) - 2):
    if np.allclose(yaws[i], yaws[i+2], atol=0.1):
        oscillating = True
        print(f"\n❌ OSCILLATION detected: yaws[{i}] ≈ yaws[{i+2}]")
        break

if not oscillating:
    print("\n✓ No periodic oscillations detected")

# Check convergence
yaw_changes = np.abs(np.diff(yaws[-10:], axis=0))
max_change = np.max(yaw_changes)
mean_change = np.mean(yaw_changes)

print(f"\nYaw changes in last 10 steps:")
print(f"  Max: {max_change:.3f}°")
print(f"  Mean: {mean_change:.3f}°")

if max_change < 0.1:
    print("  ✓ CONVERGED (changes < 0.1°)")
elif max_change < 0.5:
    print("  ✓ Nearly converged (changes < 0.5°)")
else:
    print("  ⚠ Still changing")

# Check final state
psi_final = yaws[-1]
P_final = powers[-1]

print(f"\nFinal state:")
print(f"  Yaws: {np.round(psi_final, 2)}°")
print(f"  Power: {P_final:.3f} MW")

if np.allclose(psi_final, 0, atol=1.0):
    print("  ✓ Converged to ψ≈[0,0,0,0] (optimal!)")
else:
    print(f"  Distance from optimal: {np.linalg.norm(psi_final):.2f}°")

# Check power trend
power_changes = np.diff(powers)
decreases = np.sum(power_changes < -1e-6)

print(f"\nPower trend:")
print(f"  Initial: {powers[0]:.3f} MW")
print(f"  Final: {powers[-1]:.3f} MW")
print(f"  Change: {1000*(powers[-1] - powers[0]):+.1f} kW")
print(f"  Decreases: {decreases}/{len(power_changes)} steps")

if decreases < len(power_changes) / 10:
    print("  ✓ Power mostly increasing")
else:
    print("  ⚠ Many power decreases")

print("\n" + "="*70)
print("Conclusion")
print("="*70)

if not oscillating and max_change < 0.5:
    print("\n✅ SUCCESS! Small rate limit prevents oscillations.")
    print("    MPC converges smoothly to optimal configuration.")
    print("\nThe key insight:")
    print("  • Linearization P(ψ) ≈ P₀ + ∇P·ψ is only valid for SMALL ψ")
    print("  • Large steps (7.5°) violate this assumption")
    print("  • Small steps (0.75°) keep linearization accurate")
    print("\nRecommendation:")
    print("  • Use rate limit ≤ 0.1 deg/s for this approach")
    print("  • OR use nonlinear MPC (more complex)")
else:
    print("\n❌ Still has issues.")
    print("May need fundamentally different approach:")
    print("  • Nonlinear MPC instead of successive linearization")
    print("  • Or accept that ψ=[0,0,0,0] is optimal (don't optimize!)")
