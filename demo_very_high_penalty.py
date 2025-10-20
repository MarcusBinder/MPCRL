"""
Test with VERY HIGH move penalty (λ=100).

The gradient magnitude is ~15,000-30,000 W/deg.
With rate limit 0.5 deg/s and λ=5.0, move cost ~0.625
But gradient term is ~200,000x larger!

Need much higher λ to balance.
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
limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.5)

# Test multiple penalty values
penalty_values = [5.0, 50.0, 500.0]

for lam in penalty_values:
    print("\n" + "="*70)
    print(f"Testing λ = {lam}")
    print("="*70)

    cfg = MPCConfig(dt=15.0, N_h=20, lam_move=lam)

    print(f"Move penalty: λ = {lam}")
    print()

    # Create controller
    controller = AcadosYawMPC(farm, wind, limits, cfg)

    # Start from same initial condition
    psi_initial = np.array([-1.3, 4.5, 2.3, 1.0])
    controller.psi_current = psi_initial.copy()

    print(f"Initial yaws: {np.round(psi_initial, 1)}°")

    # Run 10 steps
    yaws = [psi_initial.copy()]
    powers = []

    for t in range(10):
        info = controller.step()
        yaws.append(info['psi'].copy())
        powers.append(info['power'] / 1e6)

    # Check for oscillations
    oscillating = False
    for i in range(max(0, len(yaws) - 6), len(yaws) - 2):
        if np.allclose(yaws[i], yaws[i+2], atol=0.5):
            oscillating = True
            break

    # Check yaw changes
    yaw_changes = np.abs(np.diff(yaws[-5:], axis=0))
    max_change = np.max(yaw_changes)

    print(f"\nAfter 10 steps:")
    print(f"  Final yaws: {np.round(yaws[-1], 1)}°")
    print(f"  Final power: {powers[-1]:.3f} MW")
    print(f"  Max yaw change (last 5): {max_change:.2f}°")

    if oscillating:
        print(f"  ❌ Still oscillating")
    elif max_change < 1.0:
        print(f"  ✓ Converged (no oscillations)")
    else:
        print(f"  ⚠ Large changes but not periodic")

    if np.allclose(yaws[-1], 0, atol=2.0):
        print(f"  ✓ Near optimal ψ≈[0,0,0,0]")

print("\n" + "="*70)
print("Recommendation")
print("="*70)

print("\nIf λ=500 still oscillates, the problem is more fundamental.")
print("Possible issues:")
print("  1. Gradient scaling is wrong (need to scale differently)")
print("  2. Cost function formulation issue")
print("  3. Successive linearization approach won't work for this problem")
print("\nMay need to:")
print("  • Use nonlinear cost instead of linearized")
print("  • Add explicit damping term")
print("  • Limit yaw changes directly in constraints")
