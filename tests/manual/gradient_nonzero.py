"""
Test gradient at a non-zero yaw configuration.

Since ψ=[0,0,0,0] has zero gradient, let's test at a more interesting point.
"""

import numpy as np
from nmpc_windfarm_acados_fixed import (
    build_pywake_model, pywake_farm_power, finite_diff_gradient,
    Farm, Wind
)

# Setup - simple 4-turbine case
np.random.seed(42)
D = 178.0
spacing = 7 * D  # 7D spacing
x = np.array([0.0, spacing, 2*spacing, 3*spacing])
y = np.zeros_like(x)
farm = Farm(x=x, y=y, D=D)

wind = Wind(U=8.0, theta=0.0)

# Build PyWake model
print("Building PyWake model...")
wf_model, layout = build_pywake_model(x, y, D)

# Test at several different points
test_points = [
    ("Random small yaws", np.array([2.0, -1.5, 1.0, -0.5])),
    ("Large yaw on T0", np.array([10.0, 0.0, 0.0, 0.0])),
    ("Large yaw on T3", np.array([0.0, 0.0, 0.0, 10.0])),
    ("All positive", np.array([5.0, 5.0, 5.0, 5.0])),
]

for name, psi_test in test_points:
    print("\n" + "="*70)
    print(f"TEST: {name}")
    print("="*70)
    print(f"Test point: ψ = {psi_test}°")

    # Compute power at test point
    P_center = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_test)
    print(f"Power: {P_center/1e6:.6f} MW")

    # Compute gradient using finite differences
    P_from_grad, grad_P = finite_diff_gradient(
        wf_model, layout, wind.U, wind.theta, psi_test, eps=1e-2
    )

    # Convert to MW/deg for readability
    grad_P_MW = grad_P / 1e6

    print(f"Gradient (MW/deg): {grad_P_MW}")
    print(f"Gradient norm: {np.linalg.norm(grad_P_MW):.4f} MW/deg")

    # Check if gradient is zero
    if np.allclose(grad_P, 0, atol=100):
        print("⚠ Gradient is essentially ZERO - this is a stationary point!")
        continue

    # Test what happens if we take a small step in the gradient direction
    print(f"\nTaking step in gradient ascent direction...")
    step_size = 1.0  # 1 degree step
    grad_normalized = grad_P / np.linalg.norm(grad_P)
    psi_stepped = psi_test + step_size * grad_normalized

    P_stepped = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_stepped)

    print(f"Step: {step_size}° in gradient direction")
    print(f"New yaws: {psi_stepped}°")
    print(f"Power before: {P_center/1e6:.6f} MW")
    print(f"Power after:  {P_stepped/1e6:.6f} MW")
    print(f"Change: {(P_stepped - P_center)/1e6:.6f} MW ({(P_stepped/P_center - 1)*100:+.3f}%)")

    if P_stepped > P_center:
        print("✓ Power INCREASED - gradient ascent is working!")
    else:
        print("❌ Power DECREASED - something is wrong!")

    # Also test opposite direction (should decrease power)
    psi_opposite = psi_test - step_size * grad_normalized
    P_opposite = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_opposite)

    print(f"\nTest opposite direction (should decrease power):")
    print(f"Power in -grad direction: {P_opposite/1e6:.6f} MW")
    print(f"Change: {(P_opposite - P_center)/1e6:.6f} MW ({(P_opposite/P_center - 1)*100:+.3f}%)")

    if P_opposite < P_center:
        print("✓ Power DECREASED as expected")
    else:
        print("❌ Power INCREASED - gradient may have wrong sign!")

print("\n" + "="*70)
print("CRITICAL FINDING")
print("="*70)
print("\nIf ψ=[0,0,0,0] has zero gradient, the MPC has NO direction to move!")
print("This could explain why we see strange behavior.")
print("\nPossible solutions:")
print("1. Start from a random non-zero initial condition")
print("2. Add small random perturbations to break symmetry")
print("3. Use a different initialization strategy")
