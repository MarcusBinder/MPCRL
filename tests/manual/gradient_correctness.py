"""
Test 2: Verify gradient correctness

This tests whether the finite difference gradient computation is actually correct.
We'll compute the gradient at ψ=[0,0,0,0] and manually verify it by computing
power at nearby points.
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

print("\n" + "="*70)
print("TEST: Gradient Correctness at ψ=[0,0,0,0]")
print("="*70)

# Test point: all turbines aligned
psi_test = np.zeros(4)
print(f"\nTest point: ψ = {psi_test}")

# Compute power at test point
P_center = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_test)
print(f"Power at ψ=[0,0,0,0]: {P_center:.6f} MW")

# Compute gradient using finite differences
print("\nComputing gradient using finite_diff_gradient()...")
P_from_grad, grad_P = finite_diff_gradient(
    wf_model, layout, wind.U, wind.theta, psi_test, eps=1e-2
)
print(f"Power (from gradient function): {P_from_grad:.6f} MW")
print(f"Gradient: {grad_P}")
print(f"Gradient norm: {np.linalg.norm(grad_P):.2e}")

# Manually verify each component of the gradient
print("\n" + "="*70)
print("Manual Verification of Each Gradient Component")
print("="*70)

eps = 1e-2  # Same epsilon as used in finite_diff_gradient
grad_manual = np.zeros(4)

for i in range(4):
    # Perturb turbine i in positive direction
    psi_plus = psi_test.copy()
    psi_plus[i] += eps
    P_plus = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_plus)

    # Perturb turbine i in negative direction
    psi_minus = psi_test.copy()
    psi_minus[i] -= eps
    P_minus = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_minus)

    # Central difference
    grad_manual[i] = (P_plus - P_minus) / (2 * eps)

    print(f"\nTurbine {i}:")
    print(f"  P(ψ + ε·e_{i}) = {P_plus:.6f} MW")
    print(f"  P(ψ - ε·e_{i}) = {P_minus:.6f} MW")
    print(f"  ΔP = {P_plus - P_minus:.6f} MW")
    print(f"  ∂P/∂ψ_{i} = {grad_manual[i]:.6f} MW/deg")
    print(f"  From function: {grad_P[i]:.6f} MW/deg")

    # Check if they match
    if np.abs(grad_manual[i] - grad_P[i]) < 1e-6:
        print(f"  ✓ Match!")
    else:
        print(f"  ❌ MISMATCH! Difference: {abs(grad_manual[i] - grad_P[i]):.6e}")

print("\n" + "="*70)
print("Summary")
print("="*70)

print(f"\nGradient from function: {grad_P}")
print(f"Gradient manual check:  {grad_manual}")
print(f"Difference: {grad_P - grad_manual}")
print(f"Max difference: {np.max(np.abs(grad_P - grad_manual)):.6e}")

if np.allclose(grad_P, grad_manual, atol=1e-6):
    print("\n✓ Gradient computation is CORRECT")
else:
    print("\n❌ Gradient computation has ERRORS")

# Interpretation
print("\n" + "="*70)
print("Gradient Interpretation")
print("="*70)

print("\nWhat should the gradient look like at ψ=[0,0,0,0]?")
print("- Turbine 0 (most upstream): Should have small gradient (not in wake)")
print("- Turbines 1,2,3 (downstream): In strong wakes, gradient depends on wake deflection")
print("\nPositive gradient → Power increases when yaw increases")
print("Negative gradient → Power decreases when yaw increases")

for i in range(4):
    direction = "increasing yaw INCREASES power" if grad_P[i] > 0 else "increasing yaw DECREASES power"
    print(f"Turbine {i}: ∂P/∂ψ = {grad_P[i]:+.2e} MW/deg → {direction}")

# Sanity checks
print("\n" + "="*70)
print("Sanity Checks")
print("="*70)

# Check 1: Turbine 0 should have very small gradient (not affected by wakes)
if np.abs(grad_P[0]) < 100:
    print("✓ Turbine 0 (upstream) has small gradient (not in wake)")
else:
    print(f"⚠ Turbine 0 gradient is large: {grad_P[0]:.2e} MW/deg")
    print("  This is unexpected for the most upstream turbine")

# Check 2: Downstream turbines should have non-zero gradients
downstream_grad_norm = np.linalg.norm(grad_P[1:])
if downstream_grad_norm > 100:
    print(f"✓ Downstream turbines have significant gradients: {downstream_grad_norm:.2e}")
else:
    print(f"⚠ Downstream turbines have very small gradients: {downstream_grad_norm:.2e}")
    print("  This is unexpected - they should be affected by upstream wakes")

# Check 3: Gradient magnitude should be reasonable (MW/deg)
# For a ~14 MW farm, ±0.01 MW per degree is reasonable
max_grad = np.max(np.abs(grad_P))
if 0.0001 < max_grad < 1.0:
    print(f"✓ Gradient magnitude is reasonable: {max_grad:.2e} MW/deg")
else:
    print(f"⚠ Gradient magnitude seems off: {max_grad:.2e} MW/deg")

# Test what happens if we take a small step in the gradient direction
print("\n" + "="*70)
print("Test: Take a small step in gradient direction")
print("="*70)

step_size = 0.1  # degrees
psi_stepped = psi_test + step_size * (grad_P / np.linalg.norm(grad_P))
P_stepped = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_stepped)

print(f"\nStep size: {step_size}° in direction of normalized gradient")
print(f"New yaws: {psi_stepped}")
print(f"Power before: {P_center:.6f} MW")
print(f"Power after:  {P_stepped:.6f} MW")
print(f"Change: {P_stepped - P_center:.6f} MW ({(P_stepped/P_center - 1)*100:+.3f}%)")

predicted_change = grad_P.T @ (psi_stepped - psi_test)
print(f"\nPredicted change (linear): {predicted_change:.6f} MW")
print(f"Actual change:             {P_stepped - P_center:.6f} MW")
print(f"Prediction error:          {abs(predicted_change - (P_stepped - P_center)):.6e} MW")

if P_stepped > P_center:
    print("\n✓ Power INCREASED - gradient points in the right direction!")
else:
    print("\n❌ Power DECREASED - gradient may be wrong sign or problem is non-convex")
