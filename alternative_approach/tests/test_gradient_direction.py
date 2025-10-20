"""
Test to verify gradient direction is correct.
"""
import numpy as np
from nmpc_windfarm_acados_fixed import (
    Farm, Wind, build_pywake_model, pywake_farm_power, finite_diff_gradient
)

# Setup
D = 178.0
x = np.array([0.0, 5*D, 10*D, 15*D])
y = np.zeros_like(x)
wind = Wind(U=8.0, theta=270.0, TI=0.06)

wf_model, layout = build_pywake_model(x, y, D, ti=wind.TI)

# Test gradient at zero yaw
psi_zero = np.zeros(4)
P_zero = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_zero)
print(f"\nPower at zero yaw: {P_zero/1e6:.6f} MW")

# Test power at small positive yaw for first turbine
psi_test = np.array([5.0, 0.0, 0.0, 0.0])
P_test = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_test)
print(f"Power with turbine 0 at +5°: {P_test/1e6:.6f} MW")
print(f"Power change: {(P_test - P_zero)/1e6:.6f} MW ({(P_test/P_zero - 1)*100:.3f}%)")

# Compute gradient with small epsilon
P_small, grad_small, hess_small = finite_diff_gradient(
    wf_model, layout, wind.U, wind.theta, psi_zero,
    eps=1e-2, return_hessian=True
)

# Compute gradient with larger epsilon
P, grad, hess = finite_diff_gradient(
    wf_model, layout, wind.U, wind.theta, psi_zero,
    eps=0.5, return_hessian=True
)

print(f"\nGradient with eps=1e-2 deg (too small):")
for i in range(4):
    print(f"  Turbine {i}: dP/dψ = {grad_small[i]:.2e} W/deg")

print(f"\nGradient with eps=0.5 deg (better):")
for i in range(4):
    print(f"  Turbine {i}: dP/dψ = {grad[i]:.2e} W/deg")

print(f"\nInterpretation:")
print(f"  Positive gradient → power INCREASES with positive yaw")
print(f"  Negative gradient → power DECREASES with positive yaw")
print(f"  To maximize power, move in direction of gradient (positive grad → increase yaw)")
