"""
Manually test gradient computation.
"""
import numpy as np
from nmpc_windfarm_acados_fixed import (
    Farm, Wind, build_pywake_model, pywake_farm_power
)

# Setup
D = 178.0
x = np.array([0.0, 5*D, 10*D, 15*D])
y = np.zeros_like(x)
wind = Wind(U=8.0, theta=270.0, TI=0.06)

wf_model, layout = build_pywake_model(x, y, D, ti=wind.TI)

# Test gradient at zero yaw
psi_zero = np.zeros(4)
eps = 0.5

# Compute gradient for turbine 0
P0 = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_zero)
print(f"P at [0, 0, 0, 0]: {P0/1e6:.6f} MW")

psi_plus = psi_zero.copy()
psi_plus[0] += eps
P_plus = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_plus)
print(f"P at [+{eps}, 0, 0, 0]: {P_plus/1e6:.6f} MW")

psi_minus = psi_zero.copy()
psi_minus[0] -= eps
P_minus = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_minus)
print(f"P at [-{eps}, 0, 0, 0]: {P_minus/1e6:.6f} MW")

grad_0 = (P_plus - P_minus) / (2 * eps)
print(f"\nGradient for turbine 0: dP/dψ = {grad_0:.2e} W/deg")
print(f"Expected (from +5° test): ~13,600 W/deg")

# Also test the arrays themselves
print(f"\nDEBUG:")
print(f"  psi_zero: {psi_zero}")
print(f"  psi_plus: {psi_plus}")
print(f"  psi_minus: {psi_minus}")
print(f"  P0: {P0}")
print(f"  P_plus: {P_plus}")
print(f"  P_minus: {P_minus}")
print(f"  Difference: P_plus - P_minus = {P_plus - P_minus}")
