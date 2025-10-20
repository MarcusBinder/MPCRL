"""
Test 3: Simple gradient ascent without MPC.

This tests if we can optimize yaws using just gradient ascent with a small step size.
If this works but MPC doesn't, it confirms the problem is overstepping in MPC.
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

wind = Wind(U=8.0, theta=0.0)

# Build PyWake model
print("Building PyWake model...")
wf_model, layout = build_pywake_model(x, y, D)

print("\n" + "="*70)
print("Simple Gradient Ascent Test")
print("="*70)

# Start from random yaws (same as demo)
psi = np.array([2.0, -1.5, 1.0, -0.5])
print(f"\nInitial yaws: {psi}°")

# Compute initial power
P_initial = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi) / 1e6
print(f"Initial power: {P_initial:.6f} MW")

# Gradient ascent parameters
alpha = 0.5  # Step size (degrees)
max_iter = 50
yaw_limit = 25.0

print(f"\nGradient ascent with step size α = {alpha}°")
print(f"Yaw limits: ±{yaw_limit}°")
print()

P_history = [P_initial]
psi_history = [psi.copy()]

for i in range(max_iter):
    # Compute gradient
    P_current, grad_P = finite_diff_gradient(
        wf_model, layout, wind.U, wind.theta, psi, eps=1e-2
    )
    P_current_MW = P_current / 1e6
    grad_P_MW = grad_P / 1e6

    grad_norm = np.linalg.norm(grad_P_MW)

    # Take a gradient ascent step
    if grad_norm > 1e-6:
        psi_new = psi + alpha * (grad_P / np.linalg.norm(grad_P))
    else:
        print(f"t={i:02d}: Gradient is zero - converged!")
        break

    # Apply limits
    psi_new = np.clip(psi_new, -yaw_limit, yaw_limit)

    # Compute new power
    P_new = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_new) / 1e6

    # Print status
    print(f"t={i:02d}, ψ={np.round(psi_new, 1)}, P={P_new:.3f} MW, ΔP={1000*(P_new - P_current_MW):+.1f} kW, |∇P|={grad_norm:.2e} MW/deg")

    # Store history
    P_history.append(P_new)
    psi_history.append(psi_new.copy())

    # Check convergence
    if np.abs(P_new - P_current_MW) < 1e-6:
        print(f"\nConverged! Power change < 1 W")
        break

    # Update state
    psi = psi_new

print("\n" + "="*70)
print("Summary")
print("="*70)

P_final = P_history[-1]
psi_final = psi_history[-1]

print(f"\nInitial power: {P_initial:.6f} MW")
print(f"Final power:   {P_final:.6f} MW")
print(f"Power gain:    {1000*(P_final - P_initial):+.1f} kW ({100*(P_final/P_initial - 1):+.2f}%)")
print(f"\nInitial yaws: {psi_history[0]}°")
print(f"Final yaws:   {np.round(psi_final, 2)}°")
print(f"Total change: {np.round(psi_final - psi_history[0], 2)}°")

# Check if power increased monotonically
power_diffs = np.diff(P_history)
if np.all(power_diffs >= -1e-6):
    print("\n✓ Power increased MONOTONICALLY - gradient ascent works!")
else:
    print("\n❌ Power decreased at some point - something is wrong")
    decreases = np.where(power_diffs < -1e-6)[0]
    print(f"   Decreases at steps: {decreases}")

# Check if we converged to a steady state
final_changes = np.abs(np.diff(psi_history[-5:], axis=0))
if np.max(final_changes) < 0.1:
    print("✓ Yaws CONVERGED to steady state (no oscillations)")
else:
    print("❌ Yaws did NOT converge (still changing)")

print("\n" + "="*70)
print("COMPARISON TO MPC")
print("="*70)
print("\nFrom the demo log, MPC exhibited:")
print("  - Oscillations between two states")
print("  - Power DECREASED overall")
print("  - Large sudden changes in yaws")
print("\nSimple gradient ascent shows:")
if np.all(power_diffs >= -1e-6) and np.max(final_changes) < 0.1:
    print("  ✓ Monotonic power increase")
    print("  ✓ Convergence to steady state")
    print("  ✓ No oscillations")
    print("\n→ This CONFIRMS the MPC problem is OVERSTEPPING in successive linearization!")
    print("→ The gradient is correct, but MPC takes too large steps")
else:
    print("  ⚠ Gradient ascent also has issues")
    print("→ Need to investigate further")
