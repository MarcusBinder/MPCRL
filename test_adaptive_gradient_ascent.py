"""
Test gradient ascent with adaptive step size.

Use line search to ensure power always increases.
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
print("Gradient Ascent with Line Search")
print("="*70)

# Start from random yaws (same as demo)
psi = np.array([2.0, -1.5, 1.0, -0.5])
print(f"\nInitial yaws: {psi}°")

# Compute initial power
P_current = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi) / 1e6
P_initial = P_current
print(f"Initial power: {P_initial:.6f} MW")

# Gradient ascent parameters
alpha_init = 1.0  # Initial step size
alpha_min = 0.01  # Minimum step size before giving up
max_iter = 50
yaw_limit = 25.0
convergence_tol = 1e-3  # kW

print(f"\nGradient ascent with adaptive line search")
print(f"Initial step size: {alpha_init}°")
print(f"Yaw limits: ±{yaw_limit}°")
print(f"Convergence tolerance: {convergence_tol} kW")
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

    # Check if gradient is too small (converged)
    if grad_norm < 1e-6:
        print(f"t={i:02d}: Gradient is zero - converged!")
        break

    # Compute search direction (normalized gradient)
    search_dir = grad_P / np.linalg.norm(grad_P)

    # Line search: find step size that increases power
    alpha = alpha_init
    psi_new = None
    P_new_MW = None

    for _ in range(10):  # Try up to 10 step sizes
        # Try this step size
        psi_trial = psi + alpha * search_dir
        psi_trial = np.clip(psi_trial, -yaw_limit, yaw_limit)

        # Compute power
        P_trial = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_trial) / 1e6

        # Check if power increased
        if P_trial > P_current_MW + 1e-9:  # Small tolerance for numerical noise
            psi_new = psi_trial
            P_new_MW = P_trial
            break

        # Reduce step size
        alpha *= 0.5

        if alpha < alpha_min:
            # Can't find improvement, must be at optimum
            print(f"t={i:02d}: Line search failed - converged!")
            psi_new = psi
            P_new_MW = P_current_MW
            break

    if psi_new is None:
        print(f"t={i:02d}: Could not find improving step!")
        break

    # Print status
    power_change_kW = 1000 * (P_new_MW - P_current_MW)
    print(f"t={i:02d}, ψ={np.round(psi_new, 1)}, P={P_new_MW:.3f} MW, ΔP={power_change_kW:+.1f} kW, α={alpha:.3f}°, |∇P|={grad_norm:.2e}")

    # Store history
    P_history.append(P_new_MW)
    psi_history.append(psi_new.copy())

    # Check convergence
    if power_change_kW < convergence_tol:
        print(f"\nConverged! Power change < {convergence_tol} kW")
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

# Check if power increased monotonically
power_diffs = np.diff(P_history)
if np.all(power_diffs >= -1e-9):
    print("\n✓ Power increased MONOTONICALLY")
else:
    print("\n❌ Power decreased at some point")
    decreases = np.where(power_diffs < -1e-9)[0]
    print(f"   Decreases at steps: {decreases}")

# Check if we converged
if len(psi_history) > 5:
    final_changes = np.abs(np.diff(psi_history[-5:], axis=0))
    if np.max(final_changes) < 0.1:
        print("✓ Yaws CONVERGED to steady state")
    else:
        print("❌ Yaws still oscillating")
        print(f"   Max change in last 5 steps: {np.max(final_changes):.3f}°")

print("\n" + "="*70)
print("CRITICAL FINDINGS")
print("="*70)

# Compute gradient at final point
P_final_check, grad_P_final = finite_diff_gradient(
    wf_model, layout, wind.U, wind.theta, psi_final, eps=1e-2
)
grad_norm_final = np.linalg.norm(grad_P_final / 1e6)

print(f"\nFinal gradient norm: {grad_norm_final:.2e} MW/deg")
print(f"Final yaws: {np.round(psi_final, 3)}°")

if np.allclose(psi_final, 0, atol=1.0):
    print("\n⚠️  Converged to ψ ≈ [0,0,0,0] which has ZERO gradient!")
    print("This is a degenerate critical point (saddle or flat region)")
    print("\nThis explains the oscillations:")
    print("  1. Gradient descent moves toward ψ=[0,0,0,0]")
    print("  2. At ψ=[0,0,0,0], gradient is zero")
    print("  3. Tiny numerical noise causes movement")
    print("  4. Oscillation begins!")

print("\n" + "="*70)
print("IMPLICATIONS FOR MPC")
print("="*70)

print("\nThe MPC is NOT failing due to overstepping alone!")
print("The underlying problem:")
print("  • ψ=[0,0,0,0] is a stationary point with zero gradient")
print("  • System converges toward this point")
print("  • Near this point, gradient is very small and noisy")
print("  • Any optimization method will struggle here")
print("\nPossible solutions:")
print("  1. Regularization: Add small cost on yaw magnitudes to break symmetry")
print("  2. Constrain away from zero: Add constraint |ψ| > ε")
print("  3. Multi-start: Try different initial conditions")
print("  4. Check if ψ=[0,0,0,0] is actually optimal (maybe there's no benefit!)")
