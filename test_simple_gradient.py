"""
Simple test: Check if optimizer responds to a strong gradient.

This bypasses all the wind farm complexity and just tests if the
acados solver can follow a simple gradient.
"""

import numpy as np
from nmpc_windfarm_acados_fixed import (
    AcadosYawMPC, Farm, Wind, Limits, MPCConfig,
    build_pywake_model, pywake_farm_power
)

# Setup
np.random.seed(42)
D = 178.0
x = np.array([0.0, 5*D, 10*D, 15*D])
y = np.zeros_like(x)
farm = Farm(x=x, y=y, D=D)

wind = Wind(U=8.0, theta=0.0)
limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=1.0)  # High rate limit
cfg = MPCConfig(dt=10.0, N_h=10, lam_move=0.01)  # Low move penalty

# Create controller
print("Creating controller...")
controller = AcadosYawMPC(farm, wind, limits, cfg)

# Set initial yaws to zero
controller.psi_current = np.zeros(controller.N)

print(f"\nInitial yaws: {controller.psi_current}")

# Manually set a LARGE gradient pointing in +direction for all turbines
# This should definitely cause movement if the optimizer works
manual_gradient = np.array([10000.0, 10000.0, 10000.0, 10000.0])

print(f"Manual gradient (huge!): {manual_gradient}")
print(f"Gradient norm: {np.linalg.norm(manual_gradient):.2e}")

# Manually inject this gradient
print("\nForcing optimizer to use manual gradient...")

# Update initial condition
controller.solver.set(0, 'lbx', controller.psi_current)
controller.solver.set(0, 'ubx', controller.psi_current)

# Scale gradient
grad_norm = np.linalg.norm(manual_gradient)
grad_scaled = manual_gradient / (grad_norm / 10.0)
print(f"Scaled gradient norm: {np.linalg.norm(grad_scaled):.2f}")

# Set gradient parameters
for k in range(cfg.N_h):
    controller.solver.set(k, 'p', grad_scaled)
controller.solver.set(cfg.N_h, 'p', grad_scaled)

# Warm start at current position
for k in range(cfg.N_h + 1):
    controller.solver.set(k, 'x', controller.psi_current)

# Solve
print("\nSolving...")
status = controller.solver.solve()
print(f"Solver status: {status}")

# Extract solution
psi_plan = np.zeros((cfg.N_h, controller.N))
u_plan = np.zeros((cfg.N_h, controller.N))

for k in range(cfg.N_h):
    psi_plan[k, :] = controller.solver.get(k, 'x')
    u_plan[k, :] = controller.solver.get(k, 'u')

print("\nOptimal yaw trajectory:")
for k in range(min(5, cfg.N_h)):
    print(f"  k={k}: ψ={np.round(psi_plan[k,:], 2)}, u={np.round(u_plan[k,:], 3)} deg/s")

print(f"\nFirst step control: u={np.round(u_plan[0,:], 3)} deg/s")
print(f"Expected change: Δψ = u*dt = {np.round(u_plan[0,:] * cfg.dt, 2)}°")
print(f"Next state prediction: ψ[1] = {np.round(psi_plan[0,:] + u_plan[0,:]*cfg.dt, 2)}°")
print(f"Actual ψ[1] from solver: {np.round(psi_plan[1,:], 2)}°")

# Check if there's ANY movement
if np.allclose(u_plan, 0, atol=1e-6):
    print("\n❌ PROBLEM: All controls are zero! Optimizer isn't responding to gradient.")
else:
    print("\n✓ Controls are non-zero - optimizer is working!")
    print(f"Max control magnitude: {np.max(np.abs(u_plan)):.3f} deg/s")

# Check cost improvement
print("\n" + "="*70)
print("Cost Analysis:")
print("="*70)

# Cost at initial guess (all zeros, no movement)
cost_initial = 0.0
for k in range(cfg.N_h):
    cost_initial += -grad_scaled.T @ psi_plan[0,:] + (cfg.lam_move/2) * 0**2
print(f"Cost at initial guess (no movement): {cost_initial:.2f}")

# Cost at optimal solution
cost_optimal = 0.0
for k in range(cfg.N_h):
    cost_optimal += -grad_scaled.T @ psi_plan[k,:] + (cfg.lam_move/2) * np.sum(u_plan[k,:]**2)
print(f"Cost at optimal solution: {cost_optimal:.2f}")

if cost_optimal < cost_initial:
    print(f"✓ Optimizer improved cost by {cost_initial - cost_optimal:.2f}")
else:
    print(f"❌ Optimizer made cost worse or didn't improve it")

# Check gradient direction
gradient_term_at_final = -grad_scaled.T @ psi_plan[-1,:]
print(f"\nGradient term at final state: {gradient_term_at_final:.2f}")
print(f"(More negative = better for maximizing power)")
