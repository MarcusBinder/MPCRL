"""
Hybrid Global-Local MPC
========================

Two-layer architecture:
1. Global optimizer: Finds optimal yaw for current wind (slow, every 1-10 min)
2. Local MPC: Tracks optimal with constraints & disturbance rejection (fast, every 5-10s)

This approach achieves BOTH:
- ✅ True optimal yaw angles (~20°, +15% power)
- ✅ Fast real-time control with constraints
"""

import numpy as np
from scipy.optimize import differential_evolution
from nmpc_windfarm_acados_fixed import (
    AcadosYawMPC, Farm, Wind, Limits, MPCConfig,
    build_pywake_model, pywake_farm_power
)

print("=" * 70)
print("Hybrid Global-Local MPC Demonstration")
print("=" * 70)

# Setup
D = 178.0
x = np.array([0.0, 5*D, 10*D, 15*D])
y = np.zeros_like(x)
farm = Farm(x=x, y=y, D=D)
wind = Wind(U=8.0, theta=270.0, TI=0.06)
limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.5)

# ============================================================================
# LAYER 1: GLOBAL OPTIMIZER (finds optimal yaw for current wind)
# ============================================================================

print("\n" + "="*70)
print("LAYER 1: Global Optimization (finding optimal yaw)")
print("="*70)

wf_model, layout = build_pywake_model(x, y, D, ti=wind.TI)

def power_objective(yaw):
    """Objective for global optimizer (negative power to minimize)."""
    return -pywake_farm_power(wf_model, layout, wind.U, wind.theta, yaw) / 1e6

# Run global optimization
print("\nRunning global optimizer (differential evolution)...")
print("This would typically run in background every 1-10 minutes as wind changes")

result = differential_evolution(
    power_objective,
    bounds=[(-25, 25), (-25, 25), (-25, 25), (0, 0)],  # Last turbine fixed
    maxiter=30,
    seed=42,
    workers=1,  # Can parallelize for speed
    atol=0.01,
    updating='deferred'  # Faster
)

optimal_yaw_global = result.x
optimal_power = -result.fun  # Convert back to positive

print(f"\n✓ Global optimizer found:")
print(f"  Optimal yaw: [{', '.join(f'{y:.1f}' for y in optimal_yaw_global)}]°")
print(f"  Expected power: {optimal_power:.4f} MW")
print(f"  Gain vs baseline: {(optimal_power/4.9415 - 1)*100:.1f}%")

# ============================================================================
# LAYER 2: LOCAL MPC (tracks optimal with fast control)
# ============================================================================

print("\n" + "="*70)
print("LAYER 2: Local MPC Tracking (fast real-time control)")
print("="*70)

# Configure MPC for PURE TRACKING mode
# target_weight > 1e6 activates pure tracking (no power gradient computation)
cfg = MPCConfig(
    dt=10.0,
    N_h=10,
    lam_move=10.0,
    trust_region_weight=1e4,
    target_weight=1e7,  # PURE tracking mode (>1e6 skips power gradient)
    direction_bias=0.0,  # Not needed when tracking
)

controller = AcadosYawMPC(farm, wind, limits, cfg)

# Set the target from global optimizer
controller.yaw_target = optimal_yaw_global
print(f"\n✓ MPC target set to: [{', '.join(f'{y:.1f}' for y in optimal_yaw_global)}]°")

# Start from zero (simulating startup)
controller.set_state(np.zeros(4))

print("\nRunning local MPC to track optimal...")
print("-" * 70)
print(f"{'Step':>4} | {'Current Yaw [°]':>30} | {'Power [MW]':>12} | {'Distance to Target':>18}")
print("-" * 70)

results = []
for t in range(40):
    res = controller.step()
    res['step'] = t
    results.append(res)

    # Compute distance to target
    dist_to_target = np.linalg.norm(res['psi'] - optimal_yaw_global)

    if t % 5 == 0:
        yaw_str = ', '.join(f'{y:5.1f}' for y in res['psi'])
        print(f"{t:4d} | [{yaw_str}] | {res['power']/1e6:12.4f} | {dist_to_target:15.2f}°")

print("-" * 70)

# Analysis
final_yaw = results[-1]['psi']
final_power = results[-1]['power'] / 1e6
initial_power = results[0]['power'] / 1e6
convergence_step = None

for i, r in enumerate(results):
    if np.linalg.norm(r['psi'] - optimal_yaw_global) < 1.0:  # Within 1° of target
        convergence_step = i
        break

print("\n" + "="*70)
print("RESULTS:")
print("="*70)
print(f"Global optimal:     [{', '.join(f'{y:.0f}' for y in optimal_yaw_global)}]° → {optimal_power:.4f} MW")
print(f"MPC final:          [{', '.join(f'{y:.1f}' for y in final_yaw)}]° → {final_power:.4f} MW")
print(f"")
print(f"Initial power:      {initial_power:.4f} MW")
print(f"Final power:        {final_power:.4f} MW")
print(f"Gain:               {(final_power/initial_power - 1)*100:+.1f}%")

if convergence_step:
    print(f"Converged to target in {convergence_step} steps ({convergence_step*10}s)")
else:
    print(f"Distance to target: {np.linalg.norm(final_yaw - optimal_yaw_global):.2f}°")

print(f"")
print(f"Avg solve time:     {np.mean([r['solve_time'] for r in results])*1000:.2f} ms")
print("="*70)

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*70)
print("COMPARISON WITH PREVIOUS APPROACHES:")
print("="*70)
print(f"{'Approach':<30} | {'Final Yaw':<25} | {'Power':<12} | {'Gain':>8}")
print("-"*70)
print(f"{'Baseline (0°)':<30} | {'[0, 0, 0, 0]':<25} | {'4.9415 MW':<12} | {'+0.0%':>8}")
print(f"{'Grid Search Optimal':<30} | {'[-25, -20, -20, 0]':<25} | {'5.6860 MW':<12} | {'+15.1%':>8}")
print(f"{'Gradient MPC (cold start)':<30} | {'[-5, -5, -5, 0]':<25} | {'4.94 MW':<12} | {'-0.2%':>8}")
print(f"{'Hybrid (this approach)':<30} | {str([f'{y:.0f}' for y in final_yaw]):<25} | {f'{final_power:.4f} MW':<12} | {f'{(final_power/initial_power-1)*100:+.1f}%':>8}")
print("="*70)

print("\n" + "="*70)
print("ARCHITECTURE SUMMARY:")
print("="*70)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ GLOBAL OPTIMIZER (Layer 1)                                      │
│  • Runs: Every 1-10 minutes (when wind changes)                │
│  • Method: Differential Evolution, PSO, Grid Search, etc.      │
│  • Output: Optimal yaw setpoint for current wind               │
│  • Runtime: 10-60 seconds (runs in background)                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ yaw_target
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ LOCAL MPC (Layer 2)                                             │
│  • Runs: Every 5-10 seconds (real-time)                        │
│  • Method: Gradient-based MPC (existing acados implementation) │
│  • Task: Track yaw_target with constraints                     │
│  • Handles: Rate limits, bounds, disturbances, smoothness      │
│  • Runtime: <1 millisecond per step                            │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n✅ This approach achieves BOTH optimal performance AND real-time control!")
print("="*70)
