"""
MPC with Warm Start - Demonstration
====================================

Shows that MPC works well when started near optimal,
but struggles when started from zero (linearization limitation).
"""

import numpy as np
import matplotlib.pyplot as plt
from nmpc_windfarm_acados_fixed import AcadosYawMPC, Farm, Wind, Limits, MPCConfig

print("=" * 70)
print("MPC Warm Start Comparison")
print("=" * 70)

# Setup
D = 178.0
x = np.array([0.0, 5*D, 10*D, 15*D])
y = np.zeros_like(x)
farm = Farm(x=x, y=y, D=D)
wind = Wind(U=8.0, theta=270.0, TI=0.06)
limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.5)

cfg = MPCConfig(dt=10.0, N_h=15, lam_move=10.0, direction_bias=1e5)

print(f"\nOptimal yaw (from grid search): [-25°, -20°, -20°, 0°] → +15.1% gain\n")

# Test 1: Start from zero (cold start)
print("-" * 70)
print("Test 1: Cold Start (from 0°)")
print("-" * 70)

controller1 = AcadosYawMPC(farm, wind, limits, cfg)
controller1.set_state(np.zeros(4))

results_cold = []
for t in range(30):
    res = controller1.step()
    res['step'] = t
    results_cold.append(res)
    if t % 10 == 0:
        print(f"  Step {t}: yaw={[f'{y:.1f}' for y in res['psi']]}, P={res['power']/1e6:.4f} MW")

cold_final_yaw = results_cold[-1]['psi']
cold_final_power = results_cold[-1]['power'] / 1e6
cold_initial_power = results_cold[0]['power'] / 1e6
cold_gain = (cold_final_power / cold_initial_power - 1) * 100

print(f"\nCold start result:")
print(f"  Final yaw: [{', '.join(f'{y:.1f}' for y in cold_final_yaw)}]°")
print(f"  Final power: {cold_final_power:.4f} MW ({cold_gain:+.2f}% gain)")

# Test 2: Warm start near optimal
print("\n" + "-" * 70)
print("Test 2: Warm Start (from -18°)")
print("-" * 70)

controller2 = AcadosYawMPC(farm, wind, limits, cfg)
warm_start = np.array([-18.0, -15.0, -15.0, 0.0])  # Near optimal
controller2.set_state(warm_start)

results_warm = []
for t in range(30):
    res = controller2.step()
    res['step'] = t
    results_warm.append(res)
    if t % 10 == 0:
        print(f"  Step {t}: yaw={[f'{y:.1f}' for y in res['psi']]}, P={res['power']/1e6:.4f} MW")

warm_final_yaw = results_warm[-1]['psi']
warm_final_power = results_warm[-1]['power'] / 1e6
warm_initial_power = results_warm[0]['power'] / 1e6
warm_gain = (warm_final_power / warm_initial_power - 1) * 100

print(f"\nWarm start result:")
print(f"  Final yaw: [{', '.join(f'{y:.1f}' for y in warm_final_yaw)}]°")
print(f"  Final power: {warm_final_power:.4f} MW ({warm_gain:+.2f}% from warm start)")

# Compute gain vs baseline (0°)
baseline_power = cold_initial_power
warm_vs_baseline = (warm_final_power / baseline_power - 1) * 100

print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)
print(f"Baseline (0° yaw):        {baseline_power:.4f} MW")
print(f"Optimal (grid search):    5.6860 MW (+15.1% vs baseline)")
print(f"")
print(f"Cold start MPC final:     {cold_final_power:.4f} MW ({cold_gain:+.1f}% vs baseline)")
print(f"Warm start MPC final:     {warm_final_power:.4f} MW ({warm_vs_baseline:+.1f}% vs baseline)")
print("=" * 70)

if warm_vs_baseline > 10:
    print("\n✅ Warm start MPC achieves >10% gain (good performance!)")
else:
    print(f"\n⚠️  Even warm start only achieves {warm_vs_baseline:.1f}% (still suboptimal)")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Power trajectories
steps_cold = [r['step'] for r in results_cold]
powers_cold = [r['power']/1e6 for r in results_cold]
steps_warm = [r['step'] for r in results_warm]
powers_warm = [r['power']/1e6 for r in results_warm]

ax1.plot(steps_cold, powers_cold, 'o-', label='Cold start (from 0°)', linewidth=2, color='#1f77b4')
ax1.plot(steps_warm, powers_warm, 'o-', label='Warm start (from -18°)', linewidth=2, color='#ff7f0e')
ax1.axhline(5.686, color='green', linestyle='--', label='Optimal (grid search)', alpha=0.7)
ax1.axhline(baseline_power, color='gray', linestyle='--', label='Baseline (0°)', alpha=0.5)
ax1.set_xlabel('Step')
ax1.set_ylabel('Farm Power [MW]')
ax1.set_title('Power vs Time', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Yaw trajectories
yaw_cold = np.array([r['psi'] for r in results_cold])
yaw_warm = np.array([r['psi'] for r in results_warm])

for i in range(3):  # Only first 3 turbines
    ax2.plot(steps_cold, yaw_cold[:, i], '-', alpha=0.5, label=f'T{i} cold' if i==0 else None, color='#1f77b4')
    ax2.plot(steps_warm, yaw_warm[:, i], '-', linewidth=2, label=f'T{i} warm' if i==0 else None, color='#ff7f0e')

ax2.axhline(-20, color='green', linestyle='--', label='Optimal (~-20°)', alpha=0.7)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel('Step')
ax2.set_ylabel('Yaw Angle [deg]')
ax2.set_title('Yaw Trajectories (Turbines 0-2)', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('warm_start_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: warm_start_comparison.png")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("  • MPC with cold start gets stuck near 0° (linearization limitation)")
print("  • MPC with warm start can refine near-optimal solution")
print("  • Best practice: Use global search + MPC for tracking/refinement")
print("=" * 70)
