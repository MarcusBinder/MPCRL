"""
Minimal Wind Farm MPC Example - 2 Turbines
===========================================

Simplest possible demonstration:
- Only 2 turbines
- Very close spacing (3D) for fast wake response
- Short run to show immediate optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from nmpc_windfarm_acados_fixed import AcadosYawMPC, Farm, Wind, Limits, MPCConfig

print("=" * 70)
print("Minimal Example: 2 Turbines, Wake Steering Optimization")
print("=" * 70)

# Minimal setup: 2 turbines, close together
D = 178.0
x = np.array([0.0, 3*D])  # Just 2 turbines, 3D spacing (fast wake response)
y = np.array([0.0, 0.0])
farm = Farm(x=x, y=y, D=D)

wind = Wind(U=8.0, theta=270.0, TI=0.06)
limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.8)  # Faster rate

cfg = MPCConfig(
    dt=5.0,   # Faster timestep
    N_h=8,    # Shorter horizon
    lam_move=5.0,
)

print(f"\nSetup:")
print(f"  2 turbines, {3*D:.0f}m apart ({3*D/wind.U:.1f}s wake travel time)")
print(f"  Wind: {wind.U} m/s from west")
print(f"  Goal: Yaw turbine 0 to deflect wake away from turbine 1")

# Create and run
controller = AcadosYawMPC(farm, wind, limits, cfg)
controller.set_state(np.zeros(2))

print("\nOptimizing...")
results = []
for t in range(15):
    res = controller.step()
    res['step'] = t
    results.append(res)

# Show results
print(f"\n{'Step':>4} | Turbine 0 | Turbine 1 | Power [MW]")
print("-" * 50)
for r in [results[0], results[4], results[9], results[-1]]:
    print(f"{r['step']:4d} | {r['psi'][0]:9.1f} | {r['psi'][1]:9.1f} | {r['power']/1e6:10.3f}")

initial_power = results[0]['power'] / 1e6
final_power = results[-1]['power'] / 1e6
print("-" * 50)
print(f"\nResults:")
print(f"  Initial power: {initial_power:.3f} MW")
print(f"  Final power:   {final_power:.3f} MW")
print(f"  Gain:          {(final_power/initial_power - 1)*100:+.1f}%")
print(f"\n  Turbine 0 yaw: {results[-1]['psi'][0]:.1f}° (deflects wake)")
print(f"  Turbine 1 yaw: {results[-1]['psi'][1]:.1f}° (stays aligned)")

# Simple plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

# Power
steps = [r['step'] for r in results]
powers = [r['power']/1e6 for r in results]
ax1.plot(steps, powers, 'o-', linewidth=2, markersize=7, color='#2ca02c')
ax1.axhline(initial_power, color='gray', linestyle='--', alpha=0.5,
            label=f'Initial: {initial_power:.3f} MW')
ax1.set_xlabel('Step')
ax1.set_ylabel('Farm Power [MW]')
ax1.set_title('Power Increases as Wake is Deflected', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Yaw angles
yaw_array = np.array([r['psi'] for r in results])
ax2.plot(steps, yaw_array[:, 0], 'o-', linewidth=2, label='Turbine 0 (upstream)', color='#1f77b4')
ax2.plot(steps, yaw_array[:, 1], 'o-', linewidth=2, label='Turbine 1 (downstream)', color='#ff7f0e')
ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel('Step')
ax2.set_ylabel('Yaw Angle [deg]')
ax2.set_title('Upstream Turbine Steers Wake', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('minimal_example.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: minimal_example.png")

print("\n" + "=" * 70)
print("Explanation:")
print("  1. Turbine 0 (upstream) yaws to deflect its wake sideways")
print("  2. This reduces wake interference on Turbine 1 (downstream)")
print("  3. Turbine 1 sees cleaner wind → produces more power")
print("  4. Total farm power increases!")
print("=" * 70)
