"""
Simple Wind Farm MPC Example
============================

This script demonstrates the MPC controller optimizing yaw angles
to increase wind farm power production.
"""

import numpy as np
import matplotlib.pyplot as plt
from nmpc_windfarm_acados_fixed import AcadosYawMPC, Farm, Wind, Limits, MPCConfig

print("=" * 70)
print("Wind Farm Yaw Optimization - Simple Example")
print("=" * 70)

# Setup wind farm: 4 turbines in a row
D = 178.0  # Rotor diameter (meters)
x = np.array([0.0, 5*D, 10*D, 15*D])  # Turbine positions (5D spacing)
y = np.zeros_like(x)
farm = Farm(x=x, y=y, D=D)

# Wind conditions
wind = Wind(U=8.0, theta=270.0, TI=0.06)  # 8 m/s from west

# Control limits
limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.5)

# MPC configuration
cfg = MPCConfig(
    dt=10.0,           # 10 second time steps
    N_h=10,            # 10-step horizon (100 seconds)
    lam_move=10.0,     # Regularization
)

print(f"\nSetup:")
print(f"  Turbines: 4 in a row, spaced {5*D:.0f}m apart")
print(f"  Wind: {wind.U} m/s from {wind.theta}° (west)")
print(f"  Time step: {cfg.dt}s, Horizon: {cfg.N_h} steps")

# Create controller
print("\nInitializing MPC controller...")
controller = AcadosYawMPC(farm, wind, limits, cfg)

# Start from zero yaw (all turbines aligned with wind)
controller.set_state(np.zeros(4))

print("\nRunning optimization...")
print("-" * 70)
print(f"{'Step':>4} | {'Yaw Angles [deg]':>25} | {'Power [MW]':>12} | {'Time [ms]':>10}")
print("-" * 70)

# Run for 50 steps to see delayed power effects
# (wake delay is 33 steps, so need ~40+ steps to see full impact)
history = []
for t in range(50):
    result = controller.step()
    result['step'] = t  # Add step number
    history.append(result)

    # Print every 10 steps
    if t % 10 == 0 or t == 49:
        yaw_str = ", ".join(f"{y:5.1f}" for y in result['psi'])
        print(f"{t:4d} | [{yaw_str}] | {result['power']/1e6:12.3f} | {result['solve_time']*1000:10.2f}")

print("-" * 70)

# Summary
initial_power = history[0]['power'] / 1e6
final_power = history[-1]['power'] / 1e6
gain_pct = (final_power / initial_power - 1) * 100
final_yaw = history[-1]['psi']

print(f"\n{'Results Summary:':^70}")
print("=" * 70)
print(f"Initial power:     {initial_power:.3f} MW")
print(f"Final power:       {final_power:.3f} MW")
print(f"Power gain:        {gain_pct:+.2f}%")
print(f"Final yaw angles:  [{', '.join(f'{y:.1f}' for y in final_yaw)}]°")
print(f"Avg solve time:    {np.mean([h['solve_time'] for h in history])*1000:.2f} ms")
print("=" * 70)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot power over time
steps = [h['step'] for h in history]
powers = [h['power']/1e6 for h in history]
ax1.plot(steps, powers, 'o-', linewidth=2, markersize=6, color='#2ca02c')
ax1.axhline(initial_power, color='gray', linestyle='--', alpha=0.5,
            label=f'Initial: {initial_power:.3f} MW')
ax1.set_xlabel('Time Step', fontsize=11)
ax1.set_ylabel('Farm Power [MW]', fontsize=11)
ax1.set_title('Power Production Over Time', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot yaw angles over time
yaw_trajectories = np.array([h['psi'] for h in history])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i in range(4):
    ax2.plot(steps, yaw_trajectories[:, i], 'o-', label=f'Turbine {i}',
             linewidth=2, markersize=5, color=colors[i])
ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel('Time Step', fontsize=11)
ax2.set_ylabel('Yaw Angle [deg]', fontsize=11)
ax2.set_title('Yaw Angle Trajectories', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_example_result.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved: simple_example_result.png")

# Show interpretation
print("\nInterpretation:")
print("  • Turbines 0-2 (upstream) deflect their yaw to steer wakes away from downstream turbines")
print("  • Turbine 3 (last) stays at 0° since there are no turbines behind it")
print("  • Wake deflection reduces wake interference → higher total power output")
print("  • MPC finds optimal yaw angles automatically!")
