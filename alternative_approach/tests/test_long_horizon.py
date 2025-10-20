"""
Test MPC with Long Horizon
===========================

The wake delay is 330s but our horizon was only 100s!
Test with horizon >> delay to see if MPC can find optimal yaw.
"""

import numpy as np
from nmpc_windfarm_acados_fixed import AcadosYawMPC, Farm, Wind, Limits, MPCConfig

print("=" * 70)
print("MPC with Extended Horizon")
print("=" * 70)

D = 178.0
x = np.array([0.0, 5*D, 10*D, 15*D])
y = np.zeros_like(x)
farm = Farm(x=x, y=y, D=D)

wind = Wind(U=8.0, theta=270.0, TI=0.06)
limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.5)

# Test different horizons
test_configs = [
    (10, "Short (100s) - Original"),
    (40, "Medium (400s) - Covers delay"),
    (60, "Long (600s) - 2× delay"),
]

print(f"\nWake delay: 330 seconds (33 steps × 10s)")
print(f"Optimal yaw from grid search: [-25°, -20°, -20°, 0°] → +15% gain\n")

for N_h, desc in test_configs:
    print("-" * 70)
    print(f"Testing N_h={N_h} ({N_h*10}s): {desc}")
    print("-" * 70)

    cfg = MPCConfig(
        dt=10.0,
        N_h=N_h,
        lam_move=10.0,
        trust_region_weight=1e4,
    )

    controller = AcadosYawMPC(farm, wind, limits, cfg)
    controller.set_state(np.zeros(4))

    # Run for more steps with longer horizon
    n_steps = max(50, N_h + 10)
    print(f"Running {n_steps} steps...")

    results = []
    for t in range(n_steps):
        res = controller.step()
        res['step'] = t
        results.append(res)

        # Print progress every 10 steps
        if t % 10 == 0:
            print(f"  Step {t:2d}: yaw={[f'{y:.1f}' for y in res['psi']]}, P={res['power']/1e6:.4f} MW")

    # Summary
    initial_power = results[0]['power'] / 1e6
    final_power = results[-1]['power'] / 1e6
    final_yaw = results[-1]['psi']
    gain = (final_power / initial_power - 1) * 100

    print(f"\nResults for N_h={N_h}:")
    print(f"  Final yaw: [{', '.join(f'{y:.1f}' for y in final_yaw)}]°")
    print(f"  Final power: {final_power:.4f} MW")
    print(f"  Gain: {gain:+.2f}%")
    print(f"  Avg solve time: {np.mean([r['solve_time'] for r in results])*1000:.2f} ms")

print("\n" + "=" * 70)
print("SUMMARY:")
print("  Optimal (grid search): [-25°, -20°, -20°, 0°] → +15.1% gain")
print("  Question: Can MPC reach this with long enough horizon?")
print("=" * 70)
