"""
Demo with HIGH move penalty to prevent oscillations.

Based on debugging results:
- ψ=[0,0,0,0] is the GLOBAL MAXIMUM
- Low move penalty causes wandering and oscillations
- Solution: Increase λ from 0.01 to 5.0 (500x!)
"""

import numpy as np
import sys
from datetime import datetime
from nmpc_windfarm_acados_fixed import (
    AcadosYawMPC, Farm, Wind, Limits, MPCConfig
)

# Create log file
log_filename = f"mpc_fixed_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

class TeeOutput:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = file
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

with open(log_filename, 'w') as log_file:
    sys.stdout = TeeOutput(log_file)

    print("="*70)
    print("Wind Farm MPC with HIGH Move Penalty (Fixed)")
    print("="*70)
    print()
    print(f"Logging to: {log_filename}")
    print()

    # Setup
    np.random.seed(42)
    D = 178.0
    spacing = 7 * D
    x = np.array([0.0, spacing, 2*spacing, 3*spacing])
    y = np.zeros_like(x)
    farm = Farm(x=x, y=y, D=D)

    wind = Wind(U=8.0, theta=0.0)
    limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.5)  # 0.5 deg/s rate limit

    # HIGH MOVE PENALTY to prevent oscillations
    cfg = MPCConfig(dt=15.0, N_h=20, lam_move=5.0)  # Was 0.01, now 5.0 (500x!)

    print("Configuration:")
    print(f"  Turbine spacing: {spacing/D:.1f}D = {spacing:.0f}m")
    print(f"  Wind speed: {wind.U} m/s")
    print(f"  Time step: {cfg.dt}s")
    print(f"  Horizon: {cfg.N_h} steps = {cfg.dt * cfg.N_h:.0f}s")
    print(f"  Move penalty: λ = {cfg.lam_move:.2f} ← HIGH to prevent wandering")
    print()

    # Create controller
    print("Building MPC controller...")
    controller = AcadosYawMPC(farm, wind, limits, cfg)
    print("Controller ready!")
    print()

    # Start from RANDOM yaws (away from optimum)
    psi_initial = np.random.uniform(-5, 5, size=controller.N)
    controller.psi_current = psi_initial.copy()

    print(f"Initial yaws: {np.round(psi_initial, 1)}°")
    print("(Starting away from optimal ψ=[0,0,0,0] to test convergence)")
    print()

    # Run MPC
    N_steps = 30
    print(f"Running MPC for {N_steps} steps...")
    print()

    results = {
        'time': [],
        'yaws': [],
        'power': [],
        'power_MW': [],
        'grad_norm': [],
    }

    for t in range(N_steps):
        info = controller.step()

        psi_next = info['psi']
        grad_norm = np.linalg.norm(info['grad_P'])

        results['time'].append(t)
        results['yaws'].append(psi_next.copy())
        results['power'].append(info['power'])
        results['power_MW'].append(info['power'] / 1e6)
        results['grad_norm'].append(grad_norm)

        # Print status
        print(f"t={t:02d}, ψ={np.round(psi_next, 1)}, "
              f"P={info['power']/1e6:.3f} MW, "
              f"solve={info['solve_time']*1000:.1f}ms, "
              f"|∇P|={grad_norm:.2e}")

    print()
    print("="*70)
    print("Results")
    print("="*70)

    P_initial = results['power_MW'][0]
    P_final = results['power_MW'][-1]
    P_max = max(results['power_MW'])
    P_min = min(results['power_MW'])

    psi_initial = results['yaws'][0]
    psi_final = results['yaws'][-1]

    print(f"\nPower:")
    print(f"  Initial: {P_initial:.6f} MW")
    print(f"  Final:   {P_final:.6f} MW")
    print(f"  Max:     {P_max:.6f} MW")
    print(f"  Min:     {P_min:.6f} MW")
    print(f"  Gain:    {(P_final - P_initial)*1000:+.1f} kW ({(P_final/P_initial - 1)*100:+.2f}%)")

    print(f"\nYaw angles:")
    print(f"  Initial: {np.round(psi_initial, 2)}°")
    print(f"  Final:   {np.round(psi_final, 2)}°")
    print(f"  Change:  {np.round(psi_final - psi_initial, 2)}°")

    # Check for oscillations
    print()
    print("="*70)
    print("Oscillation Analysis")
    print("="*70)

    # Look for repeating patterns in last 10 steps
    last_10_yaws = results['yaws'][-10:]

    # Check if any two consecutive yaws are repeating
    oscillating = False
    for i in range(len(last_10_yaws) - 2):
        if np.allclose(last_10_yaws[i], last_10_yaws[i+2], atol=0.1):
            oscillating = True
            break

    if oscillating:
        print("\n❌ OSCILLATIONS DETECTED in last 10 steps")
        print("   System is alternating between states")
    else:
        print("\n✓ NO OSCILLATIONS - System is stable")

    # Check convergence
    yaw_changes_last_5 = np.abs(np.diff(results['yaws'][-5:], axis=0))
    max_change = np.max(yaw_changes_last_5)

    if max_change < 0.5:
        print(f"✓ CONVERGED - Max yaw change in last 5 steps: {max_change:.3f}°")
    else:
        print(f"⚠ Not converged - Max yaw change: {max_change:.3f}°")

    # Check if converged to ψ≈0
    if np.allclose(psi_final, 0, atol=2.0):
        print(f"✓ CONVERGED TO ψ≈[0,0,0,0] - This is the optimal configuration!")
    else:
        print(f"⚠ Converged to ψ={np.round(psi_final, 1)}° (not zero)")

    # Check power trend
    power_changes = np.diff(results['power_MW'])
    decreases = np.sum(power_changes < -1e-6)

    if decreases > len(power_changes) / 4:
        print(f"⚠ Power decreased in {decreases}/{len(power_changes)} steps")
    else:
        print(f"✓ Power stable/increasing ({decreases} decreases in {len(power_changes)} steps)")

    print()
    print("="*70)
    print("Comparison to Original (Low Penalty)")
    print("="*70)

    print("\nOriginal MPC (λ=0.01):")
    print("  ✗ Oscillated between ψ=[6.7,-4.8,-6.1,-6.9] and ψ=[-0.8,2.7,1.4,0.6]")
    print("  ✗ Power decreased from 13.961 to 13.828 MW")
    print("  ✗ Never converged")

    print("\nFixed MPC (λ=5.0):")
    if not oscillating and max_change < 0.5:
        print("  ✓ No oscillations")
        print("  ✓ Converged to steady state")
        if np.allclose(psi_final, 0, atol=2.0):
            print("  ✓ Found correct optimum ψ≈[0,0,0,0]")
        print("\n  → FIX SUCCESSFUL!")
    else:
        print("  ⚠ Still has issues, may need even higher penalty")

    print()
    print("="*70)
    print(f"Log saved to: {log_filename}")
    print("="*70)

    sys.stdout = sys.__stdout__

print(f"\nLog file created: {log_filename}")
print("Check the results to verify the fix worked!")
