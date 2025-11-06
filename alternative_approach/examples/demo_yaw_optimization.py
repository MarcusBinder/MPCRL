"""
Wind Farm Yaw Optimization Demo

This demo shows the MPC actually optimizing yaw angles by:
1. Using closer turbine spacing (smaller delays)
2. Starting with perturbed initial yaws (non-zero gradients)
3. Using a longer horizon to capture wake effects
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import nmpc_windfarm_acados_fixed
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from nmpc_windfarm_acados_fixed import AcadosYawMPC, Farm, Wind, Limits, MPCConfig

def plot_results(history, title="MPC Results"):
    """Plot MPC optimization results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    steps = [h['step'] for h in history]
    powers = [h['power']/1e6 for h in history]
    psi = np.array([h['psi'] for h in history])
    solve_times = [h['solve_time']*1000 for h in history]
    grad_norms = [np.linalg.norm(h['grad_P']) for h in history]

    # Power over time
    ax = axes[0, 0]
    ax.plot(steps, powers, 'o-', linewidth=2, markersize=5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Farm Power [MW]')
    ax.set_title('Power Production')
    ax.grid(True, alpha=0.3)
    ax.axhline(powers[0], color='gray', linestyle='--', alpha=0.5, label=f'Initial: {powers[0]:.3f} MW')
    ax.legend()

    # Yaw angles over time
    ax = axes[0, 1]
    N_turbines = psi.shape[1]
    for i in range(N_turbines):
        ax.plot(steps, psi[:, i], 'o-', label=f'Turbine {i}', linewidth=2, markersize=4)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Yaw Angle [deg]')
    ax.set_title('Yaw Angle Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)

    # Solve times
    ax = axes[1, 0]
    ax.plot(steps, solve_times, 'o-', linewidth=2, markersize=5, color='green')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Solve Time [ms]')
    ax.set_title('Optimization Time per Step')
    ax.grid(True, alpha=0.3)
    ax.axhline(np.mean(solve_times), color='gray', linestyle='--',
               label=f'Avg: {np.mean(solve_times):.2f}ms')
    ax.legend()

    # Gradient norm
    ax = axes[1, 1]
    ax.semilogy(steps, grad_norms, 'o-', linewidth=2, markersize=5, color='red')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Gradient Norm |∇P|')
    ax.set_title('Power Gradient Magnitude')
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def demo_1_closer_spacing():
    """Demo 1: Closer turbine spacing (smaller delays)."""
    print("\n" + "="*70)
    print("DEMO 1: Closer Turbine Spacing")
    print("="*70)

    np.random.seed(42)
    D = 178.0

    # Closer spacing: 5D instead of 7D
    x = np.array([0.0, 5*D, 10*D, 15*D])
    y = np.zeros_like(x)
    farm = Farm(x=x, y=y, D=D)

    wind = Wind(U=8.0, theta=270.0)
    limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.5)
    # Use dt=15s: N_h=20 gives 300s horizon, enough to cover delays
    cfg = MPCConfig(dt=15.0, N_h=20, lam_move=0.1)

    print(f"\nSetup:")
    print(f"  Turbine spacing: 5D = {5*D:.0f}m")
    print(f"  Wind speed: {wind.U} m/s")
    print(f"  Horizon: {cfg.N_h} steps = {cfg.N_h * cfg.dt:.0f}s")

    controller = AcadosYawMPC(farm, wind, limits, cfg)

    # Start with small random perturbation
    init_yaws = np.random.uniform(-3, 3, controller.N)
    controller.set_state(init_yaws)
    print(f"  Initial yaws: {np.round(init_yaws, 1)}°")

    history = controller.run(n_steps=15, verbose=True)

    fig = plot_results(history, "Demo 1: Closer Spacing (5D)")
    plt.savefig('demo1_closer_spacing.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: demo1_closer_spacing.png")

    return history


def demo_2_longer_horizon():
    """Demo 2: Longer horizon to capture wake effects."""
    print("\n" + "="*70)
    print("DEMO 2: Longer Horizon")
    print("="*70)

    np.random.seed(42)
    D = 178.0

    # Original spacing but longer horizon
    x = np.array([0.0, 7*D, 14*D, 21*D])
    y = np.zeros_like(x)
    farm = Farm(x=x, y=y, D=D)

    wind = Wind(U=8.0, theta=270.0)
    limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.3)
    # Use dt=30s: N_h=20 gives 600s horizon, safe for HPIPM
    # Larger dt avoids memory issues while keeping long horizon
    cfg = MPCConfig(dt=30.0, N_h=20, lam_move=0.2, qp_solver="PARTIAL_CONDENSING_HPIPM")

    print(f"\nSetup:")
    print(f"  Turbine spacing: 7D = {7*D:.0f}m")
    print(f"  Wind speed: {wind.U} m/s")
    print(f"  Horizon: {cfg.N_h} steps = {cfg.N_h * cfg.dt:.0f}s")

    controller = AcadosYawMPC(farm, wind, limits, cfg)

    # Start with small perturbation
    init_yaws = np.array([2.0, -1.5, 1.0, -0.5])
    controller.set_state(init_yaws)
    print(f"  Initial yaws: {np.round(init_yaws, 1)}°")

    history = controller.run(n_steps=15, verbose=True)

    fig = plot_results(history, "Demo 2: Longer Horizon (50 steps)")
    plt.savefig('demo2_longer_horizon.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: demo2_longer_horizon.png")

    return history


def demo_3_optimal_setup():
    """Demo 3: Optimal setup for yaw control."""
    print("\n" + "="*70)
    print("DEMO 3: Optimal Setup (Closer + Longer Horizon)")
    print("="*70)

    np.random.seed(42)
    D = 178.0

    # Closer spacing AND longer horizon
    x = np.array([0.0, 4*D, 8*D, 12*D])
    y = np.zeros_like(x)
    farm = Farm(x=x, y=y, D=D)

    wind = Wind(U=8.0, theta=270.0)
    limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.5)
    # Use dt=20s: N_h=20 gives 400s horizon, covers all delays safely
    cfg = MPCConfig(dt=20.0, N_h=20, lam_move=0.05)  # Lower move penalty

    print(f"\nSetup:")
    print(f"  Turbine spacing: 4D = {4*D:.0f}m")
    print(f"  Wind speed: {wind.U} m/s")
    print(f"  Horizon: {cfg.N_h} steps = {cfg.N_h * cfg.dt:.0f}s")
    print(f"  Move penalty: {cfg.lam_move}")

    controller = AcadosYawMPC(farm, wind, limits, cfg)

    # Start with random yaws
    init_yaws = np.random.uniform(-5, 5, controller.N)
    controller.set_state(init_yaws)
    print(f"  Initial yaws: {np.round(init_yaws, 1)}°")

    history = controller.run(n_steps=25, verbose=True)

    fig = plot_results(history, "Demo 3: Optimal Setup")
    plt.savefig('demo3_optimal_setup.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: demo3_optimal_setup.png")

    # Print summary
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    powers = [h['power']/1e6 for h in history]
    print(f"Initial power: {powers[0]:.3f} MW")
    print(f"Final power:   {powers[-1]:.3f} MW")
    print(f"Power gain:    {(powers[-1]/powers[0] - 1)*100:.1f}%")
    print(f"Best power:    {max(powers):.3f} MW")
    print(f"Gain vs best:  {(max(powers)/powers[0] - 1)*100:.1f}%")

    final_yaws = history[-1]['psi']
    print(f"\nFinal yaw angles: {np.round(final_yaws, 1)}°")

    avg_solve = np.mean([h['solve_time'] for h in history]) * 1000
    avg_grad = np.mean([h['grad_time'] for h in history]) * 1000
    print(f"\nPerformance:")
    print(f"  Avg solve time: {avg_solve:.2f}ms")
    print(f"  Avg grad time:  {avg_grad:.0f}ms")
    print(f"  Total per step: {avg_solve + avg_grad:.0f}ms")

    return history


def main():
    """Run all demos."""
    import sys
    from datetime import datetime

    # Set up logging to file
    log_filename = f"mpc_demo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_filename, 'w')

    class TeeOutput:
        """Write to both stdout and file."""
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = TeeOutput(log_file)

    print("="*70)
    print("Wind Farm Yaw Optimization Demos")
    print("="*70)
    print(f"\nLogging to: {log_filename}")
    print("\nThese demos show the MPC working properly with:")
    print("  1. Closer turbine spacing (smaller delays)")
    print("  2. Longer prediction horizon")
    print("  3. Optimal combination of both")

    # Run demos
    try:
        h1 = demo_1_closer_spacing()
    except KeyboardInterrupt:
        print("\nDemo 1 interrupted")
        return

    try:
        h2 = demo_2_longer_horizon()
    except KeyboardInterrupt:
        print("\nDemo 2 interrupted")
        return

    try:
        h3 = demo_3_optimal_setup()
    except KeyboardInterrupt:
        print("\nDemo 3 interrupted")
        return

    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70)
    print("\nGenerated plots:")
    print("  - demo1_closer_spacing.png")
    print("  - demo2_longer_horizon.png")
    print("  - demo3_optimal_setup.png")
    print(f"\nFull log saved to: {log_filename}")

    # Show plots
    plt.show()

    # Close log file and restore stdout
    sys.stdout = sys.stdout.terminal
    log_file.close()


if __name__ == "__main__":
    main()
