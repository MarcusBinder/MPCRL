"""
Comparison of CasADi and acados MPC solvers for wind farm yaw control.

This script runs both implementations on the same problem and compares:
- Solution quality (final power, yaw trajectories)
- Computational performance (solve times)
- Convergence behavior

Usage:
    python compare_mpc_solvers.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List

# Import both implementations
from nmpc_windfarm import (
    Farm, Wind, Limits, NMPCConfig,
    build_pywake_model, pywake_farm_power, order_and_delays,
    build_qp_mpc, finite_diff_grad_pywake
)

try:
    from nmpc_windfarm_acados import AcadosYawMPC, MPCConfig
    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False
    print("Warning: acados not available for comparison")


def run_casadi_mpc(farm: Farm, wind: Wind, limits: Limits, cfg: NMPCConfig,
                   n_steps: int) -> Dict:
    """Run CasADi-based QP MPC."""
    print("\n" + "="*70)
    print("Running CasADi QP MPC")
    print("="*70)

    N = len(farm.x)
    wf_model, layout = build_pywake_model(farm.x, farm.y, farm.D)
    order, _, tau = order_and_delays(farm, wind, cfg.dt)

    max_tau = int(np.max(tau))
    psi_current = np.zeros(N)
    delay_hist = [psi_current.copy() for _ in range(max_tau + cfg.N_h + 10)]

    history = {
        'psi': [],
        'power': [],
        'solve_time': [],
        'step': []
    }

    for t in range(n_steps):
        t0 = time.time()

        # Build and solve QP
        opti_qp, Xvar = build_qp_mpc(
            farm, wind, limits, cfg,
            psi_prev=psi_current,
            delay_hist=delay_hist,
            tau=tau,
            wf_model=wf_model,
            layout=layout
        )

        try:
            sol = opti_qp.solve()
            X = np.array(sol.value(Xvar)).reshape(cfg.N_h, N)
            psi_plan = X
            solve_time = time.time() - t0
        except Exception as e:
            print(f"Solver failed at t={t}: {e}")
            psi_plan = np.tile(psi_current, (cfg.N_h, 1))
            solve_time = time.time() - t0

        # Apply control
        psi_next = psi_plan[0, :]
        dpsi = np.clip(
            psi_next - psi_current,
            -limits.yaw_rate_max * cfg.dt,
            limits.yaw_rate_max * cfg.dt
        )
        psi_applied = psi_current + dpsi

        # Update history
        delay_hist.insert(0, psi_applied.copy())
        if len(delay_hist) > (max_tau + cfg.N_h + 10):
            delay_hist = delay_hist[:(max_tau + cfg.N_h + 10)]

        psi_current = psi_applied

        # Compute power
        tau_i = np.max(tau, axis=1).astype(int)
        psi_delayed = np.array([delay_hist[int(tau_i[i])][i] for i in range(N)])
        P_current = pywake_farm_power(wf_model, layout, wind.U, wind.theta, psi_delayed)

        history['psi'].append(psi_current.copy())
        history['power'].append(P_current)
        history['solve_time'].append(solve_time)
        history['step'].append(t)

        print(f"t={t:02d}, ψ={np.round(psi_current, 1)}, "
              f"P={P_current/1e6:.3f} MW, solve={solve_time*1000:.1f}ms")

    avg_solve = np.mean(history['solve_time']) * 1000
    print(f"\nAverage solve time: {avg_solve:.1f}ms")

    return history


def run_acados_mpc(farm: Farm, wind: Wind, limits: Limits, cfg: MPCConfig,
                   n_steps: int) -> Dict:
    """Run acados-based MPC."""
    print("\n" + "="*70)
    print("Running acados MPC")
    print("="*70)

    controller = AcadosYawMPC(farm, wind, limits, cfg)
    history_list = controller.run(n_steps, verbose=True)

    # Convert to dict format matching CasADi
    history = {
        'psi': [h['psi'] for h in history_list],
        'power': [h['power'] for h in history_list],
        'solve_time': [h['solve_time'] for h in history_list],
        'step': [h['step'] for h in history_list]
    }

    return history


def plot_comparison(casadi_history: Dict, acados_history: Dict = None):
    """Plot comparison of both methods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Power over time
    ax = axes[0, 0]
    steps_c = casadi_history['step']
    power_c = np.array(casadi_history['power']) / 1e6
    ax.plot(steps_c, power_c, 'o-', label='CasADi QP', linewidth=2, markersize=6)

    if acados_history:
        steps_a = acados_history['step']
        power_a = np.array(acados_history['power']) / 1e6
        ax.plot(steps_a, power_a, 's-', label='acados', linewidth=2, markersize=6)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Farm Power [MW]')
    ax.set_title('Power Production Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Yaw angles over time
    ax = axes[0, 1]
    psi_c = np.array(casadi_history['psi'])
    N_turbines = psi_c.shape[1]

    for i in range(N_turbines):
        ax.plot(steps_c, psi_c[:, i], '-', label=f'T{i} (CasADi)', alpha=0.7)

    if acados_history:
        psi_a = np.array(acados_history['psi'])
        for i in range(N_turbines):
            ax.plot(steps_a, psi_a[:, i], '--', label=f'T{i} (acados)', alpha=0.7)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Yaw Angle [deg]')
    ax.set_title('Yaw Angle Trajectories')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Solve times
    ax = axes[1, 0]
    solve_c = np.array(casadi_history['solve_time']) * 1000
    ax.plot(steps_c, solve_c, 'o-', label='CasADi QP', linewidth=2)

    if acados_history:
        solve_a = np.array(acados_history['solve_time']) * 1000
        ax.plot(steps_a, solve_a, 's-', label='acados', linewidth=2)
        ax.axhline(np.mean(solve_a), color='C1', linestyle=':', alpha=0.5,
                   label=f'acados avg: {np.mean(solve_a):.1f}ms')

    ax.axhline(np.mean(solve_c), color='C0', linestyle=':', alpha=0.5,
               label=f'CasADi avg: {np.mean(solve_c):.1f}ms')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Solve Time [ms]')
    ax.set_title('Computational Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "Performance Summary\n" + "="*40 + "\n\n"

    summary_text += "CasADi QP:\n"
    summary_text += f"  Initial power: {power_c[0]:.3f} MW\n"
    summary_text += f"  Final power:   {power_c[-1]:.3f} MW\n"
    summary_text += f"  Gain:          {(power_c[-1]/power_c[0] - 1)*100:.1f}%\n"
    summary_text += f"  Avg solve:     {np.mean(solve_c):.1f} ms\n"
    summary_text += f"  Max solve:     {np.max(solve_c):.1f} ms\n"

    if acados_history:
        summary_text += "\nacados:\n"
        summary_text += f"  Initial power: {power_a[0]:.3f} MW\n"
        summary_text += f"  Final power:   {power_a[-1]:.3f} MW\n"
        summary_text += f"  Gain:          {(power_a[-1]/power_a[0] - 1)*100:.1f}%\n"
        summary_text += f"  Avg solve:     {np.mean(solve_a):.1f} ms\n"
        summary_text += f"  Max solve:     {np.max(solve_a):.1f} ms\n"

        summary_text += "\nSpeedup:\n"
        speedup = np.mean(solve_c) / np.mean(solve_a)
        summary_text += f"  acados is {speedup:.2f}x "
        summary_text += "faster\n" if speedup > 1 else "slower\n"

    ax.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
            family='monospace', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('mpc_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: mpc_comparison.png")
    plt.show()


def main():
    """Run comparison between CasADi and acados MPC."""
    np.random.seed(42)

    # Setup (same for both)
    D = 178.0
    x = np.array([0.0, 7*D, 14*D, 21*D])
    y = np.zeros_like(x)
    farm = Farm(x=x, y=y, D=D)

    wind = Wind(U=8.0, theta=0.0)
    limits = Limits(yaw_min=-25, yaw_max=25, yaw_rate_max=0.3)

    n_steps = 10

    # CasADi configuration
    cfg_casadi = NMPCConfig(dt=10.0, N_h=12, lam_move=0.2)

    # Run CasADi
    casadi_history = run_casadi_mpc(farm, wind, limits, cfg_casadi, n_steps)

    # Run acados if available
    acados_history = None
    if ACADOS_AVAILABLE:
        cfg_acados = MPCConfig(dt=10.0, N_h=12, lam_move=0.2)
        acados_history = run_acados_mpc(farm, wind, limits, cfg_acados, n_steps)
    else:
        print("\nSkipping acados comparison (not installed)")

    # Plot results
    plot_comparison(casadi_history, acados_history)


if __name__ == "__main__":
    main()
