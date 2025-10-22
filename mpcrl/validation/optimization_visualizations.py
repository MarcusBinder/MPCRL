"""
MPC Optimization Process Visualizations
========================================

Visualizations showing how the MPC optimization algorithm works,
including landscape plots, convergence analysis, and sequential optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import time
from scipy.optimize import dual_annealing

# Import MPC components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from mpcrl.mpc import (
    psi, yaw_traj, WindFarmModel,
    optimize_farm_back2front, run_farm_delay_loop_optimized,
    farm_energy
)
from py_wake.examples.data.hornsrev1 import V80

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

FIGURE_DIR = Path(__file__).parent / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_optimization_landscape(save=True):
    """
    Plot 2D optimization landscape for a single turbine.
    Shows energy as function of (o1, o2) parameters.
    """
    print("\nGenerating optimization landscape...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Optimization Landscape: Energy vs (o₁, o₂) Parameters',
                 fontsize=14, fontweight='bold')

    # Create 2-turbine model for clearer optimization landscape
    xs = np.array([0., 1000.])
    ys = np.array([0., 0.])
    wt = V80()
    model = WindFarmModel(xs, ys, wt=wt, U_inf=8.0, TI=0.06, wd=270.0)

    # Optimization parameters
    t_AH = 100.0
    r_gamma = 0.3
    dt_opt = 25.0
    T_opt = 400.0

    # Initial yaw angles
    current_yaws = np.array([0.0, 0.0])

    # Create parameter grid
    o1_grid = np.linspace(0.05, 0.95, 30)
    o2_grid = np.linspace(0.05, 0.95, 30)
    O1, O2 = np.meshgrid(o1_grid, o2_grid)

    print("  Computing energy landscape for upstream turbine...")
    # Evaluate energy for upstream turbine (T1) with T2 fixed
    energy_grid_upstream = np.zeros_like(O1)
    for i in range(O1.shape[0]):
        for j in range(O1.shape[1]):
            # Create parameters: optimize T1, T2 stays at baseline
            params = [[O1[i,j], O2[i,j]], [0.5, 0.5]]  # T2 at neutral

            t_opt, traj_list, P_opt = run_farm_delay_loop_optimized(
                model, params, current_yaws, r_gamma, t_AH, dt_opt, T_opt
            )
            energy = farm_energy(P_opt, t_opt)
            energy_grid_upstream[i,j] = energy / 1e6  # Convert to MWh

    # Plot upstream turbine landscape
    ax = axes[0]
    levels = np.linspace(energy_grid_upstream.min(), energy_grid_upstream.max(), 20)
    contour = ax.contourf(O1, O2, energy_grid_upstream, levels=levels, cmap='viridis')
    contour_lines = ax.contour(O1, O2, energy_grid_upstream, levels=10,
                               colors='k', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')

    # Find and mark optimum
    idx_max = np.unravel_index(np.argmax(energy_grid_upstream), energy_grid_upstream.shape)
    o1_opt = O1[idx_max]
    o2_opt = O2[idx_max]
    ax.plot(o1_opt, o2_opt, 'r*', markersize=20, label=f'Optimum: ({o1_opt:.2f}, {o2_opt:.2f})')

    ax.set_xlabel('o₁ (Magnitude Direction)')
    ax.set_ylabel('o₂ (Timing)')
    ax.set_title('Upstream Turbine (T1) Optimization Landscape')
    ax.legend()
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Energy (MWh)', rotation=270, labelpad=20)

    print("  Computing energy landscape for downstream turbine...")
    # Now optimize downstream turbine (T2) with T1 at optimal
    energy_grid_downstream = np.zeros_like(O1)
    t1_optimal_params = [o1_opt, o2_opt]

    for i in range(O1.shape[0]):
        for j in range(O1.shape[1]):
            params = [t1_optimal_params, [O1[i,j], O2[i,j]]]

            t_opt, traj_list, P_opt = run_farm_delay_loop_optimized(
                model, params, current_yaws, r_gamma, t_AH, dt_opt, T_opt
            )
            energy = farm_energy(P_opt, t_opt)
            energy_grid_downstream[i,j] = energy / 1e6

    # Plot downstream turbine landscape
    ax = axes[1]
    levels = np.linspace(energy_grid_downstream.min(), energy_grid_downstream.max(), 20)
    contour = ax.contourf(O1, O2, energy_grid_downstream, levels=levels, cmap='viridis')
    contour_lines = ax.contour(O1, O2, energy_grid_downstream, levels=10,
                               colors='k', alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')

    # Find and mark optimum
    idx_max = np.unravel_index(np.argmax(energy_grid_downstream), energy_grid_downstream.shape)
    o1_opt_t2 = O1[idx_max]
    o2_opt_t2 = O2[idx_max]
    ax.plot(o1_opt_t2, o2_opt_t2, 'r*', markersize=20,
           label=f'Optimum: ({o1_opt_t2:.2f}, {o2_opt_t2:.2f})')

    ax.set_xlabel('o₁ (Magnitude Direction)')
    ax.set_ylabel('o₂ (Timing)')
    ax.set_title('Downstream Turbine (T2) Optimization Landscape\n(with T1 at optimum)')
    ax.legend()
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Energy (MWh)', rotation=270, labelpad=20)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "optimization_landscape.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "optimization_landscape.pdf", bbox_inches='tight')

    print(f"  ✓ Saved optimization landscape")
    return fig


def plot_sequential_optimization(save=True):
    """
    Show the sequential (back-to-front) optimization process.
    """
    print("\nGenerating sequential optimization visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Sequential Back-to-Front Optimization Process',
                 fontsize=15, fontweight='bold')

    # 4-turbine setup
    xs = np.array([0., 800., 1600., 2400.])
    ys = np.array([0., 0., 0., 0.])
    wt = V80()
    model = WindFarmModel(xs, ys, wt=wt, U_inf=8.0, TI=0.06, wd=270.0)

    t_AH = 100.0
    r_gamma = 0.3
    dt_opt = 25.0
    T_opt = 400.0
    current_yaws = np.zeros(4)

    print("  Running full optimization...")
    # Run full optimization
    optimal_params = optimize_farm_back2front(
        model, current_yaws, r_gamma, t_AH, dt_opt, T_opt,
        maxfun=20, seed=42
    )

    # Generate optimal trajectories
    t_opt, optimal_trajs, P_opt = run_farm_delay_loop_optimized(
        model, optimal_params, current_yaws, r_gamma, t_AH, dt_opt, T_opt
    )

    # Plot 1: Optimal trajectories
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['blue', 'green', 'orange', 'red']
    for i in range(4):
        ax1.plot(t_opt, optimal_trajs[i], color=colors[i], linewidth=2.5,
                label=f'T{i+1} (order: {4-i})', alpha=0.8)

    ax1.axvline(t_AH, color='k', linestyle='--', alpha=0.5, label='Action Horizon')
    ax1.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Yaw Angle (deg)', fontsize=12)
    ax1.set_title('Optimal Yaw Trajectories (All Turbines)', fontsize=13)
    ax1.legend(loc='best', ncol=5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Power evolution
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(4):
        ax2.plot(t_opt, P_opt[i] / 1000, color=colors[i], linewidth=2,
                label=f'T{i+1}', alpha=0.8)
    ax2.plot(t_opt, P_opt.sum(axis=0) / 1000, 'k--', linewidth=2.5,
            label='Total', alpha=0.8)

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Power (kW)', fontsize=12)
    ax2.set_title('Power Production over Time', fontsize=13)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Parameter values
    ax3 = fig.add_subplot(gs[1, 1])
    turbine_labels = [f'T{i+1}' for i in range(4)]
    o1_values = [p[0] for p in optimal_params]
    o2_values = [p[1] for p in optimal_params]

    x_pos = np.arange(4)
    width = 0.35
    ax3.bar(x_pos - width/2, o1_values, width, label='o₁ (magnitude)', alpha=0.8)
    ax3.bar(x_pos + width/2, o2_values, width, label='o₂ (timing)', alpha=0.8)

    ax3.set_xlabel('Turbine', fontsize=12)
    ax3.set_ylabel('Parameter Value', fontsize=12)
    ax3.set_title('Optimal Parameters per Turbine', fontsize=13)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(turbine_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1)

    # Add text annotation explaining order
    annotation_text = """
OPTIMIZATION ORDER: T4 → T3 → T2 → T1
(Back-to-front: downstream first)

This ensures each turbine can optimize
knowing the final decisions of all
downstream turbines.
"""
    ax3.text(0.98, 0.02, annotation_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "sequential_optimization.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "sequential_optimization.pdf", bbox_inches='tight')

    print(f"  ✓ Saved sequential optimization")
    return fig


def plot_convergence_analysis(save=True):
    """
    Show convergence for different maxfun values.
    """
    print("\nGenerating convergence analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimization Convergence Analysis',
                 fontsize=14, fontweight='bold')

    # 3-turbine setup for faster computation
    xs = np.array([0., 1000., 2000.])
    ys = np.array([0., 0., 0.])
    wt = V80()
    model = WindFarmModel(xs, ys, wt=wt, U_inf=8.0, TI=0.06, wd=270.0)

    t_AH = 100.0
    r_gamma = 0.3
    dt_opt = 25.0
    T_opt = 400.0
    current_yaws = np.zeros(3)

    # Test different maxfun values
    maxfun_values = [5, 10, 20, 30, 50]
    colors_map = plt.cm.viridis(np.linspace(0, 1, len(maxfun_values)))

    print("  Testing different maxfun values...")
    energies = []
    times = []

    for idx, maxfun in enumerate(maxfun_values):
        print(f"    maxfun = {maxfun}...")
        start = time.time()
        params = optimize_farm_back2front(
            model, current_yaws, r_gamma, t_AH, dt_opt, T_opt,
            maxfun=maxfun, seed=42
        )
        elapsed = time.time() - start

        t_opt, trajs, P_opt = run_farm_delay_loop_optimized(
            model, params, current_yaws, r_gamma, t_AH, dt_opt, T_opt
        )
        energy = farm_energy(P_opt, t_opt)

        energies.append(energy / 1e6)
        times.append(elapsed)

    # Plot 1: Energy vs maxfun
    ax = axes[0, 0]
    ax.plot(maxfun_values, energies, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('maxfun (function evaluations per turbine)')
    ax.set_ylabel('Total Energy (MWh)')
    ax.set_title('Energy vs Optimization Budget')
    ax.grid(True, alpha=0.3)

    # Plot 2: Computation time vs maxfun
    ax = axes[0, 1]
    ax.plot(maxfun_values, times, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('maxfun (function evaluations per turbine)')
    ax.set_ylabel('Computation Time (s)')
    ax.set_title('Computation Time vs Optimization Budget')
    ax.grid(True, alpha=0.3)

    # Plot 3: Pareto frontier (energy vs time)
    ax = axes[1, 0]
    scatter = ax.scatter(times, energies, c=maxfun_values, s=100,
                        cmap='viridis', edgecolors='black', linewidth=1.5)
    for i, maxfun in enumerate(maxfun_values):
        ax.annotate(f'{maxfun}', (times[i], energies[i]),
                   fontsize=9, ha='right', va='bottom')
    ax.set_xlabel('Computation Time (s)')
    ax.set_ylabel('Total Energy (MWh)')
    ax.set_title('Pareto Frontier: Energy vs Computation Time')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('maxfun', rotation=270, labelpad=15)

    # Plot 4: Normalized metrics
    ax = axes[1, 1]
    energies_norm = np.array(energies) / energies[-1]  # Normalize to best
    times_norm = np.array(times) / times[-1]  # Normalize to slowest

    ax.plot(maxfun_values, energies_norm, 'bo-', linewidth=2, label='Energy (normalized)', markersize=8)
    ax.plot(maxfun_values, times_norm, 'ro-', linewidth=2, label='Time (normalized)', markersize=8)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('maxfun (function evaluations per turbine)')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Normalized Energy and Time vs maxfun')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "convergence_analysis.pdf", bbox_inches='tight')

    print(f"  ✓ Saved convergence analysis")

    # Print summary
    print("\n  Convergence Summary:")
    print("  " + "-"*50)
    print(f"  {'maxfun':<10} {'Energy (MWh)':<15} {'Time (s)':<12} {'Energy/Time':<12}")
    print("  " + "-"*50)
    for i, maxfun in enumerate(maxfun_values):
        efficiency = energies[i] / times[i]
        print(f"  {maxfun:<10} {energies[i]:<15.4f} {times[i]:<12.2f} {efficiency:<12.4f}")
    print("  " + "-"*50)

    return fig


def plot_time_shifted_cost_comparison(save=True):
    """
    Visualize the difference between standard and time-shifted cost functions.
    """
    print("\nGenerating time-shifted cost comparison...")

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Time-Shifted Cost Function: Accounting for Wake Propagation Delays',
                 fontsize=15, fontweight='bold')

    # 3-turbine setup
    xs = np.array([0., 1000., 2000.])
    ys = np.array([0., 0., 0.])
    wt = V80()
    model = WindFarmModel(xs, ys, wt=wt, U_inf=8.0, TI=0.06, wd=270.0)

    t_AH = 100.0
    r_gamma = 0.3
    dt_opt = 25.0
    T_opt = 400.0
    current_yaws = np.zeros(3)

    print("  Optimizing with standard cost...")
    # Optimize with standard cost (use_time_shifted=False)
    params_standard = optimize_farm_back2front(
        model, current_yaws, r_gamma, t_AH, dt_opt, T_opt,
        maxfun=20, seed=42, use_time_shifted=False
    )

    print("  Optimizing with time-shifted cost...")
    # Optimize with time-shifted cost (use_time_shifted=True)
    params_time_shifted = optimize_farm_back2front(
        model, current_yaws, r_gamma, t_AH, dt_opt, T_opt,
        maxfun=20, seed=42, use_time_shifted=True
    )

    # Generate trajectories and power for both
    t_opt, trajs_std, P_std = run_farm_delay_loop_optimized(
        model, params_standard, current_yaws, r_gamma, t_AH, dt_opt, T_opt
    )

    _, trajs_ts, P_ts = run_farm_delay_loop_optimized(
        model, params_time_shifted, current_yaws, r_gamma, t_AH, dt_opt, T_opt
    )

    # Plot 1: Timeline showing integration windows
    ax1 = fig.add_subplot(gs[0, :])

    delays = model.delays
    colors = ['blue', 'green', 'red']

    # Draw integration windows for each turbine
    for i in range(3):
        # Standard cost: all turbines use [0, t_AH]
        ax1.barh(i*2, t_AH, left=0, height=0.6, color=colors[i], alpha=0.3,
                label=f'T{i+1} Standard: [0, {t_AH}s]' if i == 0 else '')

        # Time-shifted: shifted by delays
        # For turbine i, we integrate power of downstream turbines j > i
        # over [delay[i,j], delay[i,j] + t_AH]
        if i < 2:  # Not last turbine
            for j in range(i+1, 3):
                delay_ij = delays[i, j]
                ax1.barh(i*2 + 0.8, t_AH, left=delay_ij, height=0.6,
                        color=colors[j], alpha=0.5, edgecolor='black', linewidth=1.5)
                # Label
                ax1.text(delay_ij + t_AH/2, i*2 + 0.8, f'T{j+1}',
                        ha='center', va='center', fontsize=9, fontweight='bold')

    ax1.axvline(t_AH, color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Action Horizon ({t_AH}s)')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_yticks([0, 0.8, 2, 2.8, 4, 4.8])
    ax1.set_yticklabels(['T1\nStd', 'T1\nShift', 'T2\nStd', 'T2\nShift', 'T3\nStd', 'T3\nShift'])
    ax1.set_title('Cost Function Integration Windows: Standard vs Time-Shifted', fontsize=13)
    ax1.set_xlim(0, T_opt)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.legend(loc='upper right')

    # Plot 2: Yaw trajectories comparison
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(3):
        ax2.plot(t_opt, trajs_std[i], color=colors[i], linewidth=2,
                linestyle='--', alpha=0.6, label=f'T{i+1} Standard')
    ax2.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax2.axvline(t_AH, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Yaw Angle (deg)')
    ax2.set_title('Yaw Trajectories: Standard Cost')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    for i in range(3):
        ax3.plot(t_opt, trajs_ts[i], color=colors[i], linewidth=2,
                alpha=0.8, label=f'T{i+1} Time-Shifted')
    ax3.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax3.axvline(t_AH, color='orange', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Yaw Angle (deg)')
    ax3.set_title('Yaw Trajectories: Time-Shifted Cost')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 3: Power comparison
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(t_opt, P_std.sum(axis=0) / 1000, 'b-', linewidth=2.5,
            label='Standard Cost', alpha=0.7)
    ax4.plot(t_opt, P_ts.sum(axis=0) / 1000, 'r-', linewidth=2.5,
            label='Time-Shifted Cost', alpha=0.7)
    ax4.axvline(t_AH, color='orange', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Total Power (kW)')
    ax4.set_title('Total Farm Power Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 4: Cumulative energy
    ax5 = fig.add_subplot(gs[2, 1])
    energy_std = np.cumsum(P_std.sum(axis=0)) * dt_opt / 3600  # kWh
    energy_ts = np.cumsum(P_ts.sum(axis=0)) * dt_opt / 3600

    ax5.plot(t_opt, energy_std, 'b-', linewidth=2.5, label='Standard Cost', alpha=0.7)
    ax5.plot(t_opt, energy_ts, 'r-', linewidth=2.5, label='Time-Shifted Cost', alpha=0.7)
    ax5.fill_between(t_opt, energy_std, energy_ts, alpha=0.2, color='green',
                     label=f'Difference: {(energy_ts[-1] - energy_std[-1]):.2f} kWh')
    ax5.axvline(t_AH, color='orange', linestyle='--', alpha=0.5, label=f'Action Horizon')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Cumulative Energy (kWh)')
    ax5.set_title('Cumulative Energy Production')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "time_shifted_cost.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "time_shifted_cost.pdf", bbox_inches='tight')

    energy_gain = (energy_ts[-1] - energy_std[-1]) / energy_std[-1] * 100
    print(f"  ✓ Saved time-shifted cost comparison")
    print(f"  Energy gain from time-shifted cost: {energy_gain:.2f}%")

    return fig


def generate_optimization_visualizations():
    """Generate all optimization-related visualizations."""
    print("\n" + "="*70)
    print("GENERATING OPTIMIZATION PROCESS VISUALIZATIONS")
    print("="*70)

    figures = {}

    figures['landscape'] = plot_optimization_landscape(save=True)
    plt.close()

    figures['sequential'] = plot_sequential_optimization(save=True)
    plt.close()

    figures['convergence'] = plot_convergence_analysis(save=True)
    plt.close()

    figures['time_shifted'] = plot_time_shifted_cost_comparison(save=True)
    plt.close()

    print("\n" + "="*70)
    print(f"OPTIMIZATION VISUALIZATIONS SAVED TO: {FIGURE_DIR}")
    print("="*70 + "\n")

    return figures


if __name__ == "__main__":
    generate_optimization_visualizations()
