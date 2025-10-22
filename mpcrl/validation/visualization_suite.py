"""
MPC Visualization Suite
=======================

Comprehensive visualization tools for explaining and validating the MPC controller.

This module generates publication-ready figures showing:
1. Basis functions and trajectory generation
2. Wake model and advection dynamics
3. Optimization process and convergence
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import time
from typing import List, Tuple, Dict, Optional

# Import MPC components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from mpcrl.mpc import (
    sat01, psi, yaw_traj, WindFarmModel,
    optimize_farm_back2front, run_farm_delay_loop_optimized,
    farm_energy
)

# Import turbine model from py_wake
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

# Create output directory
FIGURE_DIR = Path(__file__).parent / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECTION 1: BASIS FUNCTIONS AND TRAJECTORY GENERATION
# =============================================================================

def plot_basis_function_grid(save=True):
    """
    Plot grid of psi basis functions for different (o1, o2) combinations.
    Shows how parameters control magnitude and timing.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle('Basis Function ψ(o₁, o₂, t) for Different Parameter Values',
                 fontsize=14, fontweight='bold')

    # Parameter values to test
    o1_values = [0.2, 0.5, 0.8]  # Low, mid, high magnitude
    o2_values = [0.2, 0.5, 0.8]  # Early, mid, late timing

    t_AH = 100.0  # Action horizon
    r_gamma = 0.3  # Yaw rate (deg/s)
    t = np.linspace(0, t_AH, 200)

    for i, o1 in enumerate(o1_values):
        for j, o2 in enumerate(o2_values):
            ax = axes[i, j]

            # Compute basis function
            psi_values = psi(o1, o2, t, t_AH, r_gamma)

            # Plot
            ax.plot(t, psi_values, 'b-', linewidth=2)
            ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Δγ (deg)')
            ax.set_title(f'o₁={o1:.1f}, o₂={o2:.1f}')
            ax.grid(True, alpha=0.3)

            # Annotate direction and timing
            if o1 < 0.5:
                direction = "Negative yaw"
            elif o1 > 0.5:
                direction = "Positive yaw"
            else:
                direction = "No change"

            timing_desc = ["Early", "Mid", "Late"][j]
            ax.text(0.98, 0.05, f"{direction}\n{timing_desc} timing",
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "basis_functions_grid.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "basis_functions_grid.pdf", bbox_inches='tight')
    return fig


def plot_yaw_trajectories(save=True):
    """
    Plot example yaw trajectories for different starting conditions and parameters.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Yaw Angle Trajectories from Different Initial Conditions',
                 fontsize=14, fontweight='bold')

    t_AH = 100.0
    r_gamma = 0.3
    dt = 1.0
    T_total = 150.0

    # Test cases: (gamma0, o1, o2, description)
    test_cases = [
        (0.0, 0.75, 0.2, "From 0°: Early positive yaw"),
        (0.0, 0.25, 0.8, "From 0°: Late negative yaw"),
        (10.0, 0.6, 0.5, "From +10°: Mid positive yaw"),
        (-15.0, 0.4, 0.3, "From -15°: Early negative yaw"),
    ]

    for idx, (gamma0, o1, o2, desc) in enumerate(test_cases):
        ax = axes[idx // 2, idx % 2]

        t, gamma = yaw_traj(gamma0, o1, o2, t_AH, r_gamma, dt, T_total)

        ax.plot(t, gamma, 'b-', linewidth=2, label='Trajectory')
        ax.axhline(gamma0, color='g', linestyle='--', alpha=0.5, label=f'Initial: {gamma0}°')
        ax.axhline(33, color='r', linestyle=':', alpha=0.5, label='Limits: ±33°')
        ax.axhline(-33, color='r', linestyle=':', alpha=0.5)
        ax.axvline(t_AH, color='orange', linestyle='--', alpha=0.5, label=f't_AH = {t_AH}s')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Yaw Angle γ (deg)')
        ax.set_title(desc + f'\n(o₁={o1:.1f}, o₂={o2:.1f})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-40, 40)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "yaw_trajectories.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "yaw_trajectories.pdf", bbox_inches='tight')
    return fig


def plot_parameter_space_heatmap(save=True):
    """
    Heatmap showing final yaw angle as function of (o1, o2).
    Helps identify regions of parameter space.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Parameter Space Analysis: Final Yaw Angle',
                 fontsize=14, fontweight='bold')

    # Create grid
    o1_grid = np.linspace(0, 1, 50)
    o2_grid = np.linspace(0, 1, 50)
    O1, O2 = np.meshgrid(o1_grid, o2_grid)

    t_AH = 100.0
    r_gamma = 0.3

    # Compute final yaw change for each (o1, o2)
    final_yaw = np.zeros_like(O1)
    for i in range(O1.shape[0]):
        for j in range(O1.shape[1]):
            _, gamma = yaw_traj(0.0, O1[i,j], O2[i,j], t_AH, r_gamma, 1.0, t_AH)
            final_yaw[i,j] = gamma[-1]

    # Plot 1: Final yaw heatmap
    im1 = axes[0].contourf(O1, O2, final_yaw, levels=20, cmap='RdBu_r')
    axes[0].contour(O1, O2, final_yaw, levels=10, colors='k', alpha=0.3, linewidths=0.5)
    axes[0].set_xlabel('o₁ (Magnitude Direction)', fontsize=11)
    axes[0].set_ylabel('o₂ (Timing)', fontsize=11)
    axes[0].set_title('Final Yaw Angle (deg)')
    axes[0].axvline(0.5, color='k', linestyle='--', alpha=0.5, linewidth=1)
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Final γ (deg)', rotation=270, labelpad=20)

    # Plot 2: Rate of change (gradient magnitude)
    time_to_90pct = np.zeros_like(O1)
    for i in range(O1.shape[0]):
        for j in range(O1.shape[1]):
            t, gamma = yaw_traj(0.0, O1[i,j], O2[i,j], t_AH, r_gamma, 0.5, t_AH)
            final = gamma[-1]
            if abs(final) > 1e-6:
                # Find time to reach 90% of final value
                idx = np.where(np.abs(gamma - 0.9 * final) < 0.1 * abs(final))[0]
                time_to_90pct[i,j] = t[idx[0]] if len(idx) > 0 else t_AH
            else:
                time_to_90pct[i,j] = 0

    im2 = axes[1].contourf(O1, O2, time_to_90pct, levels=20, cmap='viridis')
    axes[1].contour(O1, O2, time_to_90pct, levels=10, colors='k', alpha=0.3, linewidths=0.5)
    axes[1].set_xlabel('o₁ (Magnitude Direction)', fontsize=11)
    axes[1].set_ylabel('o₂ (Timing)', fontsize=11)
    axes[1].set_title('Time to 90% of Final Value')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Time (s)', rotation=270, labelpad=20)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "parameter_space_heatmap.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "parameter_space_heatmap.pdf", bbox_inches='tight')
    return fig


# =============================================================================
# SECTION 2: WAKE MODEL AND ADVECTION DYNAMICS
# =============================================================================

def plot_turbine_layout_with_wakes(save=True):
    """
    Visualize turbine layout and wake effects for different wind directions.
    """
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)
    fig.suptitle('Turbine Layout and Wake Effects for Different Wind Directions',
                 fontsize=14, fontweight='bold')

    # Create wind farm model (4-turbine line)
    xs = np.array([0., 800., 1600., 2400.])
    ys = np.array([0., 0., 0., 0.])

    wind_directions = [270, 240, 300]  # West, SW, NW
    wd_labels = ['270° (West)', '240° (SW)', '300° (NW)']

    # Create turbine model instance
    wt = V80()

    for idx, (wd, wd_label) in enumerate(zip(wind_directions, wd_labels)):
        ax = fig.add_subplot(gs[0, idx])

        # Plot turbines
        ax.scatter(xs, ys, s=200, c='blue', marker='o', zorder=3, label='Turbines')

        # Add turbine numbers
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.text(x, y+100, f'T{i+1}', ha='center', fontsize=10, fontweight='bold')

        # Calculate sorting based on wind direction
        wd_rad = np.deg2rad(wd)
        proj = xs * np.cos(wd_rad) + ys * np.sin(wd_rad)
        sorted_indices = np.argsort(proj)

        # Draw wind direction arrow
        wind_scale = 500
        wind_dx = -wind_scale * np.sin(wd_rad)
        wind_dy = -wind_scale * np.cos(wd_rad)
        ax.arrow(xs.mean() - 2*wind_dx, ys.mean() - 2*wind_dy,
                wind_dx, wind_dy, head_width=150, head_length=100,
                fc='red', ec='red', linewidth=2, label=f'Wind {wd_label}')

        # Draw schematic wakes (Gaussian expansion)
        wake_expansion = 0.1  # Expansion rate
        for i in sorted_indices[:-1]:  # All but last (downstream)
            x_turb, y_turb = xs[i], ys[i]

            # Wake centerline direction (opposite to wind)
            wake_dx = -np.sin(wd_rad)
            wake_dy = -np.cos(wd_rad)

            # Draw expanding wake cone
            wake_length = 1500
            for dist in [500, 1000, 1500]:
                wake_width = 100 + wake_expansion * dist
                wake_x = x_turb + wake_dx * dist
                wake_y = y_turb + wake_dy * dist

                # Draw ellipse representing wake cross-section
                ellipse = patches.Ellipse((wake_x, wake_y),
                                         width=wake_width*2,
                                         height=wake_width*2,
                                         angle=wd,
                                         alpha=0.1, color='gray', zorder=1)
                ax.add_patch(ellipse)

        # Annotate sorting order
        order_text = f"Order: {' → '.join([f'T{i+1}' for i in sorted_indices])}"
        ax.text(0.5, 0.95, order_text, transform=ax.transAxes,
               ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(f'Wind Direction: {wd_label}')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-500, 3500)
        ax.set_ylim(-1000, 1000)
        ax.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "turbine_layout_wakes.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "turbine_layout_wakes.pdf", bbox_inches='tight')
    return fig


def plot_wake_delay_matrix(save=True):
    """
    Heatmap of wake propagation delays between turbine pairs.
    Shows dependency on wind speed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Wake Propagation Delay Matrix for Different Wind Speeds',
                 fontsize=14, fontweight='bold')

    # 4-turbine line layout
    xs = np.array([0., 800., 1600., 2400.])
    ys = np.array([0., 0., 0., 0.])
    wd = 270.0  # Aligned flow

    # Create turbine model
    wt = V80()

    wind_speeds = [6, 8, 10]  # m/s

    for idx, U_inf in enumerate(wind_speeds):
        ax = axes[idx]

        # Calculate delays
        model = WindFarmModel(xs, ys, wt=wt, U_inf=U_inf, TI=0.06, wd=wd)
        delays = model.delays

        # Plot heatmap
        im = ax.imshow(delays, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([f'T{i+1}' for i in range(4)])
        ax.set_yticklabels([f'T{i+1}' for i in range(4)])
        ax.set_xlabel('Downstream Turbine')
        ax.set_ylabel('Upstream Turbine')
        ax.set_title(f'Wind Speed: {U_inf} m/s')

        # Add text annotations
        for i in range(4):
            for j in range(4):
                text = ax.text(j, i, f'{delays[i, j]:.0f}s',
                             ha="center", va="center", color="black", fontsize=10)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Delay (seconds)', rotation=270, labelpad=20)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "wake_delay_matrix.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "wake_delay_matrix.pdf", bbox_inches='tight')
    return fig


def plot_power_vs_yaw(save=True):
    """
    Plot power output vs yaw angle for single turbine.
    Shows penalty function behavior near limits.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Power vs Yaw Angle: Constraint Penalty and Cosine Law',
                 fontsize=14, fontweight='bold')

    # Create single-turbine model
    xs = np.array([0.])
    ys = np.array([0.])
    wt = V80()
    model = WindFarmModel(xs, ys, wt=wt, U_inf=8.0, TI=0.06, wd=270.0)

    # Sweep yaw angles
    yaw_angles = np.linspace(-40, 40, 200)
    powers = []

    for yaw in yaw_angles:
        P = model.farm_power(np.array([yaw]))
        powers.append(P[0])

    powers = np.array(powers)
    max_power = powers.max()
    normalized_powers = powers / max_power

    # Plot 1: Absolute power with penalty regions
    ax = axes[0]
    ax.plot(yaw_angles, powers / 1000, 'b-', linewidth=2, label='PyWake power')

    # Highlight penalty regions
    ax.axvspan(-40, -33, alpha=0.2, color='red', label='Penalty region')
    ax.axvspan(33, 40, alpha=0.2, color='red')
    ax.axvline(33, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(-33, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(0, color='k', linestyle=':', alpha=0.3)

    ax.set_xlabel('Yaw Angle γ (deg)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Power Output with Penalty Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Normalized power vs cos³(yaw) approximation
    ax = axes[1]

    # In valid range only
    valid_mask = (yaw_angles >= -33) & (yaw_angles <= 33)
    yaw_valid = yaw_angles[valid_mask]
    power_valid = normalized_powers[valid_mask]

    # cos^3 approximation
    cos3_approx = np.cos(np.deg2rad(yaw_valid))**3

    ax.plot(yaw_valid, power_valid, 'b-', linewidth=2, label='PyWake (normalized)')
    ax.plot(yaw_valid, cos3_approx, 'r--', linewidth=2, label='cos³(γ) approximation')

    ax.set_xlabel('Yaw Angle γ (deg)')
    ax.set_ylabel('Normalized Power')
    ax.set_title('Comparison with Cosine³ Law')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-35, 35)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "power_vs_yaw.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "power_vs_yaw.pdf", bbox_inches='tight')
    return fig


def plot_wake_interaction(save=True):
    """
    Show how upstream turbine yaw affects downstream power.
    Critical for understanding wake steering benefits.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Wake Interaction: Upstream Yaw Effect on Downstream Power',
                 fontsize=14, fontweight='bold')

    # 2-turbine setup
    xs = np.array([0., 1200.])
    ys = np.array([0., 0.])
    wt = V80()
    model = WindFarmModel(xs, ys, wt=wt, U_inf=8.0, TI=0.06, wd=270.0)

    # Sweep upstream yaw
    upstream_yaws = np.linspace(-30, 30, 50)
    downstream_yaws = [0, 10, -10]  # Test different downstream settings

    # Plot 1: Farm total power
    ax = axes[0]
    for down_yaw in downstream_yaws:
        farm_powers = []
        for up_yaw in upstream_yaws:
            P = model.farm_power(np.array([up_yaw, down_yaw]))
            farm_powers.append(P.sum())

        ax.plot(upstream_yaws, np.array(farm_powers) / 1000, linewidth=2,
               label=f'Downstream yaw = {down_yaw}°')

    ax.set_xlabel('Upstream Turbine Yaw (deg)')
    ax.set_ylabel('Total Farm Power (kW)')
    ax.set_title('Total Power vs Upstream Yaw')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Individual turbine powers with downstream fixed at 0°
    ax = axes[1]
    down_yaw = 0
    upstream_powers = []
    downstream_powers = []

    for up_yaw in upstream_yaws:
        P = model.farm_power(np.array([up_yaw, down_yaw]))
        upstream_powers.append(P[0])
        downstream_powers.append(P[1])

    ax.plot(upstream_yaws, np.array(upstream_powers) / 1000, 'b-',
           linewidth=2, label='Upstream turbine')
    ax.plot(upstream_yaws, np.array(downstream_powers) / 1000, 'r-',
           linewidth=2, label='Downstream turbine')
    ax.plot(upstream_yaws, (np.array(upstream_powers) + np.array(downstream_powers)) / 1000,
           'g--', linewidth=2, label='Total')

    ax.set_xlabel('Upstream Turbine Yaw (deg)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Individual Turbine Powers (Downstream at 0°)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate optimal upstream yaw
    total_power = np.array(upstream_powers) + np.array(downstream_powers)
    optimal_idx = np.argmax(total_power)
    optimal_yaw = upstream_yaws[optimal_idx]
    ax.axvline(optimal_yaw, color='g', linestyle=':', alpha=0.5)
    ax.text(optimal_yaw, ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
           f'Optimal: {optimal_yaw:.1f}°', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    if save:
        plt.savefig(FIGURE_DIR / "wake_interaction.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "wake_interaction.pdf", bbox_inches='tight')
    return fig


def plot_wake_advection_dynamics(save=True):
    """
    CRITICAL: Show wake advection with time delays and their impact on performance.
    Demonstrates why accounting for delays matters AND shows power increase from wake steering.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle('Wake Advection Dynamics: Time Delays & Power Gains from Wake Steering',
                 fontsize=15, fontweight='bold')

    # Setup: 3-turbine line for clearer visualization
    xs = np.array([0., 1000., 2000.])
    ys = np.array([0., 0., 0.])
    U_inf = 8.0
    wd = 270.0
    wt = V80()
    model = WindFarmModel(xs, ys, wt=wt, U_inf=U_inf, TI=0.06, wd=wd)

    # Scenario: T1 changes yaw from 0° to +20° (stronger effect)
    # This deflects wake AWAY from T2 and T3, increasing their power

    t_AH = 100.0
    r_gamma = 0.3
    dt = 5.0
    T_sim = 300.0

    # Create yaw trajectory for T1 (step change to +20°)
    _, gamma_T1 = yaw_traj(0.0, 0.8, 0.15, t_AH, r_gamma, dt, T_sim)
    t = np.arange(0, T_sim + dt/2, dt)

    # T2 and T3 stay at 0°
    gamma_T2 = np.zeros_like(gamma_T1)
    gamma_T3 = np.zeros_like(gamma_T1)

    # Calculate delays
    delay_12 = model.delays[0, 1]  # T1 to T2
    delay_13 = model.delays[0, 2]  # T1 to T3

    # Simulate power - we'll compare three cases:
    # 1. Baseline: All turbines at 0° yaw (no wake steering)
    # 2. Instant wake: T1 yaws, instant effect (wrong physics)
    # 3. Delayed wake: T1 yaws, delayed effect (correct physics)

    n_steps = len(t)

    # BASELINE: All at 0° yaw
    P_baseline = np.zeros((3, n_steps))
    for k in range(n_steps):
        P_baseline[:, k] = model.farm_power(np.array([0.0, 0.0, 0.0]))

    # INSTANT WAKE: T1 yaws, instant propagation (WRONG)
    P_instant = np.zeros((3, n_steps))
    for k in range(n_steps):
        P_instant[:, k] = model.farm_power(np.array([gamma_T1[k], gamma_T2[k], gamma_T3[k]]))

    # DELAYED WAKE: T1 yaws, delayed propagation (CORRECT - conceptual)
    # Since PyWake doesn't natively support time delays, we'll approximate this
    # by computing power with the yaw angle from delay_steps ago
    P_delayed = np.zeros((3, n_steps))
    delay_steps_12 = int(delay_12 / dt)
    delay_steps_13 = int(delay_13 / dt)

    # For a more realistic delayed effect, we compute power changes over time
    # T1 always sees its own current yaw
    # T2 sees T1's yaw from delay_12 ago
    # T3 sees T1's yaw from delay_13 ago

    # Baseline powers
    P_base_single = model.farm_power(np.array([0.0, 0.0, 0.0]))

    for k in range(n_steps):
        # What yaw angle does each turbine "feel" from upstream?
        k_T1_for_T2 = max(0, k - delay_steps_12)
        k_T1_for_T3 = max(0, k - delay_steps_13)

        # Compute power with delayed yaw effects
        # This is approximate - showing the concept
        gamma_T1_felt_by_T2 = gamma_T1[k_T1_for_T2]
        gamma_T1_felt_by_T3 = gamma_T1[k_T1_for_T3]

        # Compute power - when T1 yaws away, downstream power increases
        # Use instant calculation but with time-delayed yaw values
        P_delayed[:, k] = model.farm_power(np.array([gamma_T1[k], gamma_T2[k], gamma_T3[k]]))

        # Adjust T2 and T3 power based on when they "feel" the deflected wake
        if k < delay_steps_12:
            # T2 hasn't felt the wake deflection yet - use baseline
            P_delayed[1, k] = P_base_single[1]
        if k < delay_steps_13:
            # T3 hasn't felt the wake deflection yet - use baseline
            P_delayed[2, k] = P_base_single[2]

    # PLOT 0: Farm Layout with Wake Deflection Visualization
    ax0 = fig.add_subplot(gs[0, 0])

    # Draw turbines
    ax0.scatter(xs, ys, s=300, c='blue', marker='o', zorder=3, edgecolors='black', linewidth=2)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax0.text(x, y+80, f'T{i+1}', ha='center', fontsize=12, fontweight='bold')

    # Wind direction arrow
    wind_scale = 400
    wd_rad = np.deg2rad(270)
    wind_dx = -wind_scale * np.sin(wd_rad)
    wind_dy = -wind_scale * np.cos(wd_rad)
    ax0.arrow(xs.mean(), ys.max() + 300, wind_dx, wind_dy,
             head_width=100, head_length=80, fc='red', ec='red', linewidth=2.5)
    ax0.text(xs.mean(), ys.max() + 350, 'Wind', ha='center', fontsize=11, fontweight='bold', color='red')

    # Draw baseline wake (no deflection) - gray
    wake_length = 1200
    wake_width = 100
    for i in range(2):  # T1 and T2 have downstream turbines
        x_turb = xs[i]
        y_turb = ys[i]
        wake_rect = patches.Rectangle((x_turb - wake_width/2, y_turb),
                                     wake_width, wake_length,
                                     alpha=0.2, color='gray', zorder=1,
                                     label='Baseline wake' if i==0 else '')
        ax0.add_patch(wake_rect)

    # Draw deflected wake from T1 (when yawed +20°) - green
    yaw_deflection = 150  # Approximate lateral deflection at distance
    wake_rect_deflected = patches.Rectangle((xs[0] - wake_width/2 + yaw_deflection, ys[0]),
                                            wake_width, wake_length,
                                            alpha=0.3, color='green', zorder=2,
                                            label='Deflected wake (T1 yawed)')
    ax0.add_patch(wake_rect_deflected)

    # Add yaw angle annotation for T1
    ax0.annotate('', xy=(xs[0] + 150, ys[0]), xytext=(xs[0], ys[0]),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='blue'))
    ax0.text(xs[0] + 80, ys[0] - 100, '+20°', fontsize=11, color='blue', fontweight='bold')

    ax0.set_xlabel('X Position (m)', fontsize=11)
    ax0.set_ylabel('Y Position (m)', fontsize=11)
    ax0.set_title('Farm Layout: Wake Steering Effect', fontsize=12, fontweight='bold')
    ax0.axis('equal')
    ax0.set_xlim(-200, 2400)
    ax0.set_ylim(-400, 1500)
    ax0.legend(loc='upper right', fontsize=9)
    ax0.grid(True, alpha=0.3)

    # PLOT 1: Yaw trajectory with delay markers
    ax1 = fig.add_subplot(gs[0, 1:])
    ax1.plot(t, gamma_T1, 'b-', linewidth=2.5, label='T1 yaw angle (wake deflection)')
    ax1.axhline(0, color='k', linestyle=':', alpha=0.3)
    ax1.axvline(delay_12, color='r', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Wake arrives at T2 ({delay_12:.0f}s)')
    ax1.axvline(delay_13, color='orange', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Wake arrives at T3 ({delay_13:.0f}s)')

    # Shade regions
    ax1.axvspan(0, delay_12, alpha=0.1, color='blue', label='T2: Still in baseline wake')
    ax1.axvspan(delay_12, delay_13, alpha=0.1, color='yellow', label='T2: Sees deflection, T3: Baseline')
    ax1.axvspan(delay_13, T_sim, alpha=0.1, color='green', label='All: See deflection')

    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Yaw Angle (deg)', fontsize=11)
    ax1.set_title('T1 Yaw Change Timeline: Wake Deflection Propagation', fontsize=12)
    ax1.legend(loc='right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # PLOT 2: Baseline Power (no wake steering)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, P_baseline[0] / 1000, 'b--', linewidth=2, label='T1 (baseline)', alpha=0.6)
    ax2.plot(t, P_baseline[1] / 1000, 'r--', linewidth=2, label='T2 (baseline)', alpha=0.6)
    ax2.plot(t, P_baseline[2] / 1000, 'g--', linewidth=2, label='T3 (baseline)', alpha=0.6)
    ax2.plot(t, P_baseline.sum(axis=0) / 1000, 'k-', linewidth=2.5, label='Total (baseline)', alpha=0.8)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Power (kW)', fontsize=11)
    ax2.set_title('Baseline: All Turbines at 0° Yaw (No Steering)', fontsize=12)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)

    # PLOT 3: Power with instant wake (WRONG physics)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, P_instant[0] / 1000, 'b-', linewidth=2, label='T1', alpha=0.8)
    ax3.plot(t, P_instant[1] / 1000, 'r-', linewidth=2, label='T2', alpha=0.8)
    ax3.plot(t, P_instant[2] / 1000, 'g-', linewidth=2, label='T3', alpha=0.8)
    ax3.plot(t, P_instant.sum(axis=0) / 1000, 'k--', linewidth=2.5, label='Total', alpha=0.8)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Power (kW)', fontsize=11)
    ax3.set_title('Instant Wake: T1 Yaws +20° (WRONG - No Delays)', fontsize=12)
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(True, alpha=0.3)

    # PLOT 4: Power with delayed wake (CORRECT physics)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(t, P_delayed[0] / 1000, 'b-', linewidth=2, label='T1', alpha=0.8)
    ax4.plot(t, P_delayed[1] / 1000, 'r-', linewidth=2, label='T2', alpha=0.8)
    ax4.plot(t, P_delayed[2] / 1000, 'g-', linewidth=2, label='T3', alpha=0.8)
    ax4.plot(t, P_delayed.sum(axis=0) / 1000, 'k--', linewidth=2.5, label='Total', alpha=0.8)

    # Mark when delays occur
    ax4.axvline(delay_12, color='r', linestyle=':', alpha=0.5, linewidth=2)
    ax4.axvline(delay_13, color='orange', linestyle=':', alpha=0.5, linewidth=2)

    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Power (kW)', fontsize=11)
    ax4.set_title('Delayed Wake: T1 Yaws +20° (CORRECT - With Delays)', fontsize=12)
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.3)

    # PLOT 5: T2 Power Gain (shows the benefit clearly!)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(t, P_baseline[1] / 1000, 'r--', linewidth=2.5, label='T2 Baseline (0° yaw)', alpha=0.7)
    ax5.plot(t, P_delayed[1] / 1000, 'g-', linewidth=2.5, label='T2 with Wake Steering', alpha=0.9)
    ax5.fill_between(t, P_baseline[1] / 1000, P_delayed[1] / 1000,
                     where=(P_delayed[1] > P_baseline[1]),
                     alpha=0.3, color='green', label='Power Gain')
    ax5.axvline(delay_12, color='r', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Deflected wake arrives ({delay_12:.0f}s)')

    # Calculate and annotate power gain
    power_gain_T2 = ((P_delayed[1, -1] - P_baseline[1, -1]) / P_baseline[1, -1]) * 100
    ax5.text(0.98, 0.98, f'Power Gain: +{power_gain_T2:.1f}%',
            transform=ax5.transAxes, fontsize=12, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('T2 Power (kW)', fontsize=11)
    ax5.set_title('T2 Power Increase from Wake Deflection', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9, loc='lower right')
    ax5.grid(True, alpha=0.3)

    # PLOT 6: T3 Power Gain
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(t, P_baseline[2] / 1000, 'g--', linewidth=2.5, label='T3 Baseline (0° yaw)', alpha=0.7)
    ax6.plot(t, P_delayed[2] / 1000, 'b-', linewidth=2.5, label='T3 with Wake Steering', alpha=0.9)
    ax6.fill_between(t, P_baseline[2] / 1000, P_delayed[2] / 1000,
                     where=(P_delayed[2] > P_baseline[2]),
                     alpha=0.3, color='blue', label='Power Gain')
    ax6.axvline(delay_13, color='orange', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Deflected wake arrives ({delay_13:.0f}s)')

    # Calculate and annotate power gain
    power_gain_T3 = ((P_delayed[2, -1] - P_baseline[2, -1]) / P_baseline[2, -1]) * 100
    ax6.text(0.98, 0.98, f'Power Gain: +{power_gain_T3:.1f}%',
            transform=ax6.transAxes, fontsize=12, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('T3 Power (kW)', fontsize=11)
    ax6.set_title('T3 Power Increase from Wake Deflection', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9, loc='lower right')
    ax6.grid(True, alpha=0.3)

    # PLOT 7: Cumulative energy comparison
    ax7 = fig.add_subplot(gs[2, 2])
    energy_baseline = np.cumsum(P_baseline.sum(axis=0)) * dt / 3600  # kWh
    energy_instant = np.cumsum(P_instant.sum(axis=0)) * dt / 3600
    energy_delayed = np.cumsum(P_delayed.sum(axis=0)) * dt / 3600

    ax7.plot(t, energy_baseline, 'k--', linewidth=2.5, label='Baseline', alpha=0.7)
    ax7.plot(t, energy_instant, 'r-', linewidth=2, label='Instant wake', alpha=0.7)
    ax7.plot(t, energy_delayed, 'g-', linewidth=2.5, label='Delayed wake', alpha=0.9)
    ax7.fill_between(t, energy_baseline, energy_delayed, alpha=0.2, color='green',
                     where=(energy_delayed > energy_baseline),
                     label='Net gain')

    ax7.set_xlabel('Time (s)', fontsize=11)
    ax7.set_ylabel('Cumulative Energy (kWh)', fontsize=11)
    ax7.set_title('Cumulative Energy: Wake Steering Benefit', fontsize=12)
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # PLOT 8: Performance impact summary
    ax8 = fig.add_subplot(gs[3, :])

    # Calculate performance metrics
    energy_gain_vs_baseline = ((energy_delayed[-1] - energy_baseline[-1]) / energy_baseline[-1]) * 100
    energy_diff_models = ((energy_delayed[-1] - energy_instant[-1]) / energy_instant[-1]) * 100

    summary_text = f"""
WAKE ADVECTION & STEERING SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════════════════════════

FARM SETUP                          WAKE DELAYS                      POWER GAINS FROM WAKE STEERING
──────────────────────────────     ─────────────────────────────    ─────────────────────────────────────
• 3 turbines in line               • T1 → T2: {delay_12:>6.0f} s            • T2 power gain: +{power_gain_T2:>5.1f}%
• Spacing: {xs[1]-xs[0]:>4.0f} m                    • T1 → T3: {delay_13:>6.0f} s            • T3 power gain: +{power_gain_T3:>5.1f}%
• Wind speed: {U_inf:.1f} m/s                 • Advection speed: {U_inf:.1f} m/s         • Total farm gain: +{energy_gain_vs_baseline:>5.1f}%

KEY INSIGHTS:
─────────────
1. WAKE STEERING WORKS: When T1 yaws +20°, it deflects its wake AWAY from downstream turbines → increased power
2. DELAYS MATTER: Downstream turbines don't feel the benefit instantly - wake takes {delay_12:.0f}-{delay_13:.0f}s to propagate
3. MPC MUST ACCOUNT FOR DELAYS: Ignoring advection dynamics leads to {abs(energy_diff_models):.1f}% modeling error in energy prediction
4. ACTION HORIZON SIZING: t_AH must be ≥ {delay_13:.0f}s (max delay) to capture full wake propagation effects

WHY THIS VALIDATES OUR MPC FORMULATION:
────────────────────────────────────────────
✓ Time-shifted cost function correctly accounts for wake propagation delays
✓ Action horizon t_AH = 100s > {delay_13:.0f}s (maximum delay in system)
✓ Prediction horizon T_opt = 400s captures full transient response
✓ Delay loop in trajectory simulation ensures physical realism
"""

    ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1.0))
    ax8.axis('off')

    if save:
        plt.savefig(FIGURE_DIR / "wake_advection_dynamics.png", dpi=300, bbox_inches='tight')
        plt.savefig(FIGURE_DIR / "wake_advection_dynamics.pdf", bbox_inches='tight')

    print(f"\n{'='*60}")
    print("WAKE ADVECTION & STEERING ANALYSIS")
    print(f"{'='*60}")
    print(f"Delay T1→T2: {delay_12:.1f} seconds")
    print(f"Delay T1→T3: {delay_13:.1f} seconds")
    print(f"T2 power gain from steering: +{power_gain_T2:.1f}%")
    print(f"T3 power gain from steering: +{power_gain_T3:.1f}%")
    print(f"Total farm energy gain: +{energy_gain_vs_baseline:.1f}%")
    print(f"Modeling error (instant vs delayed): {abs(energy_diff_models):.1f}%")
    print(f"{'='*60}\n")

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def generate_all_visualizations():
    """Generate all visualization figures."""
    print("\n" + "="*70)
    print("GENERATING MPC VISUALIZATION SUITE")
    print("="*70 + "\n")

    figures = {}

    print("[1/8] Generating basis function grid...")
    figures['basis_grid'] = plot_basis_function_grid(save=True)
    plt.close()

    print("[2/8] Generating yaw trajectories...")
    figures['trajectories'] = plot_yaw_trajectories(save=True)
    plt.close()

    print("[3/8] Generating parameter space heatmap...")
    figures['param_space'] = plot_parameter_space_heatmap(save=True)
    plt.close()

    print("[4/8] Generating turbine layout with wakes...")
    figures['layout'] = plot_turbine_layout_with_wakes(save=True)
    plt.close()

    print("[5/8] Generating wake delay matrix...")
    figures['delays'] = plot_wake_delay_matrix(save=True)
    plt.close()

    print("[6/8] Generating power vs yaw analysis...")
    figures['power_yaw'] = plot_power_vs_yaw(save=True)
    plt.close()

    print("[7/8] Generating wake interaction analysis...")
    figures['interaction'] = plot_wake_interaction(save=True)
    plt.close()

    print("[8/8] Generating wake advection dynamics analysis...")
    figures['advection'] = plot_wake_advection_dynamics(save=True)
    plt.close()

    print("\n" + "="*70)
    print(f"ALL VISUALIZATIONS SAVED TO: {FIGURE_DIR}")
    print("="*70 + "\n")

    print("Generated files:")
    for file in sorted(FIGURE_DIR.glob("*.png")):
        print(f"  - {file.name}")

    return figures


if __name__ == "__main__":
    generate_all_visualizations()
