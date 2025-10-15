#!/usr/bin/env python3
"""
Visual Unit Tests for Wind Farm Optimization Code
-------------------------------------------------
These tests generate plots for manual verification of functionality.
Run each test function individually to inspect the behavior visually.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

# Add the directory containing the main code to path
# Adjust this path based on your file structure
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all functions and classes from the main module
# Replace 'main_module' with your actual module name
from MPC import (
    sat01, psi, yaw_traj, YawCache, WindFarmModel, 
    run_farm_delay_loop_optimized, farm_energy, optimize_farm_back2front
)


def test_sat01_function():
    """Test the saturation function sat01."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Test 1: Basic functionality
    x = np.linspace(-0.5, 1.5, 1000)
    y = sat01(x)
    
    ax1.plot(x, y, 'b-', linewidth=2, label='sat01(x)')
    ax1.plot(x, x, 'k--', alpha=0.5, label='y=x (reference)')
    ax1.axhline(y=0, color='r', linestyle=':', alpha=0.5)
    ax1.axhline(y=1, color='r', linestyle=':', alpha=0.5)
    ax1.axvline(x=0, color='g', linestyle=':', alpha=0.5)
    ax1.axvline(x=1, color='g', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Input x')
    ax1.set_ylabel('Output sat01(x)')
    ax1.set_title('Saturation Function Behavior')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test 2: Array input
    x_array = np.array([-1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2])
    y_array = sat01(x_array)
    
    ax2.stem(x_array, y_array, basefmt=' ', label='sat01(x)')
    ax2.plot(x_array, x_array, 'ro', alpha=0.5, label='Original values')
    ax2.axhline(y=0, color='r', linestyle=':', alpha=0.5)
    ax2.axhline(y=1, color='r', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Input x')
    ax2.set_ylabel('Output sat01(x)')
    ax2.set_title('Saturation on Discrete Points')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('TEST: sat01 Function - Verify values are clipped to [0,1]', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_psi_basis_function():
    """Test the psi basis function behavior."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    t_AH = 100.0
    r_gamma = 0.3
    t = np.linspace(0, 150, 500)
    
    # Test different o1, o2 combinations
    test_cases = [
        (0.25, 0.25), (0.5, 0.5), (0.75, 0.75),  # Different o1, fixed o2=0.5
        (0.25, 0.0), (0.5, 0.0), (0.75, 0.0),    # o2 = 0
        (0.25, 1.0), (0.5, 1.0), (0.75, 1.0),    # o2 = 1
    ]
    
    for idx, (o1, o2) in enumerate(test_cases):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        psi_values = [psi(o1, o2, ti, t_AH, r_gamma) for ti in t]
        
        ax.plot(t, psi_values, 'b-', linewidth=2)
        ax.axvline(x=t_AH, color='r', linestyle='--', alpha=0.5, label=f't_AH={t_AH}')
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax.set_xlabel('Time t')
        ax.set_ylabel('ψ(t)')
        ax.set_title(f'o1={o1:.2f}, o2={o2:.2f}')
        ax.grid(True, alpha=0.3)
        
        # Mark key points
        if o1 != 0.5:
            t_switch = o2 * t_AH
            ax.axvline(x=t_switch, color='g', linestyle=':', alpha=0.5, label=f't_switch={t_switch:.1f}')
        
        if idx == 2:
            ax.legend(fontsize=8)
    
    plt.suptitle(f'TEST: Psi Basis Function - Different Parameter Combinations (t_AH={t_AH}, r_γ={r_gamma})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_yaw_trajectory():
    """Test yaw trajectory generation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Common parameters
    dt = 1.0
    T_total = 200.0
    t_AH = 100.0
    r_gamma = 0.3
    
    # Test cases
    test_cases = [
        {"gamma0": 0.0, "o1": 0.75, "o2": 0.2, "title": "Positive yaw from 0°"},
        {"gamma0": 0.0, "o1": 0.25, "o2": 0.2, "title": "Negative yaw from 0°"},
        {"gamma0": 15.0, "o1": 0.5, "o2": 0.5, "title": "No change from 15°"},
        {"gamma0": -10.0, "o1": 0.9, "o2": 0.8, "title": "Large positive change from -10°"},
    ]
    
    for idx, test in enumerate(test_cases):
        ax = axes[idx]
        t, gamma = yaw_traj(test["gamma0"], test["o1"], test["o2"], t_AH, r_gamma, dt, T_total)
        
        ax.plot(t, gamma, 'b-', linewidth=2, label='Yaw angle')
        ax.axhline(y=test["gamma0"], color='g', linestyle='--', alpha=0.5, label=f'Initial: {test["gamma0"]}°')
        ax.axvline(x=t_AH, color='r', linestyle='--', alpha=0.5, label=f'Action horizon: {t_AH}s')
        
        # Mark final value
        final_yaw = gamma[-1]
        ax.axhline(y=final_yaw, color='purple', linestyle=':', alpha=0.5, label=f'Final: {final_yaw:.1f}°')
        
        # Add change indicator
        change = final_yaw - test["gamma0"]
        ax.text(0.05, 0.95, f'Δγ = {change:.1f}°', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Yaw Angle [deg]')
        ax.set_title(test["title"])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('TEST: Yaw Trajectory Generation - Verify smooth transitions and correct endpoints', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_yaw_cache():
    """Test YawCache functionality and quantization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Test 1: Quantization behavior
    cache = YawCache(maxsize=100, quant=0.5, wind_quant=1.0)
    
    # Generate test yaw angles
    yaw_test = np.linspace(-2, 2, 100)
    quantized_yaws = []
    
    for y in yaw_test:
        key = cache._key((y,), 8.0, 0.06, 270.0)
        quantized_yaws.append(key[0][0])
    
    ax1.plot(yaw_test, yaw_test, 'b-', alpha=0.3, label='Original')
    ax1.plot(yaw_test, quantized_yaws, 'r.', markersize=4, label='Quantized')
    ax1.set_xlabel('Original Yaw [deg]')
    ax1.set_ylabel('Quantized Yaw [deg]')
    ax1.set_title(f'Yaw Quantization (quant={cache.quant}°)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test 2: Cache hit rate over time
    cache2 = YawCache(maxsize=50, quant=0.1, wind_quant=0.5)
    
    hit_rates = []
    n_calls = []
    
    # Simulate repeated calls with some variation
    np.random.seed(42)
    for i in range(200):
        yaws = tuple(np.random.normal(0, 5, 4))  # 4 turbines
        wind = 270 + np.random.normal(0, 2)
        
        # 70% chance of reusing recent configuration
        if i > 20 and np.random.random() < 0.7:
            yaws = tuple(np.round(np.array(yaws) / cache2.quant) * cache2.quant)
            wind = round(wind / cache2.wind_quant) * cache2.wind_quant
        
        result = cache2.get(yaws, 8.0, 0.06, wind)
        if result is None:
            cache2.put(yaws, 8.0, 0.06, wind, np.random.rand())
        
        if i > 0:
            hit_rate = cache2.hits / (cache2.hits + cache2.misses) * 100
            hit_rates.append(hit_rate)
            n_calls.append(i)
    
    ax2.plot(n_calls, hit_rates, 'g-', linewidth=2)
    ax2.set_xlabel('Number of Calls')
    ax2.set_ylabel('Hit Rate [%]')
    ax2.set_title('Cache Hit Rate Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.95, 0.05, f'Final: {hit_rates[-1]:.1f}%', transform=ax2.transAxes,
             ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    # Test 3: Memory usage vs cache size
    cache_sizes = [10, 50, 100, 500, 1000, 2000]
    memory_estimates = []
    
    for size in cache_sizes:
        test_cache = YawCache(maxsize=size)
        # Fill cache to capacity
        for i in range(size):
            yaws = tuple(np.random.rand(4) * 30)
            test_cache.put(yaws, 8.0, 0.06, 270.0, np.random.rand())
        memory_estimates.append(test_cache.memory_size_mb())
    
    ax3.plot(cache_sizes, memory_estimates, 'o-', linewidth=2, markersize=8)
    ax3.set_xlabel('Cache Size (max entries)')
    ax3.set_ylabel('Memory Usage [MB]')
    ax3.set_title('Cache Memory Scaling')
    ax3.grid(True, alpha=0.3)
    
    # Test 4: Effect of quantization on cache efficiency
    quant_values = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    unique_keys = []
    
    for q in quant_values:
        test_cache = YawCache(maxsize=1000, quant=q)
        seen_keys = set()
        
        # Generate 1000 random configurations
        for _ in range(1000):
            yaws = tuple(np.random.normal(0, 10, 4))
            key = test_cache._key(yaws, 8.0, 0.06, 270.0)
            seen_keys.add(key[0])  # Just track yaw part
        
        unique_keys.append(len(seen_keys))
    
    ax4.semilogx(quant_values, unique_keys, 'ro-', linewidth=2, markersize=8)
    ax4.set_xlabel('Quantization Step [deg]')
    ax4.set_ylabel('Unique Keys (from 1000 samples)')
    ax4.set_title('Effect of Quantization on Cache Diversity')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('TEST: YawCache Functionality - Verify quantization, hit rates, and memory usage', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_wind_farm_model_sorting():
    """Test WindFarmModel turbine sorting for different wind directions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Create a non-trivial layout
    D = 80.0
    x_pos = [0, 7*D, 14*D, 0, 7*D, 14*D]
    y_pos = [0, D, -D, 3*D, 4*D, 2*D]
    
    wind_directions = [270, 225, 180, 135, 90, 45]
    
    for idx, wd in enumerate(wind_directions):
        ax = axes[idx]
        
        model = WindFarmModel(x_pos, y_pos, D=D, U_inf=8.0, TI=0.06, wd=wd)
        
        # Plot original layout
        ax.scatter(x_pos, y_pos, s=200, c='blue', marker='o', edgecolors='black', 
                  linewidth=2, label='Original order', zorder=3)
        
        # Label original indices
        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            ax.annotate(f'{i}', (x, y), ha='center', va='center', color='white', 
                       fontweight='bold', fontsize=10)
        
        # Draw wind direction arrow
        arrow_length = 3 * D
        wd_rad = np.deg2rad(270 - wd)
        ax.arrow(7*D, 2*D, arrow_length*np.cos(wd_rad), arrow_length*np.sin(wd_rad),
                head_width=D/2, head_length=D/3, fc='red', ec='red', alpha=0.5)
        
        # Show sorted order
        sorted_text = ', '.join([str(i) for i in model.sorted_indices])
        ax.text(0.02, 0.98, f'Sorted order: [{sorted_text}]', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Verify upstream/downstream relationship
        for i in range(len(model.sorted_indices)-1):
            idx1 = model.sorted_indices[i]
            idx2 = model.sorted_indices[i+1]
            ax.plot([x_pos[idx1], x_pos[idx2]], [y_pos[idx1], y_pos[idx2]], 
                   'g--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f'Wind Direction: {wd}°')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add north arrow
        ax.annotate('N', xy=(0.9, 0.9), xytext=(0.9, 0.95), 
                   xycoords='axes fraction', textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', lw=1.5), ha='center')
    
    plt.suptitle('TEST: Turbine Sorting by Wind Direction - Verify upstream/downstream ordering', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_delay_calculation():
    """Test delay calculation between turbines."""
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    fig = plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4, projection='polar')


    # Simple line of turbines
    D = 80.0
    x_pos = [0, 7*D, 14*D, 21*D]
    y_pos = [0, 0, 0, 0]
    
    # Test different wind speeds
    U_advs = [5.0, 8.0, 12.0, 20.0]
    colors = ['blue', 'green', 'orange', 'red']
    
    model = WindFarmModel(x_pos, y_pos, D=D, U_inf=8.0, TI=0.06, wd=270.0)
    
    # Plot 1: Delays vs wind speed
    for U_adv, color in zip(U_advs, colors):
        model.update_conditions(8.0, 0.06, 270.0, U_adv=U_adv)
        delays_to_last = [model.delays[i, -1] for i in range(3)]
        ax1.plot(range(3), delays_to_last, 'o-', color=color, 
                label=f'U_adv={U_adv} m/s', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Upstream Turbine Index')
    ax1.set_ylabel('Delay to Last Turbine [s]')
    ax1.set_title('Wake Propagation Delays (Line of Turbines)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(3))
    
    # Plot 2: Delay matrix heatmap
    model.update_conditions(8.0, 0.06, 270.0, U_adv=8.0)
    im = ax2.imshow(model.delays, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Downstream Turbine')
    ax2.set_ylabel('Upstream Turbine')
    ax2.set_title('Delay Matrix [s] (U_adv=8 m/s)')
    
    # Add values to heatmap
    for i in range(4):
        for j in range(4):
            text = ax2.text(j, i, f'{model.delays[i, j]:.0f}', 
                           ha='center', va='center', color='white' if model.delays[i, j] > 50 else 'black')
    
    plt.colorbar(im, ax=ax2, label='Delay [s]')
    
    # Plot 3: Complex layout delays
    x_pos2 = [0, 5*D, 10*D, 3*D, 8*D]
    y_pos2 = [0, 2*D, -D, -3*D, 4*D]
    model2 = WindFarmModel(x_pos2, y_pos2, D=D, U_inf=8.0, TI=0.06, wd=270.0, U_adv=8.0)
    
    ax3.scatter(model2.xs, model2.ys, s=200, c=range(5), cmap='coolwarm', 
               edgecolors='black', linewidth=2)
    
    # Show delays with arrows
    for i in range(5):
        for j in range(i+1, 5):
            if model2.delays[i, j] > 0:
                ax3.annotate('', xy=(model2.xs[j], model2.ys[j]), 
                           xytext=(model2.xs[i], model2.ys[i]),
                           arrowprops=dict(arrowstyle='->', alpha=0.5, 
                                         color='gray', lw=1))
                
                # Add delay text
                mid_x = (model2.xs[i] + model2.xs[j]) / 2
                mid_y = (model2.ys[i] + model2.ys[j]) / 2
                ax3.text(mid_x, mid_y, f'{model2.delays[i, j]:.0f}s', 
                        fontsize=8, ha='center', 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    ax3.set_title('Delay Propagation in Complex Layout')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Effect of wind direction on delays
    wind_dirs = np.linspace(0, 360, 37)
    max_delays = []
    
    for wd in wind_dirs:
        model2.update_conditions(8.0, 0.06, wd, U_adv=8.0)
        max_delays.append(np.max(model2.delays))


    ax4.plot(np.deg2rad(wind_dirs), max_delays, 'b-', linewidth=2)

    # ax4.polar(np.deg2rad(wind_dirs), max_delays, 'b-', linewidth=2)
    ax4.set_theta_direction(-1)
    ax4.set_theta_offset(np.pi/2)
    ax4.set_title('Maximum Delay vs Wind Direction', pad=20)
    ax4.set_rlabel_position(45)
    
    plt.suptitle('TEST: Wake Delay Calculations - Verify physical correctness', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_power_calculation_with_yaw():
    """Test power calculation with different yaw angles."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Simple 2-turbine case
    D = 80.0
    x_pos = [0, 7*D]
    y_pos = [0, 0]
    
    model = WindFarmModel(x_pos, y_pos, D=D, U_inf=8.0, TI=0.06, wd=270.0)
    
    # Test 1: Upstream turbine yaw effect
    yaw_angles = np.linspace(-30, 30, 61)
    upstream_powers = []
    downstream_powers = []
    
    for yaw in yaw_angles:
        yaws = np.array([yaw, 0])
        powers = model.farm_power(yaws)
        upstream_powers.append(powers[0])
        downstream_powers.append(powers[1])
    
    ax1.plot(yaw_angles, np.array(upstream_powers)/1e3, 'b-', linewidth=2, label='Upstream (T0)')
    ax1.plot(yaw_angles, np.array(downstream_powers)/1e3, 'r-', linewidth=2, label='Downstream (T1)')
    ax1.set_xlabel('Upstream Yaw Angle [deg]')
    ax1.set_ylabel('Power [kW]')
    ax1.set_title('Effect of Upstream Yaw on Power')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test 2: Power gain heatmap
    yaw1_range = np.linspace(-25, 25, 21)
    yaw2_range = np.linspace(-25, 25, 21)
    
    baseline_power = np.sum(model.farm_power(np.array([0, 0])))
    power_gain_map = np.zeros((len(yaw2_range), len(yaw1_range)))
    
    for i, yaw1 in enumerate(yaw1_range):
        for j, yaw2 in enumerate(yaw2_range):
            yaws = np.array([yaw1, yaw2])
            total_power = np.sum(model.farm_power(yaws))
            power_gain_map[j, i] = (total_power - baseline_power) / baseline_power * 100
    
    im = ax2.imshow(power_gain_map, extent=[yaw1_range[0], yaw1_range[-1], 
                                           yaw2_range[0], yaw2_range[-1]],
                   origin='lower', cmap='RdBu_r', vmin=-10, vmax=10)
    ax2.set_xlabel('Upstream Yaw [deg]')
    ax2.set_ylabel('Downstream Yaw [deg]')
    ax2.set_title('Total Power Gain [%]')
    plt.colorbar(im, ax=ax2)
    
    # Mark optimal point
    max_idx = np.unravel_index(np.argmax(power_gain_map), power_gain_map.shape)
    ax2.plot(yaw1_range[max_idx[1]], yaw2_range[max_idx[0]], 'w*', markersize=15)
    
    # Test 3: Yaw penalty function (if enabled)
    model_with_penalty = WindFarmModel(x_pos, y_pos, D=D, apply_yaw_penalty=True)
    model_no_penalty = WindFarmModel(x_pos, y_pos, D=D, apply_yaw_penalty=False)
    
    yaw_test = np.linspace(-40, 40, 81)
    power_with_penalty = []
    power_no_penalty = []
    
    for yaw in yaw_test:
        yaws = np.array([yaw, 0])
        power_with_penalty.append(model_with_penalty.farm_power(yaws)[0])
        power_no_penalty.append(model_no_penalty.farm_power(yaws)[0])
    
    ax3.plot(yaw_test, np.array(power_no_penalty)/1e3, 'b--', linewidth=2, 
             label='Without penalty', alpha=0.7)
    ax3.plot(yaw_test, np.array(power_with_penalty)/1e3, 'r-', linewidth=2, 
             label='With penalty (Eq. 5)')
    ax3.axvline(x=-33, color='k', linestyle=':', alpha=0.5)
    ax3.axvline(x=33, color='k', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Yaw Angle [deg]')
    ax3.set_ylabel('Upstream Turbine Power [kW]')
    ax3.set_title('Effect of Yaw Penalty Function')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Test 4: 4-turbine optimization result
    x_pos4 = [0, 7*D, 14*D, 21*D]
    y_pos4 = [0, 0, 0, 0]
    model4 = WindFarmModel(x_pos4, y_pos4, D=D)
    
    # Compare different yaw strategies
    strategies = {
        'Baseline': [0, 0, 0, 0],
        'Greedy T0': [25, 0, 0, 0],
        'Alternating': [20, 0, 20, 0],
        'Progressive': [25, 15, 10, 0],
        'Optimal (approx)': [23, 12, 8, 0]
    }
    
    results = []
    for name, yaws in strategies.items():
        powers = model4.farm_power(np.array(yaws))
        total = np.sum(powers)
        results.append((name, total, powers))
    
    # Sort by total power
    results.sort(key=lambda x: x[1], reverse=True)
    
    names = [r[0] for r in results]
    totals = [r[1]/1e6 for r in results]
    
    bars = ax4.bar(range(len(names)), totals, color=['gold' if i==0 else 'skyblue' for i in range(len(names))])
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.set_ylabel('Total Power [MW]')
    ax4.set_title('Comparison of Yaw Strategies (4 Turbines)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{total:.3f}', ha='center', va='bottom')
    
    plt.suptitle('TEST: Power Calculations with Yaw Control - Verify wake steering effects', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_optimization_trajectories():
    """Test the optimization algorithm behavior."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # Setup
    D = 80.0
    x_pos = [0, 7*D, 14*D]
    y_pos = [0, 0, 0]
    
    model = WindFarmModel(x_pos, y_pos, D=D, U_inf=8.0, TI=0.06, wd=270.0)
    
    # Test different initial conditions
    test_cases = [
        {"current_yaws": [0, 0, 0], "title": "Starting from neutral"},
        {"current_yaws": [15, 10, 5], "title": "Starting from positive yaws"},
        {"current_yaws": [-10, -5, 0], "title": "Starting from negative yaws"},
    ]
    
    for idx, test in enumerate(test_cases):
        # Run optimization
        current_yaws_sorted = np.array(test["current_yaws"])
        
        opt_params = optimize_farm_back2front(
            model, current_yaws_sorted,
            r_gamma=0.3, t_AH=100.0,
            dt_opt=5.0, T_opt=200.0,
            maxfun=50, seed=idx
        )
        
        # Generate trajectories
        t, trajectories, P = run_farm_delay_loop_optimized(
            model, opt_params, current_yaws_sorted,
            r_gamma=0.3, t_AH=100.0, dt=1.0, T=200.0
        )
        
        # Plot trajectories
        ax1 = fig.add_subplot(gs[idx, 0])
        for i, traj in enumerate(trajectories):
            ax1.plot(t, traj, linewidth=2, label=f'Turbine {i}')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Yaw Angle [deg]')
        ax1.set_title(f'Trajectories: {test["title"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=100, color='r', linestyle='--', alpha=0.5, label='t_AH')
        
        # Plot power evolution
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.plot(t, np.sum(P, axis=0)/1e6, 'k-', linewidth=2, label='Total')
        for i in range(3):
            ax2.plot(t, P[i, :]/1e6, '--', alpha=0.7, label=f'T{i}')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Power [MW]')
        ax2.set_title('Power Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot optimization parameters
        ax3 = fig.add_subplot(gs[idx, 2])
        turbines = range(3)
        o1_values = opt_params[:, 0]
        o2_values = opt_params[:, 1]
        
        x = np.arange(3)
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, o1_values, width, label='o1', alpha=0.8)
        bars2 = ax3.bar(x + width/2, o2_values, width, label='o2', alpha=0.8)
        
        ax3.set_xlabel('Turbine')
        ax3.set_ylabel('Parameter Value')
        ax3.set_title('Optimized Parameters')
        ax3.set_xticks(x)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('TEST: Back-to-Front Optimization - Verify algorithm produces reasonable trajectories', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_farm_energy_integration():
    """Test energy calculation and integration methods."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Test 1: Compare integration methods
    t = np.linspace(0, 100, 101)
    
    # Different power profiles
    constant_power = np.ones_like(t) * 2.0  # 2 MW constant
    linear_power = t / 50  # Linear increase
    sine_power = 1 + 0.5 * np.sin(2 * np.pi * t / 20)  # Oscillating
    step_power = np.where(t < 50, 1.0, 3.0)  # Step change
    
    profiles = [
        ("Constant", constant_power),
        ("Linear", linear_power),
        ("Sinusoidal", sine_power),
        ("Step", step_power)
    ]
    
    for name, power in profiles:
        ax1.plot(t, power, linewidth=2, label=name)
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Power [MW]')
    ax1.set_title('Test Power Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test 2: Energy accumulation
    for name, power in profiles:
        cumulative_energy = []
        for i in range(1, len(t)):
            energy = np.trapezoid(power[:i+1], t[:i+1]) / 3600  # Convert to MWh
            cumulative_energy.append(energy)
        
        ax2.plot(t[1:], cumulative_energy, linewidth=2, label=name)
    
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Cumulative Energy [MWh]')
    ax2.set_title('Energy Integration Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Test 3: Multi-turbine energy calculation
    n_turbines = 4
    t_test = np.linspace(0, 200, 201)
    
    # Create synthetic power matrix with delays
    P_matrix = np.zeros((n_turbines, len(t_test)))
    
    for i in range(n_turbines):
        # Each turbine starts producing power with a delay
        delay_steps = i * 20
        P_matrix[i, delay_steps:] = 1.5 * (1 + 0.3 * np.sin(2 * np.pi * t_test[delay_steps:] / 50))
    
    # Plot individual turbine powers
    for i in range(n_turbines):
        ax3.plot(t_test, P_matrix[i, :], linewidth=2, label=f'Turbine {i}')
    
    ax3.plot(t_test, np.sum(P_matrix, axis=0), 'k--', linewidth=2, label='Total')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Power [MW]')
    ax3.set_title('Multi-turbine Power with Delays')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Test 4: Energy calculation accuracy
    dt_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    errors = []
    
    # True energy (analytical) for sine wave
    true_energy = 100  # For 1 MW average over 100s
    
    for dt in dt_values:
        t_discrete = np.arange(0, 100 + dt, dt)
        power_discrete = 1 + 0.5 * np.sin(2 * np.pi * t_discrete / 20)
        
        calc_energy = farm_energy(power_discrete.reshape(1, -1), t_discrete) / 3600
        error = abs(calc_energy - true_energy) / true_energy * 100
        errors.append(error)
    
    ax4.loglog(dt_values, errors, 'ro-', linewidth=2, markersize=8)
    ax4.set_xlabel('Time Step [s]')
    ax4.set_ylabel('Energy Calculation Error [%]')
    ax4.set_title('Integration Accuracy vs Time Step')
    ax4.grid(True, alpha=0.3, which='both')
    
    # Add reference line for quadratic convergence
    reference_errors = np.array(dt_values)**2 * errors[0] / dt_values[0]**2
    ax4.loglog(dt_values, reference_errors, 'b--', alpha=0.5, label='O(dt²) reference')
    ax4.legend()
    
    plt.suptitle('TEST: Energy Integration Methods - Verify correctness and accuracy', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def run_all_tests():
    """Run all visual tests with user interaction."""
    tests = [
        ("Saturation Function (sat01)", test_sat01_function),
        ("Psi Basis Function", test_psi_basis_function),
        ("Yaw Trajectory Generation", test_yaw_trajectory),
        ("Yaw Cache Functionality", test_yaw_cache),
        ("Wind Farm Model Sorting", test_wind_farm_model_sorting),
        ("Delay Calculations", test_delay_calculation),
        ("Power Calculations with Yaw", test_power_calculation_with_yaw),
        ("Optimization Trajectories", test_optimization_trajectories),
        ("Energy Integration", test_farm_energy_integration),
    ]
    
    print("=" * 60)
    print("VISUAL UNIT TESTS FOR WIND FARM OPTIMIZATION")
    print("=" * 60)
    print("\nAvailable tests:")
    
    for i, (name, _) in enumerate(tests):
        print(f"{i+1}. {name}")
    
    print(f"\n0. Run all tests")
    print("q. Quit")
    
    while True:
        choice = input("\nEnter test number (or 'q' to quit): ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == '0':
            for name, test_func in tests:
                print(f"\n{'='*60}")
                print(f"Running: {name}")
                print(f"{'='*60}")
                test_func()
                input("\nPress Enter to continue to next test...")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(tests):
                    name, test_func = tests[idx]
                    print(f"\n{'='*60}")
                    print(f"Running: {name}")
                    print(f"{'='*60}")
                    test_func()
                else:
                    print("Invalid test number!")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")


if __name__ == "__main__":
    # You can either run individual tests or use the interactive menu
    
    # Option 1: Run specific tests directly
    # test_yaw_trajectory()
    # test_wind_farm_model_sorting()
    
    # Option 2: Run interactive menu
    run_all_tests()