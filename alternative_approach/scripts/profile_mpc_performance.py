"""
Profile MPC performance to identify bottlenecks.
"""

import time
import numpy as np
from py_wake.examples.data.hornsrev1 import V80
from mpcrl import MPCenv, make_config
from mpcrl.mpc import optimize_farm_back2front

def profile_single_mpc_step():
    """Profile a single MPC optimization step with detailed timing."""

    print("\n" + "="*70)
    print("MPC PERFORMANCE PROFILING")
    print("="*70)

    # Setup environment (same as training)
    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])

    env = MPCenv(
        turbine=V80(),
        x_pos=x_pos,
        y_pos=y_pos,
        config=make_config(),
        ws_scaling_min=6, ws_scaling_max=15,
        wd_scaling_min=250, wd_scaling_max=290,
        ti_scaling_min=0.01, ti_scaling_max=0.15,
        turbtype="None",
        dt_env=30,
        dt_sim=10,
        yaw_step_sim=10*0.3,
        yaw_init='Zeros',
    )

    obs, info = env.reset()

    # Warm up (first call is slower due to compilation/caching)
    print("\n[WARM-UP] Running first step (includes compilation overhead)...")
    t_start = time.time()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    t_warmup = time.time() - t_start
    print(f"  Warm-up time: {t_warmup:.2f}s")

    # Profile multiple steps to get average
    n_steps = 5
    step_times = []
    cache_stats_list = []

    print(f"\n[PROFILING] Running {n_steps} steps with random actions...")

    for i in range(n_steps):
        action = env.action_space.sample()

        t_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        t_step = time.time() - t_start

        step_times.append(t_step)
        cache_stats = env.mpc_model.cache.get_stats()
        cache_stats_list.append(cache_stats)

        print(f"  Step {i+1}: {t_step:.2f}s - {cache_stats}")

    avg_time = np.mean(step_times)
    std_time = np.std(step_times)

    print("\n" + "="*70)
    print("TIMING RESULTS")
    print("="*70)
    print(f"Average time per step: {avg_time:.2f}s ± {std_time:.2f}s")
    print(f"Steps per minute: {60/avg_time:.1f}")
    print(f"With 6 envs (parallel): ~{6*60/avg_time:.1f} steps/min")
    print(f"For 100k timesteps: ~{100000/(6*60/avg_time):.1f} minutes = ~{100000/(6*60*60/avg_time):.1f} hours")

    # Extract cache statistics
    final_cache = env.mpc_model.cache
    total_calls = final_cache.hits + final_cache.misses
    hit_rate = (final_cache.hits / total_calls * 100) if total_calls > 0 else 0

    print("\n" + "="*70)
    print("CACHE STATISTICS")
    print("="*70)
    print(f"Cache size: {final_cache.maxsize} entries")
    print(f"Cache quantization: {final_cache.quant}°")
    print(f"Total power calculations: {total_calls}")
    print(f"Cache hits: {final_cache.hits} ({hit_rate:.1f}%)")
    print(f"Cache misses (PyWake calls): {final_cache.misses}")
    print(f"Memory usage: {final_cache.memory_size_mb():.2f} MB")

    return avg_time, hit_rate


def profile_optimization_params():
    """Test different optimization parameter settings."""

    print("\n" + "="*70)
    print("TESTING DIFFERENT OPTIMIZATION PARAMETERS")
    print("="*70)

    x_pos = np.array([0, 500, 1000])
    y_pos = np.array([0, 0, 0])

    # Test configurations
    configs = [
        {"maxfun": 50, "T_opt": 500, "dt_opt": 10, "label": "Current (baseline)"},
        {"maxfun": 20, "T_opt": 500, "dt_opt": 10, "label": "Reduced maxfun (50→20)"},
        {"maxfun": 10, "T_opt": 500, "dt_opt": 10, "label": "Minimal maxfun (50→10)"},
        {"maxfun": 50, "T_opt": 300, "dt_opt": 10, "label": "Shorter horizon (500→300s)"},
        {"maxfun": 50, "T_opt": 500, "dt_opt": 20, "label": "Coarser timestep (10→20s)"},
        {"maxfun": 20, "T_opt": 300, "dt_opt": 20, "label": "Combined optimizations"},
    ]

    results = []

    for config in configs:
        print(f"\n[TESTING] {config['label']}")
        print(f"  maxfun={config['maxfun']}, T_opt={config['T_opt']}s, dt_opt={config['dt_opt']}s")

        env = MPCenv(
            turbine=V80(),
            x_pos=x_pos,
            y_pos=y_pos,
            config=make_config(),
            ws_scaling_min=6, ws_scaling_max=15,
            wd_scaling_min=250, wd_scaling_max=290,
            ti_scaling_min=0.01, ti_scaling_max=0.15,
            turbtype="None",
            dt_env=30,
            dt_sim=10,
            yaw_step_sim=10*0.3,
            yaw_init='Zeros',
        )

        obs, info = env.reset()

        # Override MPC parameters by modifying the step function's calls
        # We'll time a single optimization directly
        from mpcrl.mpc import optimize_farm_back2front

        action = env.action_space.sample()
        estimated_wd = (action[0] + 1) / 2 * (290 - 250) + 250
        estimated_ws = (action[1] + 1) / 2 * (15 - 6) + 6
        estimated_TI = (action[2] + 1) / 2 * (0.15 - 0.01) + 0.01

        env.mpc_model.update_conditions(U_inf=estimated_ws, TI=estimated_TI, wd=estimated_wd)
        current_yaws_orig = env.current_yaw.copy()
        current_yaws_sorted = current_yaws_orig[env.mpc_model.sorted_indices]

        t_start = time.time()
        optimized_params = optimize_farm_back2front(
            env.mpc_model, current_yaws_sorted,
            r_gamma=env.yaw_step_sim/env.dt_sim,
            t_AH=100.0,
            dt_opt=config['dt_opt'],
            T_opt=config['T_opt'],
            maxfun=config['maxfun'],
            seed=42,
            initial_params=None
        )
        t_opt = time.time() - t_start

        cache_stats = env.mpc_model.cache.get_stats()

        print(f"  Optimization time: {t_opt:.2f}s")
        print(f"  {cache_stats}")

        speedup = results[0]['time'] / t_opt if results else 1.0

        results.append({
            'label': config['label'],
            'config': config,
            'time': t_opt,
            'speedup': speedup,
            'cache_stats': cache_stats
        })

    print("\n" + "="*70)
    print("OPTIMIZATION PARAMETER COMPARISON")
    print("="*70)
    print(f"{'Configuration':<40} {'Time (s)':<12} {'Speedup':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['label']:<40} {r['time']:>8.2f}    {r['speedup']:>8.2f}x")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    best = min(results[1:], key=lambda x: x['time'])  # Skip baseline
    print(f"\nBest configuration: {best['label']}")
    print(f"  Speedup: {best['speedup']:.2f}x")
    print(f"  Time per step: {best['time']:.2f}s")
    print(f"  Estimated training time (100k steps, 6 envs): {100000*best['time']/(6*3600):.1f} hours")

    return results


def estimate_computational_cost():
    """Analyze the theoretical computational cost."""

    print("\n" + "="*70)
    print("COMPUTATIONAL COST ANALYSIS")
    print("="*70)

    n_turbines = 3
    maxfun = 50
    T_opt = 500
    dt_opt = 10
    n_time_steps = int(T_opt / dt_opt)

    print("\nCurrent configuration:")
    print(f"  Number of turbines: {n_turbines}")
    print(f"  Optimization budget per turbine: {maxfun} function evaluations")
    print(f"  Prediction horizon: {T_opt}s")
    print(f"  Time step: {dt_opt}s")
    print(f"  Number of time steps: {n_time_steps}")

    print("\nPer environment step:")
    total_opt_calls = n_turbines * maxfun
    power_calcs_per_opt = n_time_steps * n_turbines
    total_power_calcs = total_opt_calls * power_calcs_per_opt

    print(f"  Total optimization function calls: {total_opt_calls}")
    print(f"  Power calculations per opt call: {power_calcs_per_opt}")
    print(f"  Maximum power calculations: {total_power_calcs:,}")

    print("\nWith caching (assuming 70% hit rate):")
    pywake_calls = int(total_power_calcs * 0.3)
    print(f"  PyWake simulations: ~{pywake_calls:,}")

    print("\nFor full training (100k steps, 6 envs):")
    total_steps = 100000
    pywake_total = pywake_calls * total_steps / 6
    print(f"  Total PyWake simulations: ~{pywake_total:,.0f}")

    # Estimate with reduced parameters
    print("\n" + "-"*70)
    print("WITH OPTIMIZATIONS (maxfun=20, T_opt=300s, dt_opt=20s):")
    print("-"*70)

    maxfun_new = 20
    T_opt_new = 300
    dt_opt_new = 20
    n_time_steps_new = int(T_opt_new / dt_opt_new)

    total_opt_calls_new = n_turbines * maxfun_new
    power_calcs_per_opt_new = n_time_steps_new * n_turbines
    total_power_calcs_new = total_opt_calls_new * power_calcs_per_opt_new
    pywake_calls_new = int(total_power_calcs_new * 0.3)

    print(f"  Time steps reduced: {n_time_steps} → {n_time_steps_new}")
    print(f"  Opt calls reduced: {total_opt_calls} → {total_opt_calls_new}")
    print(f"  Power calcs reduced: {total_power_calcs:,} → {total_power_calcs_new:,}")
    print(f"  PyWake calls reduced: {pywake_calls:,} → {pywake_calls_new:,}")
    print(f"  Expected speedup: {total_power_calcs / total_power_calcs_new:.2f}x")


if __name__ == "__main__":
    # Run all profiling
    estimate_computational_cost()
    avg_time, hit_rate = profile_single_mpc_step()
    results = profile_optimization_params()

    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)
    print("\nSee results above for detailed analysis and recommendations.")
