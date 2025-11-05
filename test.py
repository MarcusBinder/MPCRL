import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from itertools import product
from py_wake.examples.data.hornsrev1 import V80
from mpcrl.mpc import *
from windgym.WindGym import PyWakeAgent
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define the farm
x_pos = np.array([0, 500, 900])
y_pos = np.array([0, 0, 0])
D = 80
APPLY_YAW_PENALTY = False

# Wind conditions to test
WIND_CONDITIONS = [
    # (WS, TI, WD)
    (8, 0.06, 270.5),
    (8, 0.06, 275),
    (8, 0.06, 272),
    (10, 0.07, 270.5),
    (10, 0.07, 275),
    (10, 0.07, 272),
    (12, 0.08, 270.5),
    (12, 0.08, 275),
    (12, 0.08, 272),
]

# Evaluation parameters (fixed)
EVAL_T = 500  # Evaluation time horizon
EVAL_DT = 10  # Evaluation time step
R_GAMMA = 0.3
T_AH = 100
SEED = 42

# Parameter search space
PARAM_GRID = {
    'dt_opt': [5, 10, 20],
    't_opt': [300, 600, 1000],
    'maxfun': [50, 100, 250, 500, 1000],
    'use_time_shifted': [False, True]
}

# Multiprocessing configuration
N_WORKERS = mp.cpu_count() - 1  # Leave one CPU free
print(f"Using {N_WORKERS} workers (out of {mp.cpu_count()} CPUs)")

# ============================================================================
# HELPER FUNCTIONS (must be at module level for pickling)
# ============================================================================

def setup_models(ws, ti, wd, x_pos, y_pos, D, apply_yaw_penalty):
    """Initialize MPC model and PyWake reference for given conditions"""
    # MPC Model
    mpc_model = WindFarmModel(
        x_pos, y_pos, D=D, cache_size=64000,
        wt=V80(), cache_quant=0.25, wind_quant=0.25,
        apply_yaw_penalty=apply_yaw_penalty
    )
    mpc_model.update_conditions(U_inf=ws, TI=ti, wd=wd)
    
    # PyWake Reference
    pywake_agent = PyWakeAgent(x_pos=x_pos, y_pos=y_pos, turbine=V80())
    pywake_agent.update_wind(wind_speed=ws, wind_direction=wd, TI=ti)
    pywake_agent.optimize()
    
    return mpc_model, pywake_agent

def calculate_yaw_distance(optimized_params, reference_yaws, model, r_gamma, t_ah, x_pos):
    """Calculate distance between MPC solution and PyWake reference"""
    # Get final yaw angles from MPC trajectory
    _, trajectories, _ = run_farm_delay_loop_optimized(
        model=model,
        yaw_params=optimized_params,
        current_yaw_angles_sorted=np.zeros(len(x_pos)),
        r_gamma=r_gamma,
        t_AH=t_ah,
        dt=10,
        T=t_ah
    )
    
    # Get final yaw angles (sorted order)
    final_yaws_sorted = np.array([traj[-1] for traj in trajectories])
    
    # Convert to original order
    final_yaws_orig = final_yaws_sorted[model.unsorted_indices]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((final_yaws_orig - reference_yaws)**2))
    
    # Also calculate max absolute difference
    max_diff = np.max(np.abs(final_yaws_orig - reference_yaws))
    
    return rmse, max_diff, final_yaws_orig

def evaluate_single_configuration(args):
    """
    Worker function to evaluate a single configuration.
    Must be at module level for pickling.
    
    Args is a tuple of (ws, ti, wd, dt_opt, t_opt, maxfun, use_time_shifted, config_params)
    """
    ws, ti, wd, dt_opt, t_opt, maxfun, use_time_shifted, config_params = args
    
    # Unpack config params
    x_pos = config_params['x_pos']
    y_pos = config_params['y_pos']
    D = config_params['D']
    apply_yaw_penalty = config_params['apply_yaw_penalty']
    r_gamma = config_params['r_gamma']
    t_ah = config_params['t_ah']
    eval_dt = config_params['eval_dt']
    eval_t = config_params['eval_t']
    seed = config_params['seed']
    
    results = {
        'ws': ws,
        'ti': ti,
        'wd': wd,
        'dt_opt': dt_opt,
        't_opt': t_opt,
        'maxfun': maxfun,
        'use_time_shifted': use_time_shifted,
    }
    
    try:
        # Setup models (fresh for each worker)
        mpc_model, pywake_agent = setup_models(ws, ti, wd, x_pos, y_pos, D, apply_yaw_penalty)
        
        # Calculate PyWake reference power
        pywake_reference_power = pywake_agent.calc_power(pywake_agent.optimized_yaws)
        results['pywake_power'] = pywake_reference_power
        
        # Time the optimization
        start_time = time()
        
        optimized_params = optimize_farm_back2front(
            mpc_model,
            current_yaw_angles_sorted=np.zeros(len(x_pos)),
            r_gamma=r_gamma,
            t_AH=t_ah,
            dt_opt=dt_opt,
            T_opt=t_opt,
            maxfun=maxfun,
            seed=seed,
            use_time_shifted=use_time_shifted,
        )
        
        optimization_time = time() - start_time
        results['optimization_time'] = optimization_time
        results['success'] = True
        
        # Evaluate power output
        _, _, P = run_farm_delay_loop_optimized(
            model=mpc_model,
            yaw_params=optimized_params,
            current_yaw_angles_sorted=np.zeros(len(x_pos)),
            r_gamma=r_gamma,
            t_AH=t_ah,
            dt=eval_dt,
            T=eval_t
        )
        
        total_power = P.sum()
        results['total_power'] = total_power
        
        # Calculate power ratio
        pywake_total_energy = pywake_reference_power * eval_t / eval_dt
        results['power_ratio'] = total_power / pywake_total_energy
        
        # Calculate distance to PyWake solution
        rmse, max_diff, final_yaws = calculate_yaw_distance(
            optimized_params, 
            pywake_agent.optimized_yaws,
            mpc_model,
            r_gamma,
            t_ah,
            x_pos
        )
        
        results['yaw_rmse'] = rmse
        results['yaw_max_diff'] = max_diff
        
        # Store final yaw angles
        for i, yaw in enumerate(final_yaws):
            results[f'yaw_{i}'] = yaw
        
        # Store PyWake reference yaws
        for i, yaw in enumerate(pywake_agent.optimized_yaws):
            results[f'pywake_yaw_{i}'] = yaw
            
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        results['optimization_time'] = 0.0
        results['total_power'] = np.nan
        results['power_ratio'] = np.nan
        results['yaw_rmse'] = np.nan
        results['yaw_max_diff'] = np.nan
        results['pywake_power'] = np.nan
    
    return results

# ============================================================================
# MAIN SEARCH
# ============================================================================

def run_parameter_search_parallel():
    """Run the full parameter search in parallel across all conditions"""
    
    print("=" * 80)
    print("MPC PARAMETER SEARCH - MULTI-CONDITION (PARALLEL)")
    print("=" * 80)
    
    # Generate all parameter combinations
    param_combinations = list(product(
        PARAM_GRID['dt_opt'],
        PARAM_GRID['t_opt'],
        PARAM_GRID['maxfun'],
        PARAM_GRID['use_time_shifted']
    ))
    
    # Generate all tasks (condition × parameter combination)
    all_tasks = []
    config_params = {
        'x_pos': x_pos,
        'y_pos': y_pos,
        'D': D,
        'apply_yaw_penalty': APPLY_YAW_PENALTY,
        'r_gamma': R_GAMMA,
        't_ah': T_AH,
        'eval_dt': EVAL_DT,
        'eval_t': EVAL_T,
        'seed': SEED,
    }
    
    for ws, ti, wd in WIND_CONDITIONS:
        for dt_opt, t_opt, maxfun, use_time_shifted in param_combinations:
            task = (ws, ti, wd, dt_opt, t_opt, maxfun, use_time_shifted, config_params)
            all_tasks.append(task)
    
    total_tasks = len(all_tasks)
    print(f"\nTesting {len(param_combinations)} parameter configurations")
    print(f"across {len(WIND_CONDITIONS)} wind conditions")
    print(f"Total evaluations: {total_tasks}")
    print(f"Using {N_WORKERS} parallel workers\n")
    print(f"Parameter space: {PARAM_GRID}")
    print(f"Wind conditions (WS, TI, WD): {WIND_CONDITIONS}\n")
    
    # Run in parallel with progress bar
    print("Starting parallel evaluation...")
    start_time = time()
    
    with mp.Pool(processes=N_WORKERS) as pool:
        # Use imap_unordered for progress tracking
        results_list = list(tqdm(
            pool.imap_unordered(evaluate_single_configuration, all_tasks),
            total=total_tasks,
            desc="Evaluating configurations",
            unit="eval"
        ))
    
    total_time = time() - start_time
    
    print(f"\n✓ Parallel evaluation complete in {total_time:.1f}s")
    print(f"  Average time per evaluation: {total_time/total_tasks:.2f}s")
    print(f"  Speedup vs sequential (estimated): {total_tasks * 10 / total_time:.1f}x")
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Filter successful runs
    df_success = df[df['success']].copy()
    
    print(f"\nResults summary:")
    print(f"  Successful runs: {len(df_success)}/{len(df)}")
    
    if len(df_success) < len(df):
        print(f"  Failed runs: {len(df) - len(df_success)}")
        failed_df = df[~df['success']]
        if 'error' in failed_df.columns:
            print(f"  Error types:")
            for error in failed_df['error'].value_counts().head(3).items():
                print(f"    - {error[0]}: {error[1]} times")
    
    # Aggregate statistics by parameter configuration
    df_agg = df_success.groupby(['dt_opt', 't_opt', 'maxfun', 'use_time_shifted']).agg({
        'power_ratio': ['mean', 'std', 'min', 'max'],
        'yaw_rmse': ['mean', 'std', 'min', 'max'],
        'optimization_time': ['mean', 'std', 'min', 'max'],
        'total_power': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]
    
    if len(df_success) > 0:
        print(f"\nBest average configurations across all conditions:")
        best_power_idx = df_agg['power_ratio_mean'].idxmax()
        best_rmse_idx = df_agg['yaw_rmse_mean'].idxmin()
        best_time_idx = df_agg['optimization_time_mean'].idxmin()
        
        print(f"\n  Highest avg power ratio:")
        print(f"    {df_agg.loc[best_power_idx, ['dt_opt', 't_opt', 'maxfun', 'use_time_shifted', 'power_ratio_mean', 'power_ratio_std']].to_dict()}")
        
        print(f"\n  Lowest avg RMSE:")
        print(f"    {df_agg.loc[best_rmse_idx, ['dt_opt', 't_opt', 'maxfun', 'use_time_shifted', 'yaw_rmse_mean', 'yaw_rmse_std']].to_dict()}")
        
        print(f"\n  Fastest avg time:")
        print(f"    {df_agg.loc[best_time_idx, ['dt_opt', 't_opt', 'maxfun', 'use_time_shifted', 'optimization_time_mean', 'optimization_time_std']].to_dict()}")
    
    return df_success, df_agg

# ============================================================================
# VISUALIZATION (same as before)
# ============================================================================

def plot_results(df, df_agg):
    """Create comprehensive visualization of results"""
    
    if len(df) == 0:
        print("No successful runs to plot!")
        return
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Power Ratio Distribution by Configuration (Box Plot)
    ax1 = plt.subplot(3, 4, 1)
    df_plot = df.copy()
    df_plot['config'] = df_plot.apply(
        lambda x: f"dt={int(x['dt_opt'])}\nT={int(x['t_opt'])}\nmf={int(x['maxfun'])}\n{'TS' if x['use_time_shifted'] else 'ST'}", 
        axis=1
    )
    
    # Select top 10 configs by mean power
    top_configs = df_agg.nlargest(10, 'power_ratio_mean')
    top_config_tuples = list(zip(
        top_configs['dt_opt'], 
        top_configs['t_opt'], 
        top_configs['maxfun'], 
        top_configs['use_time_shifted']
    ))
    
    df_top = df_plot[df_plot.apply(
        lambda x: (x['dt_opt'], x['t_opt'], x['maxfun'], x['use_time_shifted']) in top_config_tuples, 
        axis=1
    )]
    
    if len(df_top) > 0:
        df_top.boxplot(column='power_ratio', by='config', ax=ax1)
        ax1.set_xlabel('Configuration', fontsize=9)
        ax1.set_ylabel('Power Ratio vs PyWake', fontsize=10)
        ax1.set_title('Top 10 Configs: Power Ratio Distribution', fontsize=11, fontweight='bold')
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
        plt.sca(ax1)
        plt.xticks(rotation=45, ha='right')
    
    # 2. RMSE Distribution by Configuration
    ax2 = plt.subplot(3, 4, 2)
    if len(df_top) > 0:
        df_top.boxplot(column='yaw_rmse', by='config', ax=ax2)
        ax2.set_xlabel('Configuration', fontsize=9)
        ax2.set_ylabel('Yaw RMSE (°)', fontsize=10)
        ax2.set_title('Top 10 Configs: RMSE Distribution', fontsize=11, fontweight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
    
    # 3. Average Power vs Time (colored by RMSE)
    ax3 = plt.subplot(3, 4, 3)
    scatter3 = ax3.scatter(df_agg['optimization_time_mean'], df_agg['power_ratio_mean'], 
                          c=df_agg['yaw_rmse_mean'], s=100, alpha=0.6, cmap='RdYlGn_r')
    ax3.errorbar(df_agg['optimization_time_mean'], df_agg['power_ratio_mean'],
                xerr=df_agg['optimization_time_std'], yerr=df_agg['power_ratio_std'],
                fmt='none', alpha=0.3, color='gray')
    ax3.set_xlabel('Avg Optimization Time (s)', fontsize=10)
    ax3.set_ylabel('Avg Power Ratio', fontsize=10)
    ax3.set_title('Power vs Time (Avg Across Conditions)', fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Avg RMSE (°)', fontsize=9)
    
    # 4. Average RMSE vs Time
    ax4 = plt.subplot(3, 4, 4)
    scatter4 = ax4.scatter(df_agg['optimization_time_mean'], df_agg['yaw_rmse_mean'],
                          c=df_agg['power_ratio_mean'], s=100, alpha=0.6, cmap='viridis')
    ax4.errorbar(df_agg['optimization_time_mean'], df_agg['yaw_rmse_mean'],
                xerr=df_agg['optimization_time_std'], yerr=df_agg['yaw_rmse_std'],
                fmt='none', alpha=0.3, color='gray')
    ax4.set_xlabel('Avg Optimization Time (s)', fontsize=10)
    ax4.set_ylabel('Avg Yaw RMSE (°)', fontsize=10)
    ax4.set_title('RMSE vs Time (Avg Across Conditions)', fontsize=11, fontweight='bold')
    ax4.grid(alpha=0.3)
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Avg Power Ratio', fontsize=9)
    
    # 5. Performance by Wind Speed
    ax5 = plt.subplot(3, 4, 5)
    ws_grouped = df.groupby(['ws', 'use_time_shifted']).agg({
        'power_ratio': 'mean',
        'yaw_rmse': 'mean'
    }).reset_index()
    
    for shifted in [False, True]:
        mask = ws_grouped['use_time_shifted'] == shifted
        label = 'Time-Shifted' if shifted else 'Standard'
        marker = 'o' if shifted else 's'
        ax5.plot(ws_grouped[mask]['ws'], ws_grouped[mask]['power_ratio'], 
                marker=marker, label=label, linewidth=2, markersize=8)
    
    ax5.set_xlabel('Wind Speed (m/s)', fontsize=10)
    ax5.set_ylabel('Avg Power Ratio', fontsize=10)
    ax5.set_title('Performance vs Wind Speed', fontsize=11, fontweight='bold')
    ax5.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Performance by Wind Direction
    ax6 = plt.subplot(3, 4, 6)
    wd_grouped = df.groupby(['wd', 'use_time_shifted']).agg({
        'power_ratio': 'mean',
        'yaw_rmse': 'mean'
    }).reset_index()
    
    for shifted in [False, True]:
        mask = wd_grouped['use_time_shifted'] == shifted
        label = 'Time-Shifted' if shifted else 'Standard'
        marker = 'o' if shifted else 's'
        ax6.plot(wd_grouped[mask]['wd'], wd_grouped[mask]['power_ratio'], 
                marker=marker, label=label, linewidth=2, markersize=8)
    
    ax6.set_xlabel('Wind Direction (°)', fontsize=10)
    ax6.set_ylabel('Avg Power Ratio', fontsize=10)
    ax6.set_title('Performance vs Wind Direction', fontsize=11, fontweight='bold')
    ax6.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # 7. Heatmap: Power Ratio by WS and WD (Time-Shifted)
    ax7 = plt.subplot(3, 4, 7)
    df_shifted = df[df['use_time_shifted'] == True]
    pivot_power = df_shifted.pivot_table(values='power_ratio', index='ws', columns='wd', aggfunc='mean')
    im7 = ax7.imshow(pivot_power.values, cmap='RdYlGn', aspect='auto', vmin=0.95, vmax=1.05)
    ax7.set_xticks(range(len(pivot_power.columns)))
    ax7.set_yticks(range(len(pivot_power.index)))
    ax7.set_xticklabels(pivot_power.columns)
    ax7.set_yticklabels(pivot_power.index)
    ax7.set_xlabel('Wind Direction (°)', fontsize=10)
    ax7.set_ylabel('Wind Speed (m/s)', fontsize=10)
    ax7.set_title('Power Ratio Heatmap\n(Time-Shifted)', fontsize=11, fontweight='bold')
    plt.colorbar(im7, ax=ax7, label='Power Ratio')
    
    # 8. Heatmap: Power Ratio by WS and WD (Standard)
    ax8 = plt.subplot(3, 4, 8)
    df_standard = df[df['use_time_shifted'] == False]
    pivot_power_std = df_standard.pivot_table(values='power_ratio', index='ws', columns='wd', aggfunc='mean')
    im8 = ax8.imshow(pivot_power_std.values, cmap='RdYlGn', aspect='auto', vmin=0.95, vmax=1.05)
    ax8.set_xticks(range(len(pivot_power_std.columns)))
    ax8.set_yticks(range(len(pivot_power_std.index)))
    ax8.set_xticklabels(pivot_power_std.columns)
    ax8.set_yticklabels(pivot_power_std.index)
    ax8.set_xlabel('Wind Direction (°)', fontsize=10)
    ax8.set_ylabel('Wind Speed (m/s)', fontsize=10)
    ax8.set_title('Power Ratio Heatmap\n(Standard)', fontsize=11, fontweight='bold')
    plt.colorbar(im8, ax=ax8, label='Power Ratio')
    
    # 9. Effect of dt_opt (averaged across conditions)
    ax9 = plt.subplot(3, 4, 9)
    dt_grouped = df_agg.groupby('dt_opt').agg({
        'power_ratio_mean': 'mean',
        'yaw_rmse_mean': 'mean',
        'optimization_time_mean': 'mean'
    })
    x = np.arange(len(dt_grouped))
    width = 0.25
    ax9.bar(x - width, dt_grouped['power_ratio_mean'], width, label='Power Ratio', alpha=0.7)
    ax9_twin = ax9.twinx()
    ax9_twin.bar(x, dt_grouped['yaw_rmse_mean'], width, label='RMSE', alpha=0.7, color='orange')
    ax9_twin.bar(x + width, dt_grouped['optimization_time_mean'] / 10, width, label='Time/10', alpha=0.7, color='green')
    ax9.set_xlabel('dt_opt (s)', fontsize=10)
    ax9.set_ylabel('Avg Power Ratio', fontsize=10)
    ax9_twin.set_ylabel('Avg RMSE (°) / Time/10 (s)', fontsize=10)
    ax9.set_title('Effect of Time Discretization', fontsize=11, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels(dt_grouped.index)
    ax9.legend(loc='upper left', fontsize=8)
    ax9_twin.legend(loc='upper right', fontsize=8)
    ax9.grid(alpha=0.3, axis='y')
    
    # 10. Effect of t_opt
    ax10 = plt.subplot(3, 4, 10)
    t_grouped = df_agg.groupby('t_opt').agg({
        'power_ratio_mean': 'mean',
        'yaw_rmse_mean': 'mean',
        'optimization_time_mean': 'mean'
    })
    x = np.arange(len(t_grouped))
    ax10.bar(x - width, t_grouped['power_ratio_mean'], width, label='Power Ratio', alpha=0.7)
    ax10_twin = ax10.twinx()
    ax10_twin.bar(x, t_grouped['yaw_rmse_mean'], width, label='RMSE', alpha=0.7, color='orange')
    ax10_twin.bar(x + width, t_grouped['optimization_time_mean'] / 10, width, label='Time/10', alpha=0.7, color='green')
    ax10.set_xlabel('t_opt (s)', fontsize=10)
    ax10.set_ylabel('Avg Power Ratio', fontsize=10)
    ax10_twin.set_ylabel('Avg RMSE (°) / Time/10 (s)', fontsize=10)
    ax10.set_title('Effect of Prediction Horizon', fontsize=11, fontweight='bold')
    ax10.set_xticks(x)
    ax10.set_xticklabels(t_grouped.index)
    ax10.legend(loc='upper left', fontsize=8)
    ax10_twin.legend(loc='upper right', fontsize=8)
    ax10.grid(alpha=0.3, axis='y')
    
    # 11. Effect of maxfun
    ax11 = plt.subplot(3, 4, 11)
    maxfun_grouped = df_agg.groupby('maxfun').agg({
        'power_ratio_mean': 'mean',
        'yaw_rmse_mean': 'mean',
        'optimization_time_mean': 'mean'
    })
    x = np.arange(len(maxfun_grouped))
    ax11.bar(x - width, maxfun_grouped['power_ratio_mean'], width, label='Power Ratio', alpha=0.7)
    ax11_twin = ax11.twinx()
    ax11_twin.bar(x, maxfun_grouped['yaw_rmse_mean'], width, label='RMSE', alpha=0.7, color='orange')
    ax11_twin.bar(x + width, maxfun_grouped['optimization_time_mean'] / 10, width, label='Time/10', alpha=0.7, color='green')
    ax11.set_xlabel('maxfun', fontsize=10)
    ax11.set_ylabel('Avg Power Ratio', fontsize=10)
    ax11_twin.set_ylabel('Avg RMSE (°) / Time/10 (s)', fontsize=10)
    ax11.set_title('Effect of Optimization Budget', fontsize=11, fontweight='bold')
    ax11.set_xticks(x)
    ax11.set_xticklabels(maxfun_grouped.index)
    ax11.legend(loc='upper left', fontsize=8)
    ax11_twin.legend(loc='upper right', fontsize=8)
    ax11.grid(alpha=0.3, axis='y')
    
    # 12. Robustness: Std Dev vs Mean Performance
    ax12 = plt.subplot(3, 4, 12)
    scatter12 = ax12.scatter(df_agg['power_ratio_mean'], df_agg['power_ratio_std'],
                            c=df_agg['optimization_time_mean'], s=100, alpha=0.6, cmap='plasma')
    
    # Highlight time-shifted vs standard
    for shifted in [False, True]:
        mask = df_agg['use_time_shifted'] == shifted
        label = 'Time-Shifted' if shifted else 'Standard'
        marker = 'o' if shifted else 's'
        ax12.scatter([], [], marker=marker, s=100, label=label, color='gray')
    
    ax12.set_xlabel('Mean Power Ratio', fontsize=10)
    ax12.set_ylabel('Std Dev Power Ratio', fontsize=10)
    ax12.set_title('Robustness: Mean vs Variability', fontsize=11, fontweight='bold')
    ax12.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax12.grid(alpha=0.3)
    ax12.legend(fontsize=8)
    cbar12 = plt.colorbar(scatter12, ax=ax12)
    cbar12.set_label('Avg Time (s)', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('mpc_parameter_search_multicondition_parallel.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plots saved to 'mpc_parameter_search_multicondition_parallel.png'")
    plt.show()

def print_recommendations(df, df_agg):
    """Print recommendations based on different use cases"""
    
    if len(df) == 0 or len(df_agg) == 0:
        return
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS (Aggregated Across All Conditions)")
    print("=" * 80)
    
    # 1. Best overall (balanced: high power, low variability, reasonable speed)
    df_agg['robustness_score'] = df_agg['power_ratio_mean'] - 0.5 * df_agg['power_ratio_std']
    df_agg['efficiency_score'] = df_agg['robustness_score'] / (df_agg['optimization_time_mean'] / 10)
    best_overall = df_agg.loc[df_agg['efficiency_score'].idxmax()]
    
    print("\n1. 🏆 BEST OVERALL (Robust, Efficient):")
    print(f"   dt_opt={best_overall['dt_opt']}, t_opt={best_overall['t_opt']}, "
          f"maxfun={best_overall['maxfun']}, time_shifted={best_overall['use_time_shifted']}")
    print(f"   Avg Power Ratio: {best_overall['power_ratio_mean']:.4f} (±{best_overall['power_ratio_std']:.4f})")
    print(f"   Avg RMSE: {best_overall['yaw_rmse_mean']:.2f}° (±{best_overall['yaw_rmse_std']:.2f}°)")
    print(f"   Avg Time: {best_overall['optimization_time_mean']:.2f}s (±{best_overall['optimization_time_std']:.2f}s)")
    
    # 2. Maximum average power
    best_power = df_agg.loc[df_agg['power_ratio_mean'].idxmax()]
    print("\n2. ⚡ MAXIMUM AVERAGE POWER:")
    print(f"   dt_opt={best_power['dt_opt']}, t_opt={best_power['t_opt']}, "
          f"maxfun={best_power['maxfun']}, time_shifted={best_power['use_time_shifted']}")
    print(f"   Avg Power Ratio: {best_power['power_ratio_mean']:.4f} (±{best_power['power_ratio_std']:.4f})")
    print(f"   Avg RMSE: {best_power['yaw_rmse_mean']:.2f}° (±{best_power['yaw_rmse_std']:.2f}°)")
    print(f"   Avg Time: {best_power['optimization_time_mean']:.2f}s (±{best_power['optimization_time_std']:.2f}s)")
    
    # 3. Most robust (lowest variability with good performance)
    df_agg_good = df_agg[df_agg['power_ratio_mean'] > 0.99]
    if len(df_agg_good) > 0:
        most_robust = df_agg_good.loc[df_agg_good['power_ratio_std'].idxmin()]
        print("\n3. 🛡️ MOST ROBUST (Low Variability):")
        print(f"   dt_opt={most_robust['dt_opt']}, t_opt={most_robust['t_opt']}, "
              f"maxfun={most_robust['maxfun']}, time_shifted={most_robust['use_time_shifted']}")
        print(f"   Avg Power Ratio: {most_robust['power_ratio_mean']:.4f} (±{most_robust['power_ratio_std']:.4f})")
        print(f"   Avg RMSE: {most_robust['yaw_rmse_mean']:.2f}° (±{most_robust['yaw_rmse_std']:.2f}°)")
        print(f"   Avg Time: {most_robust['optimization_time_mean']:.2f}s (±{most_robust['optimization_time_std']:.2f}s)")
    
    # 4. Best accuracy
    best_accuracy = df_agg.loc[df_agg['yaw_rmse_mean'].idxmin()]
    print("\n4. 🎯 BEST AVERAGE ACCURACY:")
    print(f"   dt_opt={best_accuracy['dt_opt']}, t_opt={best_accuracy['t_opt']}, "
          f"maxfun={best_accuracy['maxfun']}, time_shifted={best_accuracy['use_time_shifted']}")
    print(f"   Avg Power Ratio: {best_accuracy['power_ratio_mean']:.4f} (±{best_accuracy['power_ratio_std']:.4f})")
    print(f"   Avg RMSE: {best_accuracy['yaw_rmse_mean']:.2f}° (±{best_accuracy['yaw_rmse_std']:.2f}°)")
    print(f"   Avg Time: {best_accuracy['optimization_time_mean']:.2f}s (±{best_accuracy['optimization_time_std']:.2f}s)")
    
    # 5. Fastest
    fastest = df_agg.loc[df_agg['optimization_time_mean'].idxmin()]
    print("\n5. ⚡ FASTEST:")
    print(f"   dt_opt={fastest['dt_opt']}, t_opt={fastest['t_opt']}, "
          f"maxfun={fastest['maxfun']}, time_shifted={fastest['use_time_shifted']}")
    print(f"   Avg Power Ratio: {fastest['power_ratio_mean']:.4f} (±{fastest['power_ratio_std']:.4f})")
    print(f"   Avg RMSE: {fastest['yaw_rmse_mean']:.2f}° (±{fastest['yaw_rmse_std']:.2f}°)")
    print(f"   Avg Time: {fastest['optimization_time_mean']:.2f}s (±{fastest['optimization_time_std']:.2f}s)")
    
    # 6. Time-shifted vs Standard comparison
    print("\n6. 📊 TIME-SHIFTED vs STANDARD (Across All Conditions):")
    for shifted in [False, True]:
        subset = df_agg[df_agg['use_time_shifted'] == shifted]
        name = "Time-Shifted" if shifted else "Standard"
        print(f"\n   {name}:")
        print(f"     Avg Power Ratio: {subset['power_ratio_mean'].mean():.4f} "
              f"(std across configs: {subset['power_ratio_mean'].std():.4f})")
        print(f"     Avg RMSE: {subset['yaw_rmse_mean'].mean():.2f}° "
              f"(std across configs: {subset['yaw_rmse_mean'].std():.2f}°)")
        print(f"     Avg Time: {subset['optimization_time_mean'].mean():.2f}s "
              f"(std across configs: {subset['optimization_time_mean'].std():.2f}s)")
        print(f"     Avg Robustness (power std): {subset['power_ratio_std'].mean():.4f}")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Run parallel search
    df_results, df_aggregated = run_parameter_search_parallel()
    
    # Save results
    df_results.to_csv('mpc_parameter_search_parallel_detailed.csv', index=False)
    df_aggregated.to_csv('mpc_parameter_search_parallel_summary.csv', index=False)
    print("\n💾 Results saved to:")
    print("   - mpc_parameter_search_parallel_detailed.csv (all individual runs)")
    print("   - mpc_parameter_search_parallel_summary.csv (aggregated statistics)")
    
    # Visualize
    plot_results(df_results, df_aggregated)
    
    # Print recommendations
    print_recommendations(df_results, df_aggregated)
    
    print("\n" + "=" * 80)
    print("PARALLEL SEARCH COMPLETE")
    print("=" * 80)