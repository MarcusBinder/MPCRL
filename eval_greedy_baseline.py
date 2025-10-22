"""
Evaluation script for GreedyAgent (no control baseline).
This evaluates wind farm performance with no wake steering (zero yaw offset).
Represents standard turbine operation without any control strategy.
"""

import os
import numpy as np
import xarray as xr
from pathos.pools import ProcessPool
from dataclasses import dataclass

from windgym.WindGym import FarmEval, AgentEvalFast
from windgym.WindGym.Agents import GreedyAgent
from mpcrl import make_config


def make_config_greedy():
    """
    Create a config for greedy agent evaluation.
    Enable TI measurements for consistency with other evaluations.
    """
    config = make_config()
    # Enable TI measurements
    config["mes_level"]["turb_TI"] = True
    config["mes_level"]["farm_TI"] = True
    return config


def test_fun_greedy(arg):
    """
    Function to evaluate greedy agent on a specific wind condition.
    """
    from windgym.WindGym.utils.generate_layouts import generate_square_grid

    T_SIM = 1000  # Simulation time in seconds

    wd = arg[0]
    ws = arg[1]
    ti = arg[2]
    eval_mode = arg[3]
    dt_sim = arg[4]
    dt_env = arg[5]
    yaw_step = arg[6]
    turbtype = arg[7]
    TI_type = arg[8]
    max_eps = arg[9]
    box = arg[10]

    turbbox_path = f"/work/users/manils/rl_timestep/Boxes/V80env/{box}"

    # Import the correct wind turbine
    if turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as wind_turbine
    elif turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as wind_turbine

    turbine = wind_turbine()

    # Get the farm layout
    x_pos, y_pos = generate_square_grid(turbine=turbine,
                                        nx=3, ny=1,
                                        xDist=5, yDist=5)

    # Create the evaluation environment (FarmEval, not MPCenv)

    if TI_type == "None":
        ti = 0.0  # Override ti to 0 if no TI measurements

    env = FarmEval(
        turbine=turbine,
        n_passthrough=max_eps,
        x_pos=x_pos,
        y_pos=y_pos,
        ws_scaling_min=6, ws_scaling_max=15,
        wd_scaling_min=250, wd_scaling_max=290,
        ti_scaling_min=0.01, ti_scaling_max=0.15,
        TurbBox=turbbox_path,
        config=make_config_greedy(),
        turbtype=TI_type,
        dt_sim=dt_sim,
        dt_env=dt_env,
        yaw_step_sim=yaw_step * dt_sim,
        Baseline_comp=True,  # Enable baseline comparison
    )

    # Create the greedy agent (no control, zero yaw offset)
    model = GreedyAgent()

    # Use windgym's AgentEvalFast for evaluation
    ds = AgentEvalFast(
        env,
        model,
        model_step=0,  # No training steps for greedy agent
        ws=ws,
        ti=ti,
        wd=wd,
        turbbox=box,
        t_sim=T_SIM,
        deterministic=eval_mode,
    )

    return ds


@dataclass
class EvalArgs:
    """Arguments for greedy baseline evaluation"""
    output_name: str = "greedy_baseline"
    """output filename (default: greedy_baseline.nc)"""
    num_workers: int = 4
    """number of parallel workers for evaluation"""
    wdirs: list = None
    """wind directions to evaluate (default: [265, 270, 275])"""
    wss: list = None
    """wind speeds to evaluate (default: [9])"""
    TIs: list = None
    """turbulence intensities to evaluate (default: [0.05])"""
    boxes: list = None
    """turbulence boxes to evaluate (default: ["Random"])"""
    deterministic: bool = False
    """whether to use deterministic policy (not used for greedy)"""
    # Environment parameters (defaults match training)
    dt_sim: int = 10
    dt_env: int = 30
    yaw_step: float = 0.3
    turbtype: str = "DTU10MW"
    TI_type: str = "None"
    max_eps: int = 30


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Greedy Agent (no control baseline)')
    parser.add_argument('--output_name', type=str, default='greedy_baseline',
                        help='Output filename (default: greedy_baseline.nc)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')

    cmd_args = parser.parse_args()

    # Create args with defaults
    args = EvalArgs(
        output_name=cmd_args.output_name,
        num_workers=cmd_args.num_workers
    )

    # Set default evaluation conditions
    wdirs = args.wdirs if args.wdirs is not None else [265, 270, 275]
    wss = args.wss if args.wss is not None else [9]
    TIs = args.TIs if args.TIs is not None else [0.05]
    BOXES = args.boxes if args.boxes is not None else ["Random"]
    deterministic_modes = [args.deterministic]

    print(f"=== Evaluating Greedy Agent (No Control Baseline) ===")
    print(f"Wind directions: {wdirs}")
    print(f"Wind speeds: {wss}")
    print(f"Turbulence intensities: {TIs}")
    print(f"Turbulence boxes: {BOXES}")

    # Create all permutations of evaluation scenarios
    all_permutations = []
    for wdir in wdirs:
        for ws in wss:
            for ti in TIs:
                for box in BOXES:
                    for eval_mode in deterministic_modes:
                        all_permutations.append([
                            wdir,
                            ws,
                            ti,
                            eval_mode,
                            args.dt_sim,
                            args.dt_env,
                            args.yaw_step,
                            args.turbtype,
                            args.TI_type,
                            args.max_eps,
                            box,
                        ])

    print(f"\nTotal evaluation scenarios: {len(all_permutations)}")

    # Check if eval file already exists
    eval_file_name = f"evals/{args.output_name}.nc"
    if os.path.exists(eval_file_name):
        print(f"Warning: Eval file already exists: {eval_file_name}")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Evaluation cancelled.")
            exit(0)

    # Create evals directory if it doesn't exist
    os.makedirs("evals", exist_ok=True)

    # Run evaluation in parallel
    print(f"\nRunning evaluation with {args.num_workers} workers...")
    pool = ProcessPool(nodes=args.num_workers)

    results = pool.imap(test_fun_greedy, all_permutations)
    results = list(results)

    pool.close()
    pool.join()
    pool.clear()

    # Merge results and save
    print("\nMerging results...")
    ds_total = xr.merge(results)
    ds_total.to_netcdf(eval_file_name)

    print(f"\nEvaluation complete! Results saved to: {eval_file_name}")

    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    print(f"Agent type: Greedy (No Control)")
    print(f"Wind directions: {wdirs}")
    print(f"Wind speeds: {wss}")
    print(f"Turbulence intensities: {TIs}")
    print(f"Total scenarios: {len(all_permutations)}")

    # Print quick statistics
    # Note: For greedy agent, pct_inc should be ~0% since baseline=greedy
    if 'powerF_a' in ds_total.data_vars:
        mean_power = ds_total.powerF_a.mean().values
        print(f"\nMean farm power: {mean_power:.2f} W")
    if 'pct_inc' in ds_total.data_vars:
        mean_power_inc = ds_total.pct_inc.mean().values
        print(f"Mean power increase vs baseline: {mean_power_inc:.2f}%")
        print("(Should be ~0% for greedy agent since baseline=greedy)")
