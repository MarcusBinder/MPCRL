"""
Evaluation script for trained MPC+RL agents.
This script evaluates SAC agents trained with MPCenv on specific wind conditions.
"""

import os
import numpy as np
import xarray as xr
from pathos.pools import ProcessPool
from dataclasses import dataclass

import torch
import torch.nn as nn

from mpcrl import MPCenvEval, make_config
import wandb


LOG_STD_MAX = 2
LOG_STD_MIN = -5


def eval_single_fast_mpc(
    env,
    model,
    model_step=1,
    ws=10.0,
    ti=0.05,
    wd=270,
    turbbox="Default",
    t_sim=1000,
    deterministic=False,
):
    """
    Custom evaluation function for MPC environments that captures estimated wind conditions.
    Based on windgym's eval_single_fast but modified to track RL agent's wind estimates.
    """
    device = torch.device("cpu")

    env.set_wind_vals(ws=ws, ti=ti, wd=wd)
    baseline_comp = env.Baseline_comp

    # Calculate the correct number of steps
    step_val = env.sim_steps_per_env_step
    total_steps = t_sim // env.dt_env + 1
    time = total_steps * step_val + 1

    n_turb = env.n_turb

    # Initialize the arrays to store the results
    powerF_a = np.zeros((time), dtype=np.float32)
    powerT_a = np.zeros((time, n_turb), dtype=np.float32)
    yaw_a = np.zeros((time, n_turb), dtype=np.float32)
    ws_a = np.zeros((time, n_turb), dtype=np.float32)
    time_plot = np.zeros((time), dtype=int)
    rew_plot = np.zeros((time), dtype=np.float32)

    # NEW: Arrays for estimated wind conditions from RL agent
    estimated_wd_plot = np.zeros((time), dtype=np.float32)
    estimated_ws_plot = np.zeros((time), dtype=np.float32)
    estimated_ti_plot = np.zeros((time), dtype=np.float32)

    if baseline_comp:
        powerF_b = np.zeros((time), dtype=np.float32)
        powerT_b = np.zeros((time, n_turb), dtype=np.float32)
        yaw_b = np.zeros((time, n_turb), dtype=np.float32)
        ws_b = np.zeros((time, n_turb), dtype=np.float32)
        pct_inc = np.zeros((time), dtype=np.float32)

    # Initialize the environment
    obs, info = env.reset()

    # Initialize action variable for later cleanup
    action = None

    # Put the initial values in the arrays
    powerF_a[0] = env.fs.windTurbines.power().sum()
    powerT_a[0] = env.fs.windTurbines.power()
    yaw_a[0] = env.fs.windTurbines.yaw
    ws_a[0] = np.linalg.norm(env.fs.windTurbines.rotor_avg_windspeed, axis=1)
    time_plot[0] = env.fs.time
    rew_plot[0] = 0.0

    # Initial estimated values (no estimate yet at t=0)
    estimated_wd_plot[0] = wd
    estimated_ws_plot[0] = ws
    estimated_ti_plot[0] = ti

    if baseline_comp:
        powerF_b[0] = env.fs_baseline.windTurbines.power().sum()
        powerT_b[0] = env.fs_baseline.windTurbines.power()
        yaw_b[0] = env.fs_baseline.windTurbines.yaw
        ws_b[0] = np.linalg.norm(env.fs_baseline.windTurbines.rotor_avg_windspeed, axis=1)
        pct_inc[0] = ((powerF_a[0] - powerF_b[0]) / powerF_b[0]) * 100

    # Run the simulation
    for i in range(0, total_steps):
        if hasattr(model, "model_type"):
            if model.model_type == "CleanRL":
                obs = np.expand_dims(obs, 0)
                action, _, _ = model.get_action(
                    torch.Tensor(obs).to(device), deterministic=deterministic
                )
                action = action.detach().cpu().numpy()
                action = action.flatten()
            else:
                # This is for baseline agents (Oracle, FrontTurbine, SimpleEstimator)
                action, _, _ = model.get_action(obs, deterministic=deterministic)
                action = action.detach().cpu().numpy()
                action = action.flatten()
        else:
            # This is for other models (Pywake and such)
            action = model.predict(obs, deterministic=deterministic)[0]

        obs, reward, terminated, truncated, info = env.step(action)

        # Put the values in the arrays
        powerF_a[i * step_val + 1 : i * step_val + step_val + 1] = info["powers"].sum(axis=1)
        powerT_a[i * step_val + 1 : i * step_val + step_val + 1] = info["powers"]
        yaw_a[i * step_val + 1 : i * step_val + step_val + 1] = info["yaws"]
        ws_a[i * step_val + 1 : i * step_val + step_val + 1] = info["windspeeds"]
        time_plot[i * step_val + 1 : i * step_val + step_val + 1] = info["time_array"]
        rew_plot[i * step_val + 1 : i * step_val + step_val + 1] = reward

        # NEW: Capture estimated wind conditions from the info dict
        # These are the values the RL agent predicted and fed to the MPC controller
        if 'estimated_wd' in info:
            estimated_wd_plot[i * step_val + 1 : i * step_val + step_val + 1] = info['estimated_wd']
        if 'estimated_ws' in info:
            estimated_ws_plot[i * step_val + 1 : i * step_val + step_val + 1] = info['estimated_ws']
        if 'estimated_TI' in info:
            estimated_ti_plot[i * step_val + 1 : i * step_val + step_val + 1] = info['estimated_TI']

        if baseline_comp:
            powerF_b[i * step_val + 1 : i * step_val + step_val + 1] = info["baseline_powers"].sum(axis=1)
            powerT_b[i * step_val + 1 : i * step_val + step_val + 1] = info["baseline_powers"]
            yaw_b[i * step_val + 1 : i * step_val + step_val + 1] = info["yaws_baseline"]
            ws_b[i * step_val + 1 : i * step_val + step_val + 1] = info["windspeeds_baseline"]
            pct_inc[i * step_val + 1 : i * step_val + step_val + 1] = (
                (info["powers"].sum(axis=1) - info["baseline_powers"].sum(axis=1))
                / info["baseline_powers"].sum(axis=1)
            ) * 100

    # Reshape the arrays
    n_ws = 1
    n_wd = 1
    n_turbbox = 1
    n_TI = 1

    powerF_a = powerF_a.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
    powerT_a = powerT_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
    yaw_a = yaw_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
    ws_a = ws_a.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
    rew_plot = rew_plot.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)

    # NEW: Reshape estimated values
    estimated_wd_plot = estimated_wd_plot.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
    estimated_ws_plot = estimated_ws_plot.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
    estimated_ti_plot = estimated_ti_plot.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)

    # Create xarray dataset
    data_vars = {
        "powerF_a": (
            ("time", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
            powerF_a,
        ),
        "powerT_a": (
            ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
            powerT_a,
        ),
        "yaw_a": (
            ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
            yaw_a,
        ),
        "ws_a": (
            ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
            ws_a,
        ),
        "reward": (
            ("time", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
            rew_plot,
        ),
        # NEW: Estimated wind conditions from RL agent
        "estimated_wd": (
            ("time", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
            estimated_wd_plot,
        ),
        "estimated_ws": (
            ("time", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
            estimated_ws_plot,
        ),
        "estimated_ti": (
            ("time", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
            estimated_ti_plot,
        ),
    }

    # Add baseline variables if applicable
    if baseline_comp:
        powerF_b = powerF_b.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        powerT_b = powerT_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        yaw_b = yaw_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        ws_b = ws_b.reshape(time, n_turb, n_ws, n_wd, n_TI, n_turbbox, 1, 1)
        pct_inc = pct_inc.reshape(time, n_ws, n_wd, n_TI, n_turbbox, 1, 1)

        data_vars.update(
            {
                "powerF_b": (
                    ("time", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
                    powerF_b,
                ),
                "powerT_b": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
                    powerT_b,
                ),
                "yaw_b": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
                    yaw_b,
                ),
                "ws_b": (
                    ("time", "turb", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
                    ws_b,
                ),
                "pct_inc": (
                    ("time", "ws", "wd", "TI", "turbbox", "model_step", "deterministic"),
                    pct_inc,
                ),
            }
        )

    # Common coordinates
    coords = {
        "ws": np.array([ws]),
        "wd": np.array([wd]),
        "turb": np.arange(n_turb),
        "time": time_plot,
        "TI": np.array([ti]),
        "turbbox": [turbbox],
        "model_step": np.array([model_step]),
        "deterministic": np.array([deterministic]),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Cleanup
    env.close()

    return ds


class Actor(nn.Module):
    def __init__(self, env, hidden_sizes=[256, 256]):
        super().__init__()

        self.model_type = "CleanRL"

        obs_dim = int(np.array(env.observation_space.shape).prod())
        act_dim = int(np.prod(env.action_space.shape))

        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        last_dim = prev if hidden_sizes else obs_dim

        self.fc_mean = nn.Linear(last_dim, act_dim)
        self.fc_logstd = nn.Linear(last_dim, act_dim)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x, deterministic=False):
        mean, log_std = self(x)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mean) * self.action_scale + self.action_bias
            log_prob = None
            return action, log_prob, mean

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def test_fun(arg):
    """
    Function to evaluate a single model on a specific wind condition.
    This function is designed to be parallelized using pathos.
    """
    from windgym.WindGym.utils.generate_layouts import generate_square_grid

    T_SIM = 1000  # Simulation time in seconds

    model_path = arg[0]
    wd = arg[1]
    ws = arg[2]
    ti = arg[3]
    eval_mode = arg[4]
    dt_sim = arg[5]
    dt_env = arg[6]
    yaw_step = arg[7]
    turbtype = arg[8]
    TI_type = arg[9]
    max_eps = arg[10]
    net_complexity = arg[11]
    box = arg[12]

    turbbox_path = f"/work/users/manils/rl_timestep/Boxes/V80env/{box}"

    model_name = model_path.split('_')[0]
    step = int(model_path.split('_')[1].split('.')[0])

    model_folder = 'runs/' + model_name + '/' + model_path

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

    # Create the evaluation environment
    env = MPCenvEval(
        turbine=turbine,
        n_passthrough=max_eps,
        x_pos=x_pos,
        y_pos=y_pos,
        ws_scaling_min=6, ws_scaling_max=15,  # wind speed scaling
        wd_scaling_min=250, wd_scaling_max=290,  # wind direction scaling
        ti_scaling_min=0.01, ti_scaling_max=0.15,  # turbulence intensity scaling
        TurbBox=turbbox_path,
        config=make_config(),
        turbtype=TI_type,  # the type of turbulence
        dt_sim=dt_sim,
        dt_env=dt_env,
        yaw_step_sim=yaw_step * dt_sim,
    )

    device = torch.device("cpu")

    # Get the network architecture
    architectures = {
        "small": [128, 128],
        "medium": [256, 256],
        "default": [256, 256],
        "large": [512, 256, 128],
        "extra_large": [1024, 512, 256, 128],
        "wide": [512, 512, 512],
        "deep": [256, 256, 256, 256]
    }

    net_arch = architectures[net_complexity]

    model = Actor(env, hidden_sizes=net_arch).to(device)

    if os.path.exists(model_folder):
        model.load_state_dict(torch.load(model_folder, weights_only=True, map_location=torch.device('cpu'))[0])
    else:
        print("Model not found: ", model_folder)
        raise FileNotFoundError(f"Model not found: {model_folder}")

    # Use our custom MPC evaluation function
    ds = eval_single_fast_mpc(
        env,
        model,
        model_step=step,
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
    """Arguments for evaluation"""
    model_folder: str = "testrun7"
    """the folder containing the trained model"""
    wandb_project: str = "MPC_RL"
    """the wandb project name to fetch run config from"""
    wandb_entity: str = None
    """the wandb entity name"""
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
    """whether to use deterministic policy"""
    eval_all_checkpoints: bool = True
    """whether to evaluate all checkpoints or just the final one"""


if __name__ == '__main__':

    # Default evaluation conditions
    args = EvalArgs()

    # Set default values if not provided
    wdirs = args.wdirs if args.wdirs is not None else [265, 270, 275]
    wss = args.wss if args.wss is not None else [9]
    TIs = args.TIs if args.TIs is not None else [0.05]
    BOXES = args.boxes if args.boxes is not None else ["Random"]
    deterministic_modes = [args.deterministic]

    # Check if the model folder exists
    model_folder = f"runs/{args.model_folder}"
    if not os.path.exists(model_folder):
        print(f"Model folder not found: {model_folder}")
        print("Please check the model folder path and try again.")
        exit(1)

    # Try to get config from wandb if available
    use_wandb = args.wandb_project is not None

    if use_wandb:
        api = wandb.Api()
        entity = args.wandb_entity if args.wandb_entity else "your-entity"
        project = args.wandb_project

        try:
            runs = api.runs(entity + "/" + project)

            # Find the run matching our model folder
            run = None
            for r in runs:
                if r.name == args.model_folder:
                    run = r
                    break

            if run is None:
                print(f"Warning: Could not find wandb run for {args.model_folder}")
                print("Using default config values")
                use_wandb = False
            else:
                # Extract config from wandb
                dt_sim = run.config.get("dt_sim", 10)
                dt_env = run.config.get("dt_env", 30)
                yaw_step = run.config.get("yaw_step", 0.3)
                turbtype = run.config.get("turbtype", "DTU10MW")
                TI_type = run.config.get("TI_type", "None")
                max_eps = run.config.get("max_eps", 30)
                net_complexity = run.config.get("NetComplexity", "default")

                print(f"Loaded config from wandb for run: {run.name}")
        except Exception as e:
            print(f"Error accessing wandb: {e}")
            print("Using default config values")
            use_wandb = False

    # Use default values if wandb is not available
    if not use_wandb:
        dt_sim = 10
        dt_env = 30
        yaw_step = 0.3
        turbtype = "DTU10MW"
        TI_type = "None"
        max_eps = 30
        net_complexity = "default"

    # Find all model files in the folder
    files = os.listdir(model_folder)
    model_files = [f for f in files if f.endswith(".pt")]

    if len(model_files) == 0:
        print(f"No model files found in: {model_folder}")
        exit(1)

    # Sort model files by step number
    model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    print("Found model files:")
    for f in model_files:
        print(f"  {f}")

    # Select which models to evaluate
    if args.eval_all_checkpoints:
        # Skip the second-to-last model (as in the original script)
        model_files_to_eval = [f for i, f in enumerate(model_files) if i != len(model_files) - 2]
    else:
        # Only evaluate the last checkpoint
        model_files_to_eval = [model_files[-1]]

    print("\nEvaluating these checkpoints:")
    for f in model_files_to_eval:
        print(f"  {f}")

    # Create all permutations of evaluation scenarios
    all_permutations = []
    for model_file in model_files_to_eval:
        for wdir in wdirs:
            for ws in wss:
                for ti in TIs:
                    for box in BOXES:
                        for eval_mode in deterministic_modes:
                            all_permutations.append([
                                model_file,
                                wdir,
                                ws,
                                ti,
                                eval_mode,
                                dt_sim,
                                dt_env,
                                yaw_step,
                                turbtype,
                                TI_type,
                                max_eps,
                                net_complexity,
                                box,
                            ])

    print(f"\nTotal evaluation scenarios: {len(all_permutations)}")

    # Check if eval file already exists
    eval_file_name = f"evals/{args.model_folder}.nc"
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

    results = pool.imap(test_fun, all_permutations)
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
    print(f"Model: {args.model_folder}")
    print(f"Wind directions: {wdirs}")
    print(f"Wind speeds: {wss}")
    print(f"Turbulence intensities: {TIs}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Number of checkpoints evaluated: {len(model_files_to_eval)}")
    print(f"Total scenarios: {len(all_permutations)}")
