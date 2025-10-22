import matplotlib.pyplot as plt
import os
import numpy as np


import xarray as xr
from pathos.pools import ProcessPool
from windgym.WindGym import AgentEvalFast
from windgym.WindGym import FarmEval
from windgym.WindGym.Agents import GreedyAgent, PyWakeAgent

from torch.distributions.normal import Normal

import random
import time
from dataclasses import dataclass

import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from windgym.WindGym import WindFarmEnv
import xarray as xr
import wandb


LOG_STD_MAX = 2
LOG_STD_MIN = -5

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


# NOTE: All evals should run with the power reward env.
# This speeds up the evaluation process.

def test_fun(arg):


    def make_config(reward_type):
        # Base configuration dictionary
        config_dict = {
            "yaw_init": "Random",
            "BaseController": "Local",
            "ActionMethod": "yaw",
            "Track_power": False,
            "farm": {
                "yaw_min": -30,
                "yaw_max": 30,
            },
            "wind": {
                "ws_min": 8,
                "ws_max": 10,
                "TI_min": 0.05,
                "TI_max": 0.05,
                "wd_min": 260,
                "wd_max": 280,
            },
            "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
            "power_def": {"Power_reward": "Power_avg", "Power_avg": 1, "Power_scaling": 1.0},
            "mes_level": {
                "turb_ws": True,
                "turb_wd": True,
                "turb_TI": False,
                "turb_power": False,
                "farm_ws": False,
                "farm_wd": False,
                "farm_TI": False,
                "farm_power": False,
            },
            "ws_mes": {
                "ws_current": False,
                "ws_rolling_mean": True,
                "ws_history_N": 3,
                "ws_history_length": 3,
                "ws_window_length": 1,
            },
            "wd_mes": {
                "wd_current": False,
                "wd_rolling_mean": True,
                "wd_history_N": 3,
                "wd_history_length": 3,
                "wd_window_length": 1,
            },
            "yaw_mes": {
                "yaw_current": False,
                "yaw_rolling_mean": True,
                "yaw_history_N": 3,
                "yaw_history_length": 3,
                "yaw_window_length": 1,
            },
            "power_mes": {
                "power_current": False,
                "power_rolling_mean": False,
                "power_history_N": 1,
                "power_history_length": 1,
                "power_window_length": 1,
            },
        }

        return config_dict




    from windgym.WindGym import AgentEvalFast
    from windgym.WindGym import FarmEval
    from windgym.WindGym.utils.generate_layouts import generate_square_grid, generate_cirular_farm
    # from py_wake.examples.data.dtu10mw import DTU10MW as wind_turbine
    T_SIM = 1000  # Simulation time in seconds, 1 hour

    model_path = arg[0]
    wd = arg[1]
    ws = arg[2]
    ti = arg[3]
    eval_mode = arg[4]
    dt_sim = arg[5]
    dt_env = arg[6]
    yaw_step = arg[7]
    reward_type = arg[8]
    box = arg[9]


    turbbox_path = f"/work/users/manils/rl_timestep/Boxes/torque_longer/{box}"

    model_name = model_path.split('_')[0]
    step = int(model_path.split('_')[1].split('.')[0])

    model_folder = 'runs/' + model_name + '/' + model_path

    number = int(''.join(x for x in model_name if x.isdigit()))

    # box_number = int(''.join(x for x in box if x.isdigit()))

    from py_wake.examples.data.dtu10mw import DTU10MW as wind_turbine
    turbine = wind_turbine()

    if ti < 0.001:
        turbulence_type = "None"
        ti_use = 0.0001
    else:
        turbulence_type = "MannLoad"
        ti_use = ti
    
    test_dict = make_config(reward_type=reward_type)

    x_pos, y_pos = generate_square_grid(turbine=wind_turbine(), 
                                        nx=3, ny=1, 
                                        xDist=5, yDist=5)

    env = FarmEval(turbine=turbine,
                   config=test_dict,
                   turbtype="Random", # Laminar
                   x_pos=x_pos,
                   y_pos=y_pos,
                   TurbBox=turbbox_path,
                   dt_sim=dt_sim,  # Simulation timestep in seconds
                   dt_env=dt_env,  # Environment timestep in seconds
                   yaw_step_sim=yaw_step,
                   )

    device = torch.device("cpu")
    model = Actor(env).to(device)

    if os.path.exists(model_folder):
        model.load_state_dict(torch.load(model_folder, weights_only=True, map_location=torch.device('cpu'))[0]) #For ones without gpu. I think
        # model.load_state_dict(torch.load(model_folder, weights_only=True, map_location=torch.device('cpu'))[0]) #For ones without gpu. I think
        # model.load_state_dict(torch.load(model_folder, weights_only=True)[0])
    else:
        print("Model not found: ", model_folder)
        0/0

    ds = AgentEvalFast(env, model,
                       model_step=step,
                       ws=ws, 
                       ti=ti_use,
                       wd=wd,
                       turbbox=f"Random",  # BARE EN STRING
                       t_sim=T_SIM,
                       deterministic=eval_mode,
                       # debug=True,
                       )  # Default values
    return ds


if __name__ == '__main__':

    model_type = "SAC"
    GLOBAL_YAML_PATH ="./envs/env501_power.yaml"


    wdirs = [265, 270, 275]
    wss = [9]
    TIs = [0.05]
    # BOXES = ["TF_seed_30.nc", "TF_seed_31.nc", "TF_seed_32.nc", "TF_seed_33.nc", "TF_seed_34.nc", "TF_seed_35.nc", "TF_seed_36.nc"]
    # deterministic_modes = [True, False]  # True for deterministic, False for stochastic

    # BOXES = ["TF_seed_30.nc", "TF_seed_31.nc", "TF_seed_32.nc", "TF_seed_33.nc", "TF_seed_34.nc", "TF_seed_35.nc"]
    BOXES = ["Random"]
    deterministic_modes = [False]
    api = wandb.Api()

    entity = "manils-danmarks-tekniske-universitet-dtu" 
    project = "rewardsV3"
    runs = api.runs(entity + "/" + project)

    total = len(runs) 
    i = 0

    runs_integers = list(range(len(runs)))
    random.shuffle(runs_integers)

    # for j, run in enumerate(runs):
    for j in runs_integers:
        run = runs[j]
        # We eval all the runs inside the project.
        name = run.name

        # Check is the run is finished
        if run.state == "finished":
            print("Run is finished.")
        else:
            print("Run is not finished yet.")
            i += 1
            continue

        algo = run.config["algo"]
        if algo != model_type:
            print("Skipping run: ", name, " because the algo is not ", model_type)
            i += 1
            continue


        model_eval = run.name # The model to evaluate

        files = os.listdir(f"./runs/{model_eval}")  # skal være runs snart

        # find the file that ends with .pt
        model_files = []
        for file in files:
            if file.endswith(".pt"):
                model_files.append(file)
                # model_file = file

        if len(model_files) == 0:
            # If there are no model files, skip this number
            print("No model files found for: ", model_eval)
            i += 1
            continue

        # Sort the model files by the number in the filename
        model_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        # Print the sorted model files
        print("Evaluating these::")

        # Skip the second last model.
        model_files_to_eval = [file for i, file in enumerate(model_files) if i != len(model_files) - 2]
        for model_file in model_files_to_eval:
            print(model_file)

        # model_files_to_eval = model_files_to_eval[-1] # Only evaluate the last model file.

        # If model_files_to_eval is a string, convert it to a list.
        if type(model_files_to_eval) is str:
            model_files_to_eval = [model_files_to_eval]

        # Do all the files.
        # model_files_to_eval = model_files


        # Extract things from wandb to pass to the function call
        dt_sim = run.config["dt_sim"]
        dt_env = run.config["dt_env"]
        yaw_step = run.config["yaw_step"]
        reward_type = run.config["reward_type"]

        all_permutations = []
        for model_file in model_files_to_eval:
            for wdir in wdirs:
                for ws in wss:
                    for ti in TIs:
                        for box in BOXES:
                            for eval_mode in deterministic_modes:
                                # Create a permutation of the arguments
                                # and append it to the list.
                                all_permutations.append([model_file, 
                                                        wdir, 
                                                        ws, 
                                                        ti, 
                                                        eval_mode,
                                                        dt_sim,
                                                        dt_env,
                                                        yaw_step,
                                                        reward_type,
                                                        box,
                                                        ])

        print("evaluating: ", model_file, " with ", len(all_permutations), " permutations.")

        # Before we start evaluating, check if the eval file exist:
        eval_file_name = f"evals/{model_file.split('.')[0].split('_')[0]}.nc"
        if os.path.exists(eval_file_name):
            print("Eval file already exists: ", eval_file_name)
            i += 1
            continue

        pool = ProcessPool(nodes=8)

        results = pool.imap(test_fun, all_permutations)
        results = list(results)

        pool.close()
        pool.join()
        pool.clear()

        temp = xr.merge(results)
        temp.to_netcdf( eval_file_name )
            # f"evals/{model_file.split('.')[0].split('_')[0]}.nc")



        temp = []
        del temp

        i += 1
        print("Done with evaluation of: ", run.name, " ", eval_mode, " ", i, " out of: ", total)
        #break
