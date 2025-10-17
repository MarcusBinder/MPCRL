"""
This is just the sac training, but I modified it, such that I could run with the optimal hyperparams from optuna.
It is simply a small modification, but I didnt wanna fuck something up.
"""

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import copy
from mpcrl import MPCenv, make_config
from typing import Literal  # <-- add

# from .same_model_vector_env import SameModelSyncVectorEnv
# from windgym.WindGym import WindFarmEnv, FarmEval
from windgym.WindGym.wrappers import RecordEpisodeVals
from windgym.WindGym.utils.generate_layouts import generate_square_grid, generate_cirular_farm
import pickle
from collections import deque

@dataclass
class Args:
    max_eps: int = 30
    """the maximum number inflow passes"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    save_interval: int = 10_000
    """the interval to save the model NOT USED ATM"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "MPC_RL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: int = 1 #Integer instead of bool. Should (hopefully) work.
    """whether to save model into the `runs/{run_name}` folder"""
    turbtype: str = "DTU10MW"
    """the type of the wind turbine"""
    TI_type: str = "None"
    """the type of the turbulence model"""
    dt_sim: int = 10
    """the time step of the simulation"""
    dt_env: int = 30
    """the time step of the environment"""
    yaw_step: float = 0.3
    """the yaw step of the environment"""
    save_init: int = 1
    """if toggled, the initial model will be saved to the `runs/{run_name}` folder"""

    # Algorithm specific arguments

    total_timesteps: int = 100_000
    """total timesteps of the experiments"""
    num_envs: int = 6
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2 #Maybe 1?
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    NetComplexity: Literal["default", "large", "small", "medium", "extra_large", "wide", "deep"] = "default"  # <-- add
    """network size preset"""
    


def make_env(seed):

    # test_dict["power_mes"]["power_history_length"] = args.power_hist_len
    x_pos, y_pos = generate_square_grid(turbine=wind_turbine(),
                                        nx=3, ny=1,
                                        xDist=5, yDist=5)

    env = MPCenv(turbine=wind_turbine(),
                      n_passthrough=args.max_eps,
                      x_pos=x_pos,
                      y_pos=y_pos,
                      ws_scaling_min=6, ws_scaling_max=15,  # wind speed scaling
                      wd_scaling_min=250, wd_scaling_max=290,  # wind direction scaling
                      ti_scaling_min=0.01, ti_scaling_max=0.15,  # turbulence intensity scaling
                      TurbBox="/work/users/manils/rl_timestep/Boxes/V80env/",
                      config=make_config(),
                      turbtype=args.TI_type,  # the type of turbulence.
                      dt_sim=args.dt_sim,
                      dt_env=args.dt_env,
                      yaw_step_sim=args.yaw_step*args.dt_sim,
                      )

    env.action_space.seed(seed)
    return env


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_sizes):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        in_dim = obs_dim + act_dim

        layers = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.q_head = nn.Linear(prev, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.backbone(x)
        x = self.q_head(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, hidden_sizes):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))

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
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
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

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def save_model(actor, qf1, qf2, step_number):
    """
    This function is used to save the model.
    """
    model_path = f"runs/{run_name}/{args.exp_name}_{step_number}.pt"
    torch.save((actor.state_dict(), qf1.state_dict(),
            qf2.state_dict()), model_path)
    print(f"model saved to {model_path}")




if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.exp_name}"
    args.algo = 'SAC'

    import wandb

    # args.resume = False
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        # import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as wind_turbine
    elif args.turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as wind_turbine
    # print("Using wind turbine type:", args.turbtype)
    envs = gym.vector.AsyncVectorEnv(
        [lambda: make_env(args.seed + i) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
    # print("The envs is constructed")
    envs = RecordEpisodeVals(envs)

    assert isinstance(envs.single_action_space,
                      gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    architectures = {
        "small": [128, 128],
        "medium": [256, 256],
        "default": [256, 256],
        "large": [512, 256, 128],
        "extra_large": [1024, 512, 256, 128],
        "wide": [512, 512, 512],
        "deep": [256, 256, 256, 256]
    }
    net_arch = architectures[args.NetComplexity]


    actor = Actor(envs, hidden_sizes=net_arch).to(device)
    qf1 = SoftQNetwork(envs, hidden_sizes=net_arch).to(device)
    qf2 = SoftQNetwork(envs, hidden_sizes=net_arch).to(device)

    if args.save_init:
        save_model(actor, qf1, qf2, 0)  # Save the initial model if requested


    # Setup the target networks as normal
    qf1_target = SoftQNetwork(envs, hidden_sizes=net_arch).to(device)
    qf2_target = SoftQNetwork(envs, hidden_sizes=net_arch).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) +
                             list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = - \
            torch.prod(torch.Tensor(
                envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    global_step = 0

    num_steps = args.total_timesteps // args.num_envs

    print("Running for num_steps", num_steps)
    print("Total timesteps", args.total_timesteps)
    print("Num envs", args.num_envs)

    save_interval = args.save_interval # How often to save the model
    next_save = save_interval

    # Deque to track last 1000 step rewards (across all envs)
    step_reward_window = deque(maxlen=1000)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    # for global_step in range(args.total_timesteps):
    for update in range(num_steps + 2):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample()
                               for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        global_step += args.num_envs

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions)

        # Track per-step rewards across envs
        step_reward_window.extend(np.array(rewards).flatten().tolist())

        if "final_info" in infos:
            print(
                f"global_step={global_step}, episodic_return={np.mean(envs.return_queue)}")
            writer.add_scalar("charts/episodic_return",
                              np.mean(envs.return_queue), global_step)
            writer.add_scalar("charts/episodic_length",
                              np.mean(envs.length_queue), global_step)
            writer.add_scalar("charts/episodic_power",
                              np.mean(envs.mean_power_queue), global_step)
            writer.add_scalar("charts/episodic_power_nowake",
                              np.mean(envs.mean_power_queue_nowake), global_step)
            writer.add_scalar("charts/episodic_yaw",
                              np.mean(envs.total_yaw_travel_queue), global_step)
            if hasattr(envs, 'mean_power_queue_baseline'):
                writer.add_scalar("charts/episodic_power_baseline",
                                np.mean(envs.mean_power_queue_baseline), global_step)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs


        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations)
                qf1_next_target = qf1_target(
                    data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(
                    data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(
                    qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * \
                    args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if update % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() *
                                      (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data)

            if update % 5 == 0:
                writer.add_scalar("losses/qf1_values",
                                  qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values",
                                  qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss",
                                  qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss",
                                  qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss",
                                  qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss",
                                  actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                
                # Mean of last 1000 step rewards (across all envs)
                writer.add_scalar("charts/step_reward_mean_1000",
                                  float(np.mean(step_reward_window)) if len(step_reward_window) > 0 else 0.0,
                                  global_step)
                
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss",
                                      alpha_loss.item(), global_step)

        if global_step >= next_save:
            if args.save_model:
                # Save the model
                save_model(actor, qf1, qf2, global_step)
            next_save += save_interval

    print("Training finished.")
    if args.save_model:
        save_model(actor, qf1, qf2, global_step)

    envs.close()
    writer.close()