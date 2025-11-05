"""
Fast variant of MPCenv with optimized parameters for training speed.

Key optimizations:
1. Reduced maxfun: 50 → 15 (3.3x fewer optimization calls)
2. Shorter horizon: T_opt 500s → 300s (1.67x fewer timesteps)
3. Coarser timestep: dt_opt 10s → 20s (2x fewer timesteps)

Combined speedup: ~11x theoretical, ~6-8x practical (with caching)

This should reduce training time from ~10+ hours to ~1.5-2 hours for 100k steps.
"""

from WindGym import WindFarmEnv
from .config import make_config
from typing import Any, Dict, Optional, Union
import gymnasium as gym
from .mpc import WindFarmModel, optimize_farm_back2front, run_farm_delay_loop_optimized
import numpy as np


class MPCenvFast(WindFarmEnv):
    """
    Fast variant of Wind Farm Environment with MPC controller.
    Optimized for training speed while maintaining solution quality.
    """

    def __init__(self,
                 mpc_maxfun: int = 15,           # Reduced from 50
                 mpc_T_opt: float = 300.0,       # Reduced from 500
                 mpc_dt_opt: float = 20.0,       # Increased from 10
                 mpc_t_AH: float = 100.0,        # Action horizon (unchanged)
                 mpc_cache_size: int = 64000,    # Cache size
                 mpc_cache_quant: float = 0.25,  # Cache quantization
                 **kwargs):
        super().__init__(**kwargs)
        self.dt_mpc = self.dt_env  # The MPC time step is the same as the environment time step

        # MPC optimization parameters (exposed for tuning)
        self.mpc_maxfun = mpc_maxfun
        self.mpc_T_opt = mpc_T_opt
        self.mpc_dt_opt = mpc_dt_opt
        self.mpc_t_AH = mpc_t_AH
        self.mpc_cache_size = mpc_cache_size
        self.mpc_cache_quant = mpc_cache_quant

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        obs, info = super().reset(seed=seed, options=options)

        self.USE_VARIABLE_HORIZON = True   # Use variable prediction horizon (paper's approach)
        self.USE_TIME_SHIFTED = True       # Use time-shifted cost function (paper's best)
        self.APPLY_YAW_PENALTY = True      # Use Equation 5 penalty for large yaw angles

        self.mpc_model = WindFarmModel(self.x_pos, self.y_pos,
                                       wt = self.turbine,
                                       D=self.D,
                                       cache_size=self.mpc_cache_size,
                                       cache_quant=self.mpc_cache_quant,
                                       wind_quant=0.25,
                                       apply_yaw_penalty=self.APPLY_YAW_PENALTY)
        self.previous_opt_params = None
        return obs, info

    def step(self, action):

        # Step 1: update the MPC model with the current state
        estimated_wd = action[0]
        estimated_ws = action[1]
        estimated_TI = action[2]
        # TODO: Should we use _scaling_ or _inflow_, or maybe something else?
        # The estimates are all from -1 to 1, we need to convert them back to the original range
        estimated_wd = (estimated_wd + 1) / 2 * (self.wd_scaling_max - self.wd_scaling_min) + self.wd_scaling_min
        estimated_ws = (estimated_ws + 1) / 2 * (self.ws_scaling_max - self.ws_scaling_min) + self.ws_scaling_min
        estimated_TI = (estimated_TI + 1) / 2 * (self.ti_scaling_max - self.ti_scaling_min) + self.ti_scaling_min

        self.mpc_model.update_conditions(U_inf=estimated_ws, TI=estimated_TI, wd=estimated_wd)

        current_yaws_orig = self.current_yaw.copy()
        current_yaws_sorted = current_yaws_orig[self.mpc_model.sorted_indices]


        # Step 2: optimize the yaw angles with FAST parameters
        optimized_params = optimize_farm_back2front(
            self.mpc_model, current_yaws_sorted,
            r_gamma=self.yaw_step_sim/self.dt_sim,  # yaw rate (deg/s)
            t_AH=self.mpc_t_AH,                      # action horizon (s)
            dt_opt=self.mpc_dt_opt,                  # optimization time step (s) - OPTIMIZED
            T_opt=self.mpc_T_opt,                    # prediction horizon (s) - OPTIMIZED
            maxfun=self.mpc_maxfun,                  # max function evaluations - OPTIMIZED
            seed=42,
            initial_params=self.previous_opt_params  # warm-start from previous solution
        )

        self.previous_opt_params = optimized_params.copy()


        # Use the optimized parameters to run the delay model and get the next yaw angles
        t_action, trajectories, _ = run_farm_delay_loop_optimized(
            self.mpc_model, optimized_params, current_yaws_sorted,
            r_gamma=0.3, t_AH=self.mpc_t_AH, dt=self.dt_sim, T=self.dt_mpc
        )

        # The next yaw angles are the simply the last element of each trajectory
        next_yaws_sorted = np.array([traj[-1] for traj in trajectories])
        next_yaws_orig = next_yaws_sorted[self.mpc_model.unsorted_indices]

        # The action should then be:
        yaw_action = (next_yaws_orig - self.yaw_min) / (self.yaw_max - self.yaw_min) * 2 - 1

        # Do the env step:
        obs, reward, done, truncated, info = super().step(yaw_action)

        if truncated:
            self.mpc_model.cache.clear()

        # Add the estimated conditions to the info dict
        info['estimated_wd'] = estimated_wd
        info['estimated_ws'] = estimated_ws
        info['estimated_TI'] = estimated_TI
        info['optimized_yaws'] = next_yaws_orig

        return obs, reward, done, truncated, info

    def _init_spaces(self):
        """
        This function initializes the observation and action spaces.
        This is done in a seperate function, so we can replace it in the multi agent version of the environment
        """
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=((self.obs_var),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=((3),), dtype=np.float32
        )
