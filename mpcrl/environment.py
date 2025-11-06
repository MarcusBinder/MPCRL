from random import seed
from WindGym import WindFarmEnv
from .config import make_config
from typing import Any, Dict, Optional, Union
import gymnasium as gym
from .mpc import WindFarmModel, optimize_farm_back2front, run_farm_delay_loop_optimized
import numpy as np


class MPCenv(WindFarmEnv):
    """
    Wind Farm Environment with MPC controller.
    """

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.dt_mpc = self.dt_env  # The MPC time step is the same as the environment time step

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        obs, info = super().reset(seed=seed, options=options)

        self.USE_VARIABLE_HORIZON = True   # Use variable prediction horizon (paper's approach)
        self.USE_TIME_SHIFTED = True       # Use time-shifted cost function (paper's best)
        self.APPLY_YAW_PENALTY = True      # Use Equation 5 penalty for large yaw angles

        self.mpc_model = WindFarmModel(self.x_pos, self.y_pos, 
                                       wt = self.turbine,
                                       D=self.D, 
                                       cache_size=64000, 
                                       cache_quant=0.25, wind_quant=0.25,
                                       apply_yaw_penalty=self.APPLY_YAW_PENALTY)
        self.previous_opt_params = None
        return obs, info

    def step(self, action):
        if self.seed is None:
            seed_for_optim = self.np_random.integers(0, 1e6)
        else:
            seed_for_optim = self.seed + self.timestep
            
            

        # Step 1: update the MPC model with the current state
        estimated_wd = action[0]
        estimated_ws = action[1]
        estimated_TI = action[2]
        # TODO: Should we use _scaling_ or _inflow_, or maybe something else?
        # The estimates are all from -1 to 1, we need to convert them back to the original range
        estimated_wd = (estimated_wd + 1) / 2 * (self.wd_scaling_max - self.wd_scaling_min) + self.wd_scaling_min
        estimated_ws = (estimated_ws + 1) / 2 * (self.ws_scaling_max - self.ws_scaling_min) + self.ws_scaling_min
        estimated_TI = (estimated_TI + 1) / 2 * (self.ti_scaling_max - self.ti_scaling_min) + self.ti_scaling_min

        # Apply minimum thresholds to prevent numerical issues in PyWake
        # MIN_TI = 0.01  # Minimum turbulence intensity
        # MIN_WS = 3.0   # Minimum wind speed

        # estimated_ws = max(estimated_ws, MIN_WS)
        # estimated_TI = max(estimated_TI, MIN_TI)

        # print(f"Estimated conditions - WD: {estimated_wd:.5f}, WS: {estimated_ws:.5f}, TI: {estimated_TI:.5f}")

        self.mpc_model.update_conditions(U_inf=estimated_ws, TI=estimated_TI, wd=estimated_wd)
        
        current_yaws_orig = self.current_yaw.copy()
        current_yaws_sorted = current_yaws_orig[self.mpc_model.sorted_indices]   


        # Step 2: optimize the yaw angles
        optimized_params = optimize_farm_back2front(
            self.mpc_model, 
            current_yaws_sorted, 
            r_gamma=self.yaw_step_sim/self.dt_sim, # yaw rate (deg/s)
            t_AH=100.0,  # action horizon (s)
            dt_opt=20.0,  # optimization time step (s)
            T_opt=500.0,  # prediction horizon (s)
            # maxfun=20,
            seed=seed_for_optim,
            use_time_shifted=False,
            method="direct",
            per_turbine_budget=20,
            verbose=False,
            initial_params=self.previous_opt_params
        )

        self.previous_opt_params = optimized_params.copy()


        # Use the optimized parameters to run the delay model and get the next yaw angles
        t_action, trajectories, _ = run_farm_delay_loop_optimized(
            self.mpc_model, optimized_params, current_yaws_sorted, 
            r_gamma=self.yaw_step_sim/self.dt_sim, 
            t_AH=100.0, dt=self.dt_sim, T=self.dt_mpc
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
    