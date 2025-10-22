"""
Baseline agents for MPC evaluation without RL.
These agents provide different strategies for estimating wind conditions to feed to the MPC controller.
"""

import numpy as np
import torch


class OracleMPCAgent:
    """
    Oracle agent that provides perfect wind condition information to the MPC controller.
    This represents the upper bound of MPC performance with perfect forecasting.
    """

    def __init__(self, env):
        """
        Args:
            env: The MPCenvEval environment
        """
        self.env = env
        self.model_type = "Oracle"  # For compatibility with eval script

        # Get the scaling ranges from the environment
        self.wd_min = env.wd_scaling_min
        self.wd_max = env.wd_scaling_max
        self.ws_min = env.ws_scaling_min
        self.ws_max = env.ws_scaling_max
        self.ti_min = env.ti_scaling_min
        self.ti_max = env.ti_scaling_max

    def get_action(self, obs, deterministic=False):
        """
        Returns the true wind conditions from the environment, scaled to [-1, 1].

        Args:
            obs: Observation from environment (not used, since we read directly from env)
            deterministic: Not used for oracle agent

        Returns:
            action: [estimated_wd, estimated_ws, estimated_TI] scaled to [-1, 1]
            log_prob: None (for compatibility)
            mean: None (for compatibility)
        """
        # Get true wind conditions directly from the environment
        true_wd = self.env.wd
        true_ws = self.env.ws
        true_ti = self.env.ti

        # Scale to [-1, 1] range (same as RL agent output)
        wd_scaled = 2 * (true_wd - self.wd_min) / (self.wd_max - self.wd_min) - 1
        ws_scaled = 2 * (true_ws - self.ws_min) / (self.ws_max - self.ws_min) - 1
        ti_scaled = 2 * (true_ti - self.ti_min) / (self.ti_max - self.ti_min) - 1

        action = np.array([wd_scaled, ws_scaled, ti_scaled], dtype=np.float32)

        # Return as torch tensor for compatibility with eval script
        action_tensor = torch.from_numpy(action).unsqueeze(0)

        return action_tensor, None, None

    def predict(self, obs, deterministic=False):
        """
        For compatibility with non-CleanRL models.
        """
        action, _, _ = self.get_action(obs, deterministic)
        return action.cpu().numpy().flatten(), None


class FrontTurbineMPCAgent:
    """
    Agent that estimates wind conditions from measurements at the front-most turbine.
    This represents a practical baseline using available sensor data.
    """

    def __init__(self, env, smoothing_window=3):
        """
        Args:
            env: The MPCenvEval environment
            smoothing_window: Number of timesteps to use for moving average smoothing
        """
        self.env = env
        self.model_type = "FrontTurbine"  # For compatibility with eval script
        self.smoothing_window = smoothing_window

        # Get the scaling ranges from the environment
        self.wd_min = env.wd_scaling_min
        self.wd_max = env.wd_scaling_max
        self.ws_min = env.ws_scaling_min
        self.ws_max = env.ws_scaling_max
        self.ti_min = env.ti_scaling_min
        self.ti_max = env.ti_scaling_max

        # History buffers for smoothing
        self.wd_history = []
        self.ws_history = []
        self.ti_history = []

    def get_action(self, obs, deterministic=False):
        """
        Estimates wind conditions from the front-most turbine measurements.
        Uses measurements from the environment's farm_measurements or directly from flowsimulator.

        Args:
            obs: Observation from environment
            deterministic: Not used for this agent

        Returns:
            action: [estimated_wd, estimated_ws, estimated_TI] scaled to [-1, 1]
            log_prob: None (for compatibility)
            mean: None (for compatibility)
        """
        # Get measurements from the front-most turbine
        # The front turbine is the one with lowest x-coordinate (assuming wind from west)
        # We need to identify which turbine is at the front

        # Get current wind direction to determine which turbine is at front
        current_wd = self.env.wd

        # For simplicity, we'll use turbine 0 as the "front" turbine
        # In a more sophisticated version, we could rotate coordinates based on wind direction

        # Get measurements from the flowsimulator
        fs = self.env.fs

        # Get wind speed at the front turbine (rotor-averaged)
        turbine_ws = np.linalg.norm(fs.windTurbines.rotor_avg_windspeed[0])

        # Estimate wind direction - this is trickier from a single turbine
        # We can use the wind vector at the turbine location
        wind_vector = fs.windTurbines.rotor_avg_windspeed[0]
        estimated_wd = np.arctan2(wind_vector[1], wind_vector[0]) * 180 / np.pi
        # Convert to meteorological convention (0 = north, 90 = east)
        estimated_wd = (90 - estimated_wd) % 360

        # Estimate TI - in practice this would come from high-frequency measurements
        # For now, use the environment's TI as a proxy (in reality, we'd estimate from variance)
        # This is a simplification; real estimation would use std dev of wind speed measurements
        estimated_ti = self.env.ti  # Placeholder - ideally estimate from measurements

        # Add to history and compute moving average
        self.wd_history.append(estimated_wd)
        self.ws_history.append(turbine_ws)
        self.ti_history.append(estimated_ti)

        if len(self.wd_history) > self.smoothing_window:
            self.wd_history.pop(0)
            self.ws_history.pop(0)
            self.ti_history.pop(0)

        # Compute smoothed estimates
        smooth_wd = np.mean(self.wd_history)
        smooth_ws = np.mean(self.ws_history)
        smooth_ti = np.mean(self.ti_history)

        # Scale to [-1, 1] range (same as RL agent output)
        wd_scaled = 2 * (smooth_wd - self.wd_min) / (self.wd_max - self.wd_min) - 1
        ws_scaled = 2 * (smooth_ws - self.ws_min) / (self.ws_max - self.ws_min) - 1
        ti_scaled = 2 * (smooth_ti - self.ti_min) / (self.ti_max - self.ti_min) - 1

        # Clip to valid range
        wd_scaled = np.clip(wd_scaled, -1, 1)
        ws_scaled = np.clip(ws_scaled, -1, 1)
        ti_scaled = np.clip(ti_scaled, -1, 1)

        action = np.array([wd_scaled, ws_scaled, ti_scaled], dtype=np.float32)

        # Return as torch tensor for compatibility with eval script
        action_tensor = torch.from_numpy(action).unsqueeze(0)

        return action_tensor, None, None

    def predict(self, obs, deterministic=False):
        """
        For compatibility with non-CleanRL models.
        """
        action, _, _ = self.get_action(obs, deterministic)
        return action.cpu().numpy().flatten(), None

    def reset(self):
        """
        Reset the history buffers. Call this at the start of each evaluation episode.
        """
        self.wd_history = []
        self.ws_history = []
        self.ti_history = []


class SimpleEstimatorMPCAgent:
    """
    A simpler estimator that uses the observations directly from the environment.
    This extracts wind condition estimates from the observation vector.
    """

    def __init__(self, env):
        """
        Args:
            env: The MPCenvEval environment
        """
        self.env = env
        self.model_type = "SimpleEstimator"

        # Get the scaling ranges
        self.wd_min = env.wd_scaling_min
        self.wd_max = env.wd_scaling_max
        self.ws_min = env.ws_scaling_min
        self.ws_max = env.ws_scaling_max
        self.ti_min = env.ti_scaling_min
        self.ti_max = env.ti_scaling_max

    def get_action(self, obs, deterministic=False):
        """
        Extracts wind condition estimates directly from the observation vector.
        Assumes the observations contain wind measurements that can be used for estimation.

        Args:
            obs: Observation from environment
            deterministic: Not used

        Returns:
            action: [estimated_wd, estimated_ws, estimated_TI] scaled to [-1, 1]
            log_prob: None
            mean: None
        """
        # The observation vector contains measurements
        # We need to extract wind speed, wind direction, and TI estimates
        # This depends on the observation space configuration

        # For the MPCenv, the observation likely includes:
        # - Turbine-level wind measurements (ws, wd)
        # - Farm-level measurements
        # We'll use farm-level or front turbine measurements

        # Get farm measurements
        farm_measurements = self.env.farm_measurements

        # Get farm-level wind measurements (unscaled)
        if hasattr(farm_measurements, 'get_ws_farm'):
            estimated_ws = farm_measurements.get_ws_farm(scaled=False)
            if isinstance(estimated_ws, np.ndarray):
                estimated_ws = estimated_ws.mean()
            # Fallback if NaN
            if np.isnan(estimated_ws):
                estimated_ws = self.env.ws
        else:
            estimated_ws = self.env.ws

        if hasattr(farm_measurements, 'get_wd_farm'):
            estimated_wd = farm_measurements.get_wd_farm(scaled=False)
            if isinstance(estimated_wd, np.ndarray):
                estimated_wd = estimated_wd.mean()
            # Fallback if NaN
            if np.isnan(estimated_wd):
                estimated_wd = self.env.wd
        else:
            estimated_wd = self.env.wd

        if hasattr(farm_measurements, 'get_TI'):
            estimated_ti = farm_measurements.get_TI(scaled=False)
            if isinstance(estimated_ti, np.ndarray):
                estimated_ti = estimated_ti.mean()
            # Fallback if NaN
            if np.isnan(estimated_ti):
                estimated_ti = self.env.ti
        else:
            estimated_ti = self.env.ti

        # Scale to [-1, 1] range
        wd_scaled = 2 * (estimated_wd - self.wd_min) / (self.wd_max - self.wd_min) - 1
        ws_scaled = 2 * (estimated_ws - self.ws_min) / (self.ws_max - self.ws_min) - 1
        ti_scaled = 2 * (estimated_ti - self.ti_min) / (self.ti_max - self.ti_min) - 1

        # Clip to valid range
        wd_scaled = np.clip(wd_scaled, -1, 1)
        ws_scaled = np.clip(ws_scaled, -1, 1)
        ti_scaled = np.clip(ti_scaled, -1, 1)

        action = np.array([wd_scaled, ws_scaled, ti_scaled], dtype=np.float32)
        action_tensor = torch.from_numpy(action).unsqueeze(0)

        return action_tensor, None, None

    def predict(self, obs, deterministic=False):
        """
        For compatibility with non-CleanRL models.
        """
        action, _, _ = self.get_action(obs, deterministic)
        return action.cpu().numpy().flatten(), None
