from .environment import MPCenv
from typing import Optional


class MPCenvEval(MPCenv):
    """
    Evaluation wrapper for MPCenv that allows setting specific wind conditions.
    Similar to FarmEval but for the MPC+RL environment.
    """

    def __init__(
        self,
        turbine,
        x_pos,
        y_pos,
        finite_episode: bool = False,
        # Max and min values for the wind measurements. Used for internal scaling
        ws_scaling_min: float = 0.0,
        ws_scaling_max: float = 30.0,
        wd_scaling_min: float = 0,
        wd_scaling_max: float = 360,
        ti_scaling_min: float = 0.0,
        ti_scaling_max: float = 1.0,
        yaw_scaling_min: float = -45,
        yaw_scaling_max: float = 45,
        yaw_init="Zeros",
        TurbBox="Default",
        config=None,
        Baseline_comp=False,
        render_mode=None,
        turbtype="MannGenerate",
        seed=None,
        dt_sim=1,  # Simulation timestep in seconds
        dt_env=1,  # Environment timestep in seconds
        yaw_step_sim=1,  # Yaw step in degrees per simulation timestep
        yaw_step_env=None,
        n_passthrough=5,
        HTC_path=None,
        reset_init=True,
        fill_window=True,
        sample_site=None,
        burn_in_passthroughs=2,
    ):
        """
        Initialize the MPCenvEval environment.
        This is a wrapper for the MPCenv class that allows setting specific wind conditions for evaluation.
        """
        self.finite_episode = finite_episode

        # Initialize the parent MPCenv
        super().__init__(
            turbine=turbine,
            x_pos=x_pos,
            y_pos=y_pos,
            n_passthrough=n_passthrough,
            ws_scaling_min=ws_scaling_min,
            ws_scaling_max=ws_scaling_max,
            wd_scaling_min=wd_scaling_min,
            wd_scaling_max=wd_scaling_max,
            ti_scaling_min=ti_scaling_min,
            ti_scaling_max=ti_scaling_max,
            yaw_scaling_min=yaw_scaling_min,
            yaw_scaling_max=yaw_scaling_max,
            burn_in_passthroughs=burn_in_passthroughs,
            TurbBox=TurbBox,
            turbtype=turbtype,
            config=config,
            Baseline_comp=Baseline_comp,
            yaw_init=yaw_init,
            render_mode=render_mode,
            seed=seed,
            dt_sim=dt_sim,
            dt_env=dt_env,
            yaw_step_sim=yaw_step_sim,
            yaw_step_env=yaw_step_env,
            HTC_path=HTC_path,
            reset_init=reset_init,
            fill_window=fill_window,
            sample_site=sample_site,
        )
        self.yaml_path = config  # Saved for legacy reasons

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Override reset to allow for infinite episodes during evaluation.
        """
        observation, info = super().reset(seed=seed, options=options)

        # Only set an "infinite" time_max if the finite_episode flag is False
        if not self.finite_episode:
            # This maintains the original "sandbox" behavior for fixed-step evaluations
            self.time_max = 9999999

        return observation, info

    def set_wind_vals(self, ws=None, ti=None, wd=None):
        """
        Set the wind values to be used in the evaluation.
        This method is required for compatibility with AgentEvalFast.
        """
        if ws is not None:
            self.ws = ws
            self.ws_inflow_min = ws
            self.ws_inflow_max = ws
        if ti is not None:
            self.ti = ti
            self.TI_inflow_min = ti
            self.TI_inflow_max = ti
        if wd is not None:
            self.wd = wd
            self.wd_inflow_min = wd
            self.wd_inflow_max = wd

    def set_yaw_vals(self, yaw_vals):
        """
        Set the yaw values to be used in the evaluation.
        This method is required for compatibility with AgentEvalFast.
        """
        self.yaw_initial = yaw_vals

    def update_tf(self, path):
        """
        Overwrite the _def_site method to set the turbulence field to the path given.
        This method is required for compatibility with AgentEvalFast.
        """
        self.TF_files = [path]
