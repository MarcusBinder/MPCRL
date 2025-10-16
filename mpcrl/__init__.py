"""
MPCRL - Model Predictive Control for Wind Farm Optimization

A wind farm control optimization system that combines Model Predictive Control (MPC)
with Reinforcement Learning to maximize power output through intelligent wake steering.
"""

__version__ = "0.1.0"

# Import main classes and functions for easy access
from .mpc import (
    WindFarmModel,
    YawCache,
    optimize_farm_back2front,
    farm_energy,
    run_farm_delay_loop_optimized,
    yaw_traj,
    psi,
    sat01,
)

from .environment import MPCenv

from .config import make_config

__all__ = [
    # Core MPC functionality
    "WindFarmModel",
    "YawCache",
    "optimize_farm_back2front",
    "farm_energy",
    "run_farm_delay_loop_optimized",
    "yaw_traj",
    "psi",
    "sat01",
    # Environment
    "MPCenv",
    # Configuration
    "make_config",
]
