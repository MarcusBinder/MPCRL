# MPCRL Documentation (SAC + MPC Focus)

This repository now centres on the original **SAC-enhanced MPC** controller implemented in `sac_MPC_local.py`. All material related to the experimental acados/surrogate workflow has been archived under `alternative_approach/`.

## Contents

- `sac_MPC_local.py` – main training + control loop combining Soft Actor-Critic with MPC rollouts.
- `mpcrl/` – core environment helpers, configuration, and MPC utilities.
- `data/` – wind-field datasets required by the SAC pipeline (large NetCDF files remain git ignored).

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Launch training: `python sac_MPC_local.py`
3. Inspect results in `runs/` or your configured logging directory.

## Alternative Approach

Curious about the archived acados-based NMPC experiments (linearised cost, surrogate modelling, etc.)? See the materials in `alternative_approach/`:

- `alternative_approach/docs/` – full investigation notes
- `alternative_approach/nmpc_windfarm_acados_fixed.py` – acados controller
- `alternative_approach/scripts/` – data generation & analysis helpers
- `alternative_approach/tests/` – legacy regression and diagnostic suites

Those files remain available for reference without cluttering the primary SAC-first workflow.

