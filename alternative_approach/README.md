# Alternative Approach (acados-based NMPC)

This folder contains the archived gradient-based MPC experiments built on acados, along with the surrogate-modelling prototypes.

## Structure

- `docs/` – detailed investigation notes, limitations, and design proposals.
- `nmpc_windfarm_acados_fixed.py` – primary acados controller implementation.
- `scripts/` – utilities for diagnostics, surrogate dataset creation, and training.
- `surrogate_module/` – helper modules for dataset management, model training, and CasADi integration.
- `tests/` – legacy regression/diagnostic suites retained for reference.
- `mpcrl_archive/` – older NMPC prototypes superseded by the SAC workflow.

These assets are no longer part of the main workflow but remain available for future reference or experimentation.

