# MPCRL – SAC + MPC for Wind Farm Wake Steering

This project implements a **Soft Actor-Critic (SAC)** agent that collaborates with a model-predictive controller to maximise wind farm power through wake steering. The SAC policy proposes yaw adjustments while MPC enforces turbine constraints and smooth trajectories.

The experimental acados/surrogate NMPC work that previously lived at the top level has been archived under `alternative_approach/` so the main repository can stay focused on the SAC workflow.

---

## Quick Start

```bash
git clone https://github.com/<your-user>/mpcrl.git
cd mpcrl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# launch the SAC + MPC training loop
python sac_MPC_local.py
```

Large wind-field datasets live under `data/` and remain ignored by git (see `data/README.md` for details).

---

## Repository Layout

| Path | Description |
|------|-------------|
| `sac_MPC_local.py` | Entry point for SAC training / evaluation. |
| `mpcrl/` | Core MPC utilities, environment wrappers, and configuration helpers. |
| `data/` | Wind-field inputs used by the SAC workflow (NetCDF, etc.). |
| `alternative_approach/` | Archived acados-based NMPC experiments, docs, and scripts. |

See `docs/README.md` for a short guide to the SAC pipeline, and browse `alternative_approach/` if you need the previous gradient-based MPC exploration.

---

## Contributing

Issues and pull requests focusing on the SAC + MPC stack are welcome. If you experiment with the archived acados workflow, please keep related files inside `alternative_approach/` to avoid mixing the two approaches.

