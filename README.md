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

### Main Training & Evaluation Scripts

| Path | Description |
|------|-------------|
| `sac_MPC_local.py` | Main entry point for SAC + MPC training loop. |
| `eval_sac_mpc.py` | Evaluate trained SAC + MPC agents on wind scenarios. |
| `eval_mpc_baselines.py` | Evaluate MPC baseline agents (Oracle, FrontTurbine, SimpleEstimator). |
| `eval_greedy_baseline.py` | Evaluate greedy baseline (no control). |
| `mpc_baseline_agents.py` | Baseline agent implementations for comparison. |
| `benchmark_optimizers.py` | Benchmark different optimization methods for MPC. |

### Core Package

| Path | Description |
|------|-------------|
| `mpcrl/` | Core MPC utilities, environment wrappers, and configuration. |
| `mpcrl/mpc.py` | Wind farm MPC model, optimization functions, and yaw control. |
| `mpcrl/environment.py` | SAC training environment (MPCenv). |
| `mpcrl/environment_eval.py` | Evaluation environment (MPCenvEval). |
| `mpcrl/config.py` | Configuration utilities. |
| `mpcrl/validation/` | Hyperparameter validation and visualization tools. |

### Data & Documentation

| Path | Description |
|------|-------------|
| `data/` | Wind-field inputs used by the SAC workflow (NetCDF, etc.). |
| `examples/` | Example Jupyter notebooks demonstrating MPCRL usage. |
| `EVALUATION_SUMMARY.md` | Complete guide to the evaluation framework and baselines. |
| `How to Eval.md` | Quick reference for evaluating trained agents. |
| `alternative_approach/` | Archived acados-based NMPC experiments, docs, and scripts. |

---

## Evaluating Models

After training an agent, you can evaluate it against baselines using the evaluation scripts. See `EVALUATION_SUMMARY.md` for a complete guide or `How to Eval.md` for a quick reference.

**Example evaluation workflow:**

```bash
# Evaluate trained SAC+MPC agent
python eval_sac_mpc.py --model_folder SAC1006

# Evaluate baselines
python eval_greedy_baseline.py
python eval_mpc_baselines.py --agent_type oracle
python eval_mpc_baselines.py --agent_type front_turbine
python eval_mpc_baselines.py --agent_type simple_estimator
```

---

## Examples and Notebooks

The `examples/` directory contains Jupyter notebooks demonstrating the MPCRL package. See `examples/README.md` for details on running the notebooks.

---

## Contributing

Issues and pull requests focusing on the SAC + MPC stack are welcome. If you experiment with the archived acados workflow, please keep related files inside `alternative_approach/` to avoid mixing the two approaches.
