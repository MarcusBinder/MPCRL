#!/usr/bin/env python3
import os
import json
import math
import time
import argparse
import traceback
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== Imports from your project =====
# from your_module import WindFarmModel, run_farm_delay_loop_optimized, optimize_farm_back2front
# from your_module import solve_2d_box  # strict-budget version with logging (already integrated by you)

from mpcrl.mpc import *
import numpy as np
from py_wake.examples.data.hornsrev1 import V80
import matplotlib.pyplot as plt
from windgym.WindGym import PyWakeAgent, WindFarmEnv

# ----------------- YOU MUST FILL THIS -----------------
def make_model():
    """
    Return a NEW WindFarmModel instance (do NOT share between processes).
    Example template:
        xs = np.array([0, 7*80.0, 14*80.0])
        ys = np.array([0, 0, 0])
        wt = <your turbine object>
        return WindFarmModel(xs, ys, wt, D=80.0, U_inf=8.0, TI=0.06, wd=270.0,
                             cache_quant=0.25, cache_size=64000, wind_quant=0.2)

    """
    x_pos = np.array([0, 500, 900])
    y_pos = np.array([0, 0, 0])
    D = 80
    APPLY_YAW_PENALTY = False

    # We can setup the MPC model like this
    mpc_model = WindFarmModel(x_pos, y_pos, D=D, cache_size=64000, 
                            wt=V80(),
                                cache_quant=0.25, wind_quant=0.25,
                                apply_yaw_penalty=APPLY_YAW_PENALTY)
    return mpc_model

    # raise NotImplementedError("Please implement make_model() to return a new WindFarmModel.")
# ------------------------------------------------------

@dataclass
class OptConfig:
    methods: List[str]
    budgets: List[int]
    reps: int = 3
    base_seed: int = 42
    dt_opt: float = 10.0
    T_opt: float = 600.0
    dt_eval: float = 10.0
    T_eval: float = 500.0
    r_gamma: float = 0.3
    t_AH: float = 100.0
    use_time_shifted: bool = False

@dataclass
class Inflow:
    WS: float
    WD: float
    TI: float

def hold_params(n_turbines: int) -> np.ndarray:
    return np.array([[0.5, 0.5]] * n_turbines, dtype=float)

def evaluate_params(model, params, current_yaw_angles_sorted, r_gamma, t_AH, dt, T):
    t0 = time.perf_counter()
    t, traj, P = run_farm_delay_loop_optimized(
        model=model, yaw_params=params,
        current_yaw_angles_sorted=current_yaw_angles_sorted,
        r_gamma=r_gamma, t_AH=t_AH, dt=dt, T=T
    )
    E = float(np.trapezoid(P.sum(axis=0), t))
    eval_time = time.perf_counter() - t0
    return t, traj, P, E, eval_time

def compute_baseline_for_inflow(inflow: Inflow, cfg: OptConfig) -> Tuple[Inflow, float]:
    """Serial: compute baseline (hold yaw) energy for an inflow to avoid redoing it in every worker."""
    model = make_model()
    model.update_conditions(U_inf=inflow.WS, TI=inflow.TI, wd=inflow.WD)
    current_yaw_angles_sorted = np.zeros(model.n_turbines, dtype=float)
    _, _, _, E_base, _ = evaluate_params(
        model, hold_params(model.n_turbines), current_yaw_angles_sorted,
        cfg.r_gamma, cfg.t_AH, cfg.dt_eval, cfg.T_eval
    )
    return inflow, E_base

def worker_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    A single (inflow, method, budget, seed) run. Runs in a separate process.
    Returns a row dict to be appended to the dataframe.
    """
    inflow: Inflow = task["inflow"]
    method: str = task["method"]
    budget: int = int(task["budget"])
    seed: int = int(task["seed"])
    cfg: OptConfig = task["cfg"]
    E_base: float = float(task["E_base"])

    try:
        model = make_model()
        model.update_conditions(U_inf=inflow.WS, TI=inflow.TI, wd=inflow.WD)
        current_yaw_angles_sorted = np.zeros(model.n_turbines, dtype=float)

        # OPTIMIZATION
        t0 = time.perf_counter()
        opt_params = optimize_farm_back2front(
            model=model,
            current_yaw_angles_sorted=current_yaw_angles_sorted,
            r_gamma=cfg.r_gamma,
            t_AH=cfg.t_AH,
            dt_opt=cfg.dt_opt,
            T_opt=cfg.T_opt,
            # maxfun=budget,  # for compatibility with dual_annealing
            seed=seed,
            use_time_shifted=cfg.use_time_shifted,
            method=method,
            per_turbine_budget=budget,
        )
        opt_time = time.perf_counter() - t0

        # EVALUATION
        _, _, _, E, eval_time = evaluate_params(
            model, opt_params, current_yaw_angles_sorted,
            cfg.r_gamma, cfg.t_AH, cfg.dt_eval, cfg.T_eval
        )
        gain_pct = (E - E_base) / E_base * 100.0
        err = ""
    except Exception as ex:
        opt_time = time.perf_counter() - t0 if "t0" in locals() else math.nan
        E, eval_time, gain_pct = math.nan, math.nan, math.nan
        err = f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}"

    return {
        "WS": inflow.WS, "WD": inflow.WD, "TI": inflow.TI,
        "method": method, "budget": budget, "seed": seed,
        "E": E, "E_base": E_base, "gain_pct": gain_pct,
        "opt_time_s": opt_time, "eval_time_s": eval_time,
        "total_time_s": (opt_time + eval_time) if (not math.isnan(opt_time) and not math.isnan(eval_time)) else math.nan,
        "error": err,
    }

def aggregate_and_plot(df: pd.DataFrame, output_dir: str, cfg: OptConfig):
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    agg = (df
        .dropna(subset=["gain_pct", "total_time_s"])
        .groupby(["method", "budget"], as_index=False)
        .agg(mean_gain_pct=("gain_pct", "mean"),
             std_gain_pct=("gain_pct", "std"),
             mean_opt_time_s=("opt_time_s", "mean"),
             std_opt_time_s=("opt_time_s", "std"),
             mean_total_time_s=("total_time_s", "mean"),
             std_total_time_s=("total_time_s", "std"),
             n=("gain_pct", "count"))
        .sort_values(["budget", "mean_gain_pct"], ascending=[True, False])
    )
    agg.to_csv(os.path.join(output_dir, "summary_by_method_budget.csv"), index=False)

    # Δ% vs budget
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for m in agg["method"].unique():
        sub = agg[agg["method"] == m]
        ax1.errorbar(sub["budget"], sub["mean_gain_pct"], yerr=sub["std_gain_pct"], marker="o", label=m)
    ax1.set_xlabel("Budget (evals per turbine)")
    ax1.set_ylabel("Mean Δ% energy vs baseline")
    ax1.set_title("Energy gain vs budget (mean ± std over inflows & reps)")
    ax1.grid(True, alpha=0.3); ax1.legend(); fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "gain_vs_budget.png"), dpi=200)

    # Optimization time vs budget
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for m in agg["method"].unique():
        sub = agg[agg["method"] == m]
        ax2.errorbar(sub["budget"], sub["mean_opt_time_s"], yerr=sub["std_opt_time_s"], marker="o", label=m)
    ax2.set_xlabel("Budget (evals per turbine)")
    ax2.set_ylabel("Mean optimization time [s]")
    ax2.set_title("Optimization time vs budget (mean ± std)")
    ax2.grid(True, alpha=0.3); ax2.legend(); fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "opt_time_vs_budget.png"), dpi=200)

    # Pareto plot
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for m in agg["method"].unique():
        sub = agg[agg["method"] == m]
        ax3.scatter(sub["mean_total_time_s"], sub["mean_gain_pct"], label=m)
        ax3.plot(sub["mean_total_time_s"], sub["mean_gain_pct"])
        for _, r in sub.iterrows():
            ax3.annotate(f"{int(r['budget'])}", (r["mean_total_time_s"], r["mean_gain_pct"]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax3.set_xlabel("Mean total time [s]"); ax3.set_ylabel("Mean Δ% energy")
    ax3.set_title("Pareto: Energy gain vs total time")
    ax3.grid(True, alpha=0.3); ax3.legend(); fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "pareto_gain_vs_time.png"), dpi=200)

    # Winners by budget
    winners = (agg.sort_values(["budget", "mean_gain_pct"], ascending=[True, False])
                  .groupby("budget").first().reset_index()
                  [["budget", "method", "mean_gain_pct", "mean_total_time_s"]])
    winners.to_csv(os.path.join(output_dir, "winners_by_budget.csv"), index=False)

def build_inflows(kind: str) -> List[Inflow]:
    if kind == "grid_small":
        WS = [8.0, 9.0, 10.0]
        WD = [265.0, 270.0, 272.0, 280.0]
        TI = [0.05, 0.06, 0.10]
        return [Inflow(ws, wd, ti) for ws in WS for wd in WD for ti in TI]
    elif kind == "study_default":
        return [
            Inflow(8.0, 270.0, 0.06),
            Inflow(8.0, 265.0, 0.06),
            Inflow(10.0, 270.0, 0.10),
            Inflow(6.0,  275.0, 0.06),
        ]
    else:
        raise ValueError(f"Unknown inflow set: {kind}")

def run_parallel(inflows: List[Inflow], cfg: OptConfig, output_dir: str, n_jobs: int) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    # Metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": asdict(cfg),
        "methods": cfg.methods,
        "inflows": [asdict(inf) for inf in inflows],
        "n_jobs": n_jobs,
        "notes": "Process-parallel benchmark; fresh model per worker",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 1) Baselines (serial)
    inflow_to_Ebase: Dict[Tuple[float,float,float], float] = {}
    for inf in inflows:
        _, E_base = compute_baseline_for_inflow(inf, cfg)
        inflow_to_Ebase[(inf.WS, inf.WD, inf.TI)] = E_base

    # 2) Build all tasks
    tasks: List[Dict[str, Any]] = []
    for inf in inflows:
        E_base = inflow_to_Ebase[(inf.WS, inf.WD, inf.TI)]
        for budget in cfg.budgets:
            for method in cfg.methods:
                for rep in range(cfg.reps):
                    seed = cfg.base_seed + rep
                    tasks.append({
                        "inflow": inf, "budget": budget, "method": method,
                        "seed": seed, "cfg": cfg, "E_base": E_base
                    })

    # 3) Run in parallel
    results = []
    t0_all = time.perf_counter()
    # Use spawn for safety if needed: multiprocessing.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(worker_task, task) for task in tasks]
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done += 1
            if done % 10 == 0 or done == len(futs):
                print(f"[progress] {done}/{len(futs)} completed")

    wall = time.perf_counter() - t0_all
    print(f"\nAll tasks completed in {wall:.2f}s using {n_jobs} workers.")

    df = pd.DataFrame(results)
    aggregate_and_plot(df, output_dir, cfg)
    return df

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="benchmark_out_mp", help="Output directory")
    p.add_argument("--inflows", type=str, default="grid_small", help="grid_small|study_default")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--budgets", type=str, default="5,10,20,30,50")
    p.add_argument("--methods", type=str,
        default="direct,shgo,sobol_powell,dual_annealing,dual_annealing_powell,gp_bo")
    p.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2), help="Number of worker processes")
    return p.parse_args()

def main():
    args = parse_cli()
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    inflows = build_inflows(args.inflows)

    cfg = OptConfig(
        methods=methods, budgets=budgets, reps=args.reps,
        dt_opt=10.0, T_opt=600.0, dt_eval=10.0, T_eval=500.0,
        r_gamma=0.3, t_AH=100.0, use_time_shifted=False
    )

    # Sanity check that make_model() is provided
    try:
        _m = make_model()
        del _m
    except NotImplementedError as e:
        print(str(e))
        print("Please edit make_model() in this file to instantiate your WindFarmModel.")
        return

    df = run_parallel(inflows, cfg, output_dir=args.out, n_jobs=args.jobs)
    print(f"\nSaved detailed results to: {os.path.join(args.out, 'results.csv')}")
    print(f"Saved summary to: {os.path.join(args.out, 'summary_by_method_budget.csv')}")
    print(f"Figures saved under: {args.out}/")

if __name__ == "__main__":
    main()
