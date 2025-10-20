# Surrogate-Based NMPC Roadmap

**Updated:** 2025-10-20  
**Owner:** MPC/acados refactor

---

## Objective

Replace the linearized cost in `nmpc_windfarm_acados_fixed.py` with a differentiable surrogate of PyWake farm power so that the acados solver can optimise over the true nonlinear landscape (large yaw offsets, wake delays) without relying on a strategic setpoint layer.

---

## Phase Breakdown

### Phase 1 – Foundations
- [ ] Confirm state/delay requirements for surrogate cost (what portion of the yaw history drives power at each stage).
- [x] Finalise dataset schema (feature ordering, units, metadata, storage format).
- [x] Implement dataset generation script that calls PyWake with sampled yaw histories and records the delayed yaw snapshot plus wind conditions.
- [x] Add CLI hooks/config so we can control sample counts, random seeds, and output paths.

### Phase 2 – Surrogate Model
- [x] Build training pipeline (normalisation, train/val/test split, logging).
- [x] Define neural network architecture (FC layers with smooth activations).
- [ ] Train prototype model and evaluate MAE / gradient agreement vs PyWake on hold‑out set.
- [x] Export weights + normalisation stats in a CasADi-friendly format (JSON/NPZ).

### Phase 3 – NMPC Integration
- [ ] Extend acados model to include any additional delay states needed by the surrogate.
- [ ] Implement CasADi graph for the neural network inside the stage cost.
- [ ] Replace linearised objective with surrogate-based nonlinear cost; keep move penalties and bounds.
- [ ] Tune solver options (horizon length, SQP settings) for stability with the new cost.

### Phase 4 – Validation & Tooling
- [ ] Create regression tests comparing surrogate predictions and gradients with PyWake.
- [ ] Add MPC integration test to verify the controller finds near-optimal yaw (~15% gain) from cold start.
- [ ] Update documentation (`docs/README.md`, `docs/FINAL_RECOMMENDATIONS.md`) with the new single-layer workflow.
- [ ] Provide usage examples (`examples/demo_yaw_optimization_surrogate.py`) and training instructions.

---

## Risks & Mitigations
- **Surrogate accuracy near extremes:** use Latin-hypercube sampling and targeted refinements around known optima; monitor gradient error.
- **State dimension growth:** if full delay queue is required, keep it compact by modelling wake advection with coarse taps or learned embeddings.
- **Solver convergence:** start with moderate horizons (e.g., 40 × 10 s) and gradually increase; leverage gauss-newton Hessian and warm starts.

---

## Immediate TODO
1. ✅ Document roadmap (this file).
2. ✅ Implement dataset generator under `scripts/`.
3. ✅ Add normalisation utilities and a training scaffold.
4. ☐ Prototype surrogate integration with offline power replay before wiring into acados.
5. ☐ Expand tests + docs once nonlinear MPC shows ≥10% gain improvement.
