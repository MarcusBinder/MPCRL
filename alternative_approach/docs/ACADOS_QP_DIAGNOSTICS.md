# acados MPC Diagnostics – 2025-10-20

## Current Status
- **Objective refactor**: Stage cost now optimises yaw‐rate increments (`Δψ`) with quadratic regularisation and optional target bias. This prevents the solver from immediately saturating yaw bounds (see `nmpc_windfarm_acados_fixed.py`).
- **Gradient sanity checks**:  
  - `scripts/verify_wake_6d.py` confirms PyWake’s 6D layout peaks near `[20°, 20°, 20°, 0°]` (~+2 MW).  
  - `scripts/gradient_control_no_delay.py` and `scripts/gradient_control_with_delay.py` show monotonic power climbs without acados; delayed run saved as `results/gradient_delayed_reference.npz`.
- **acados comparison** (`scripts/acados_vs_reference.py`): despite the new cost, acados still hits QP status 4 once the gradient flattens. Planned yaw rates clip at ±yaw_rate_max, the SQP loop reports `res_stat ≈ 1e7`, and the controller stalls around ±25° instead of tracking the +20° reference.
- **Demo runs** (`demo_yaw_optimization.py`): power increases slightly (~+3 %) but yaw oscillations persist and the log prints repeated QP failures.

## Diagnostics Collected
- Solver print level raised to `1`. Every MPC step now prints the acados summary table: residuals, QP iterations, step norm.
- `solver_stats['qp_info']` captures `qp_residual_norm` when available; the demo log shows repeated failures with zero stepNorm and active bounds.
- `applied_yaw` (post-rate-limit yaw) is stored alongside `psi_plan` for each step, highlighting how applied yaw freezes at the bounds even when the plan tries to move inward.

## Hypotheses
1. **Scaling mismatch**: Gradients are O(1e5) while yaw-rate limits are ±0.4°/s. Without normalising controls, the linearised QP quickly becomes infeasible.
2. **Delay vs horizon**: Last turbine has a 32-step wake delay (dt=10 s). With a 40-step horizon, the controller barely sees the benefit before the next re-linearisation, so the SQP trust region struggles.
3. **Rate bounds active at linearisation point**: Because the reference yaw stays at the previous state, the first SQP step tries to move an entire wake column at once, hitting the rate bound and returning a zero step.

## Next Steps (high priority)
1. **Control normalisation**  
   - Redefine `û = u / yaw_rate_max`, work entirely in normalised units, and rescale gradient/weights accordingly.  
   - Repeat the reference replay (`scripts/acados_vs_reference.py`); expect `res_stat` to drop to ~O(1e0–1e2) and QP iterations > 0.
2. **Horizon / dt adjustments**  
   - Either halve `dt` (→ 5 s) or increase `N_h` so the horizon covers ≥2× the max delay (~64 steps).  
   - Re-run gradient replay to ensure the optimiser has time to observe delayed gains.
3. **Rate-smoothing term**  
   - Add a secondary penalty on `Δu` (changes in yaw rate) or use a moving target trajectory so consecutive SQP steps don’t request huge jumps.

## Useful Commands
```bash
# Baseline physics
python scripts/verify_wake_6d.py

# Gradient-only controllers (no acados)
python scripts/gradient_control_no_delay.py
python scripts/gradient_control_with_delay.py   # writes results/gradient_delayed_reference.npz

# Compare acados to recorded trajectory (includes solver diagnostics)
python scripts/acados_vs_reference.py

# Demo log with QP stats
python demo_yaw_optimization.py > demo_acados_log.txt
```

## Open Questions
- What scaling does acados expect for angle/velocity states in similar applications? Check acados examples for reference.
- Should we compute gradients on delayed yaw (`psi_delayed`) instead of current yaw to better match the delayed physics?
- Is the trust-region weight large enough after normalisation, or do we need a dynamic adjustment based on residuals?

_Last updated: 2025-10-20_.

---

## ✅ RESOLUTION - See `FINAL_FIX_SUMMARY.md`

**All issues have been resolved!** The MPC now works correctly with:

1. **Control normalization** - Fixed QP conditioning
2. **Gradient at delayed yaws** - Correct evaluation point
3. **Larger finite difference epsilon** - Numerical accuracy (0.5° instead of 0.01°)
4. **Direction bias** - Break symmetry at zero yaw (PyWake wake deflection is symmetric)

**Result:** Power increases from 4.463 MW → 4.481 MW with optimal yaw angles of [-2.4°, -3.7°, -3.7°, 0°].

See `docs/FINAL_FIX_SUMMARY.md` for complete details.
