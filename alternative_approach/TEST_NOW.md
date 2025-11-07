# Quick Test Guide ✅

**Root Cause Found:** TorchScript can't trace conditional logic (commit b3405a3)

---

## What Happened

The original PyTorch model has `if not normalized:` conditional logic that **TorchScript can't properly trace**. This caused l4casadi to export the model WITHOUT proper normalization.

**Result:** CasADi predictions wrong (12 MW instead of 4.8 MW!)

**Fix:** Created `SimplePowerSurrogate` wrapper without conditionals that TorchScript can trace properly.

---

## Fix Steps

### FIRST: Re-export the model
```bash
cd ~/Documents/mpcrl/alternative_approach
python scripts/export_l4casadi_model_v2.py
```

This will re-export with the fixed wrapper. Should show "✅ Validation passed!"

---

## Then Test

### 1. Validate Export (should take ~1 second)
```bash
cd ~/Documents/mpcrl/alternative_approach
python validate_normalization.py
```

**Expected:**
```
✅ CasADi export is working correctly!
   Error is negligible (< 100 W)
```

### 2. Run MPC Demo (should take ~10 seconds)
```bash
python nmpc_surrogate_casadi.py
```

**Expected:**
- Solver converges (Success: True)
- Finds non-zero yaw angles (NOT stuck at [0, 0, 0, 0])
- Power increases ~15-20% vs baseline
- Solve time ~100 ms per step

---

## If It Works 🎉

Great! The surrogate MPC is functional. Next steps:
- Generate more training data (10k-100k samples)
- Retrain model with more data
- Tune MPC parameters
- Consider WindGym integration

---

## If It Still Doesn't Work 😞

**Check validation first:**
- If validation fails → l4casadi export issue
- If validation passes but MPC doesn't optimize → cost function or solver issue

**Possible issues (NOT normalization!):**
1. Model not trained well (only 800 samples)
2. Surrogate not accurate enough for optimization
3. MPC parameters need tuning (lam_move, N, etc.)
4. Solver settings need adjustment

See `DOUBLE_NORMALIZATION_BUG_FIX.md` for full details.

---

**Quick summary:** Just run the two commands above and let me know what happens! 🚀
