# Current Status - Normalization Fix Applied

**Date:** 2025-11-07
**Branch:** `claude/work-in-progress-011CUrmxkxjD4gbrA8qr6LEF`

---

## Summary

✅ **Normalization fix implemented and committed**

The MPC solver convergence issue has been addressed by implementing manual normalization in the CasADi cost function.

---

## Problem That Was Fixed

**Issue:** MPC solver not converging, surrogate predictions off by ~14 MW

**Root Cause:** TorchScript doesn't properly trace PyTorch's `register_buffer` normalization when l4casadi converts the model to CasADi.

**Solution:** Manual normalization/denormalization in `nmpc_surrogate_casadi.py`:
- Extract normalization parameters from PyTorch model
- Manually normalize inputs before passing to surrogate
- Manually denormalize outputs from surrogate

---

## Changes Made (Commit: d3a2c67)

### File: `nmpc_surrogate_casadi.py`

#### 1. Extract Normalization Parameters (in `__init__`)
```python
# Extract normalization parameters
self.input_mean = np.array(self.pytorch_model.input_mean.cpu())
self.input_std = np.array(self.pytorch_model.input_std.cpu())
self.output_mean = float(self.pytorch_model.output_mean.cpu())
self.output_std = float(self.pytorch_model.output_std.cpu())

print("  ✅ Surrogate model loaded")
print(f"  Normalization: input_mean={self.input_mean}, input_std={self.input_std}")
```

#### 2. Manual Normalization in Cost Function (in `_build_mpc`)
```python
# Stage cost loop
for k in range(N):
    yaw_deg = get_state(k)

    # Manually normalize inputs
    raw_input = ca.vertcat(yaw_deg, self.wind.U, self.wind.theta)
    normalized_input = (raw_input - self.input_mean) / (self.input_std + 1e-8)

    # Power from surrogate (on normalized input)
    power_normalized = self.power_func(normalized_input)

    # Denormalize output: y * std + mean
    power = power_normalized * self.output_std + self.output_mean

    # ... rest of cost function
```

#### 3. Manual Normalization in Terminal Cost
```python
# Terminal cost
yaw_deg_N = get_state(N)
raw_input_N = ca.vertcat(yaw_deg_N, self.wind.U, self.wind.theta)
normalized_input_N = (raw_input_N - self.input_mean) / (self.input_std + 1e-8)
power_normalized_N = self.power_func(normalized_input_N)
power_N = power_normalized_N * self.output_std + self.output_mean
J += -power_N
```

---

## What This Fixes

**Before:**
- Surrogate predictions: ~14 MW mean error vs PyTorch model
- Relative error: ~162%
- MPC solver: Not converging (hitting max iterations)

**After (Expected):**
- Surrogate predictions: Should match PyTorch model (< 100 W error)
- Relative error: < 0.01%
- MPC solver: Should converge in ~20-50 iterations

---

## Next Steps to Test

### 1. Install Dependencies (if not already done)
```bash
pip install torch l4casadi casadi numpy
```

### 2. Validate Normalization Fix
```bash
cd /home/user/MPCRL/alternative_approach
python validate_normalization.py
```

**Expected output:**
```
✅ Manual normalization is working correctly!
   Error is negligible (< 100 W)
```

### 3. Run MPC Demo
```bash
python nmpc_surrogate_casadi.py
```

**Expected output:**
```
Loading surrogate model...
  ✅ Surrogate model loaded
  Normalization: input_mean=[...], input_std=[...]
Building MPC optimization problem...
  ✅ MPC ready

Running MPC...

Step 0:
  Success: True
  Yaw: [... optimal yaw angles ...]
  Power: X.XXX MW
  Solve time: ~100 ms
  Iterations: ~20-50

Step 1:
  ...
```

---

## Files Affected

- ✅ **nmpc_surrogate_casadi.py** - Main fix applied
- ✅ **validate_normalization.py** - New validation script
- 📝 **STATUS.md** - This file

---

## Technical Details

### Why Manual Normalization is Needed

**Problem with Built-in Normalization:**
```python
# In surrogate_module/model.py
class PowerSurrogate(nn.Module):
    def __init__(self, ...):
        # These buffers are NOT properly traced by TorchScript
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
```

When l4casadi uses TorchScript to convert PyTorch → CasADi:
1. TorchScript traces the forward pass
2. It doesn't properly handle `register_buffer` operations
3. The CasADi function expects **already normalized** inputs
4. But the traced function doesn't include the normalization logic

**Solution:**
- Keep the PyTorch model as-is (with built-in normalization)
- But in CasADi, do normalization manually using extracted parameters
- This ensures the CasADi function receives properly normalized inputs

### Normalization Math

```
Input normalization:  x_norm = (x - mean) / (std + 1e-8)
Output denormalization: y = y_norm * std + mean
```

Where:
- `x`: Raw input [yaw_0, yaw_1, yaw_2, yaw_3, wind_speed, wind_direction]
- `mean`, `std`: Extracted from PyTorch model's buffers
- `x_norm`: Normalized input (what the neural network expects)
- `y_norm`: Normalized output (what the neural network produces)
- `y`: Denormalized power (actual MW)

---

## Alternative Approaches (Not Chosen)

### Option 1: Retrain Without Built-in Normalization ❌
**Pros:** Clean, no manual normalization needed
**Cons:** Requires retraining model, regenerating dataset
**Why not:** Too much rework, current approach works

### Option 2: Fix l4casadi TorchScript Tracing ❌
**Pros:** Would work automatically
**Cons:** Requires changing l4casadi internals
**Why not:** Outside our control, manual fix is simpler

### Option 3: Manual Normalization (Chosen) ✅
**Pros:** Works immediately, no retraining, simple
**Cons:** Slight code duplication (normalization in two places)
**Why chosen:** Fastest path to working solution

---

## Validation Strategy

### Step 1: Validate Normalization Match
Test that manual normalization produces identical results to PyTorch model:
```python
# PyTorch (with built-in normalization)
power_pytorch = model(x)

# CasADi (with manual normalization)
x_norm = (x - mean) / std
power_norm = surrogate_casadi(x_norm)
power_casadi = power_norm * std + mean

# Should be identical (< 100 W difference)
assert abs(power_pytorch - power_casadi) < 100
```

### Step 2: Test MPC Convergence
Run MPC and check:
- ✅ Solver converges (status = 0)
- ✅ Solve time < 200 ms (ideally ~100 ms)
- ✅ Iterations < 100 (ideally 20-50)
- ✅ Power output is reasonable (4-6 MW for 4 turbines)

### Step 3: Validate Optimal Solution
Compare MPC solution to known baseline:
- Baseline (no wake steering): ~5.3 MW
- Expected with optimal yaw: ~6.1 MW (~15% gain)

---

## Success Criteria

✅ **Normalization validation:** < 100 W error vs PyTorch
✅ **MPC convergence:** Solver succeeds > 95% of time
✅ **Solve time:** < 200 ms per step
✅ **Power gain:** ~10-15% vs baseline

---

## If Still Not Working

### Debugging Steps:

1. **Check normalization parameters:**
   ```python
   print(f"input_mean: {self.input_mean}")
   print(f"input_std: {self.input_std}")
   print(f"output_mean: {self.output_mean}")
   print(f"output_std: {self.output_std}")
   ```
   Should be non-zero, reasonable values.

2. **Test surrogate prediction manually:**
   ```python
   # Test at known point
   x = ca.DM([0, 0, 0, 0, 8, 270])
   x_norm = (x - input_mean) / input_std
   power = power_func(x_norm) * output_std + output_mean
   print(f"Power at zero yaw: {float(power)/1e6:.3f} MW")
   # Should be ~5.3 MW for 4 turbines
   ```

3. **Check solver settings:**
   - Increase `max_iter` if hitting iteration limit
   - Adjust `lam_move` if control too aggressive/conservative
   - Check bounds are reasonable

4. **Visualize cost landscape:**
   - Plot power vs yaw angles
   - Ensure smooth, reasonable shape
   - Check gradients are non-zero

---

## References

- **Main MPC file:** `nmpc_surrogate_casadi.py`
- **Surrogate model:** `surrogate_module/model.py`
- **l4casadi export:** `scripts/export_l4casadi_model.py`
- **Normalization fix commit:** d3a2c67

---

**Status:** ✅ Fix implemented, ready for testing
**Next Action:** Run validation and MPC demo
**Expected Result:** MPC converges and finds optimal yaw angles
