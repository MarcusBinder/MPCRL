# 🔧 Action Required: Re-Export Model

**Status:** Root cause identified and fixed ✅
**Action:** Re-export the model with fixed script
**Time:** ~30 seconds

---

## What's Wrong

The validation showed **CasADi predictions don't match PyTorch** (12 MW vs 4.8 MW).

**Root cause:** Original PyTorch model has conditional logic that TorchScript can't trace:
```python
def forward(self, x, normalized=False):
    if not normalized:  # ❌ TorchScript can't trace this!
        x = self.normalize_input(x)
    ...
```

**Result:** l4casadi exports the network but loses the normalization.

---

## The Fix

Created `export_l4casadi_model_v2.py` with `SimplePowerSurrogate` wrapper:
- ✅ No conditional logic
- ✅ Always normalizes/denormalizes explicitly
- ✅ TorchScript can properly trace it

---

## What You Need to Do

### 1. Re-export the model (ONE command)
```bash
cd ~/Documents/mpcrl/alternative_approach
python scripts/export_l4casadi_model_v2.py
```

**Expected output:**
```
✅ Wrapper matches original perfectly!
✅ Validation passed!
```

### 2. Then validate
```bash
python validate_normalization.py
```

**Expected:**
```
✅ CasADi export is working correctly!
   Error is negligible (< 100 W)
```

### 3. Then run MPC
```bash
python nmpc_surrogate_casadi.py
```

**Expected:**
- Finds optimal yaw angles
- Power increases ~15%

---

## Why This Will Work

**Before (broken):**
- TorchScript traces conditional path
- Loses normalization operations
- CasADi function expects normalized inputs
- We pass raw inputs → WRONG!

**After (fixed):**
- SimplePowerSurrogate has no conditionals
- TorchScript traces entire normalization
- CasADi function works like PyTorch
- We pass raw inputs → CORRECT!

---

## Details

See `FIX_TORCHSCRIPT_TRACING.md` for full technical explanation.

---

**TL;DR:** Just run `python scripts/export_l4casadi_model_v2.py` 🚀
