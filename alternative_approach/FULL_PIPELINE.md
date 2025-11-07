# Complete Pipeline Guide 🚀

**Status:** You need to run the full pipeline: Generate Data → Train → Export → Test

---

## Prerequisites

Make sure you have the environment activated:
```bash
conda activate mpcrl
cd ~/Documents/mpcrl/alternative_approach
```

---

## Step 1: Generate Training Dataset

Generate dataset from PyWake simulations:

```bash
python scripts/generate_dataset_v2.py
```

**Options:**
- Quick test: 800 samples (default)
- Full dataset: `python scripts/generate_dataset_v2.py --n_samples 10000`

**Expected output:**
```
Generating dataset...
  Sampling 800 LHS samples...
  Running 800 PyWake simulations...
  Progress: [========] 100%
  ✅ Dataset saved to data/surrogate_dataset.npz
```

**Time:** ~1-2 minutes for 800 samples, ~15-20 minutes for 10k samples

---

## Step 2: Train Surrogate Model

Train PyTorch neural network on the dataset:

```bash
python scripts/train_surrogate_v2.py
```

**Expected output:**
```
Loading dataset...
  ✅ Loaded 800 samples

Training...
Epoch 1/100: loss=0.123 val_loss=0.145
Epoch 2/100: loss=0.089 val_loss=0.112
...
Epoch 50/100: loss=0.002 val_loss=0.003

✅ Training complete!
  Best model saved to: checkpoints/power_surrogate_best.ckpt
```

**Time:** ~2-5 minutes (depends on GPU availability)

---

## Step 3: Export to l4casadi

Export trained model for MPC (using the FIXED v2 script):

```bash
python scripts/export_l4casadi_model_v2.py
```

**Expected output:**
```
======================================================================
L4CasADi Model Export (V2 - Fixed)
======================================================================

Loading model...
  ✅ Model loaded and wrapped

Validating wrapper...
  Max absolute difference: 0.00 kW
  ✅ Wrapper matches original perfectly!

Exporting to l4casadi...
  ✅ CasADi function created
  ✅ Saved to models/power_surrogate_casadi.pkl

Validating export...
  Max absolute difference: 0.05 kW
  ✅ Validation passed!

✅ Export complete!
```

**Time:** ~10-30 seconds

---

## Step 4: Validate Export

Verify CasADi matches PyTorch:

```bash
python validate_normalization.py
```

**Expected output:**
```
======================================================================
Validating CasADi Export
======================================================================

Testing 3 cases...

Case 1: yaw=[0. 0. 0. 0.], wind=8.0m/s @ 270°
  PyTorch:     4.7839 MW
  CasADi:      4.7839 MW
  Error:       0.00 kW (0.000%)

✅ CasADi export is working correctly!
   Error is negligible (< 100 W)
```

**Time:** < 1 second

---

## Step 5: Run MPC Demo

Test the full MPC controller:

```bash
python nmpc_surrogate_casadi.py
```

**Expected output:**
```
======================================================================
Surrogate-Based Nonlinear MPC Demo (CasADi/ipopt)
======================================================================

Initializing controller...
  ✅ Surrogate model loaded

Building MPC...
  ✅ MPC ready

Running MPC...
  Initial yaw: [0. 0. 0. 0.]
  Wind: 8.0 m/s @ 270.0°

Step 0:
  Success: True
  Yaw: [15.2  8.3  3.1  0.0]  (optimal yaw angles found!)
  Power: 6.123 MW  (~28% gain vs baseline)
  Solve time: 87.3 ms
  Iterations: 42

✅ Demo complete!
```

**Time:** ~10 seconds (10 MPC steps)

---

## Quick Start (All Steps)

Run all steps in sequence:

```bash
cd ~/Documents/mpcrl/alternative_approach

# Step 1: Generate data (800 samples for quick test)
python scripts/generate_dataset_v2.py

# Step 2: Train model
python scripts/train_surrogate_v2.py

# Step 3: Export to l4casadi (FIXED v2 script)
python scripts/export_l4casadi_model_v2.py

# Step 4: Validate export
python validate_normalization.py

# Step 5: Run MPC
python nmpc_surrogate_casadi.py
```

**Total time:** ~5-10 minutes for quick test with 800 samples

---

## Troubleshooting

### Error: FileNotFoundError: 'data/surrogate_dataset.npz'
**Solution:** Run Step 1 (generate_dataset_v2.py) first

### Error: FileNotFoundError: 'checkpoints/power_surrogate_best.ckpt'
**Solution:** Run Step 2 (train_surrogate_v2.py) first

### Error: FileNotFoundError: 'models/power_surrogate_casadi.pkl'
**Solution:** Run Step 3 (export_l4casadi_model_v2.py) first

### CasADi predictions don't match PyTorch
**Solution:** Make sure you used **export_l4casadi_model_v2.py** (not v1!)

### MPC doesn't optimize (stays at [0, 0, 0, 0])
**Possible causes:**
1. Model not trained well (try more samples)
2. Export validation failed (check Step 4)
3. Cost function issue (check solver output)

---

## Performance Notes

### With 800 samples (Quick Test)
- ✅ **Fast:** ~5-10 minutes total
- ⚠️ **Limited accuracy:** Model may not generalize well
- ✅ **Good for:** Testing pipeline, debugging, initial validation
- ❌ **Not good for:** Production, comparing with other approaches

### With 10,000 samples (Recommended)
- ⏱️ **Slower:** ~30-40 minutes total
- ✅ **Better accuracy:** Model generalizes better
- ✅ **Good for:** Actual experiments, comparisons, paper results
- ✅ **Recommended for:** Final evaluation

### With 100,000 samples (Best)
- ⏱️ **Slowest:** ~5-6 hours total
- ✅ **Best accuracy:** Near-perfect surrogate
- ✅ **Good for:** Publication-quality results
- ⏳ **Consider:** Running overnight

---

## What You're Building

This pipeline creates an **MPC controller with learned surrogate model**:

1. **Dataset:** PyWake simulations → training data
2. **Model:** PyTorch neural network → power predictions
3. **Export:** l4casadi → CasADi-compatible function
4. **MPC:** CasADi/ipopt → optimal yaw angles

**Goal:** Real-time MPC (~100 ms) with accurate predictions (~99% match to PyWake)

---

## Next Steps After Pipeline Works

1. **Generate more data:** 10k-100k samples
2. **Retrain model:** Better accuracy
3. **Tune MPC parameters:** Horizon N, weights, etc.
4. **Compare approaches:** Surrogate MPC vs Hybrid vs Pure gradient
5. **WindGym integration:** Real environment testing

---

## Current Status

Based on your error, you're at:
- ❌ Step 1: Not done (no dataset)
- ❌ Step 2: Not done (no checkpoint)
- ❌ Step 3: Not done (no exported model)
- ❌ Step 4: Not done (can't validate without model)
- ❌ Step 5: Not done (can't run MPC without model)

**Start with:** `python scripts/generate_dataset_v2.py`

---

**TL;DR:** Run the 5 steps in order. Start now with Step 1! 🚀
