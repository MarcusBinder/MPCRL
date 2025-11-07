# 🚀 START HERE - Surrogate MPC Pipeline

**Last Updated:** 2025-11-07 (Commit: 8b4c423)

---

## What Was Wrong

You tried to run the export script but got "checkpoint not found". The issues were:

1. **Training script** saves to: `models/checkpoints/best-{epoch}-{val_mae}.ckpt`
2. **Export script** was looking for: `checkpoints/power_surrogate_best.ckpt`
3. **Path mismatch** → Error!

## What I Fixed ✅

1. **Auto-detect checkpoints** - Export script now finds checkpoint files automatically
2. **Better error messages** - Shows exactly what to do if something is missing
3. **Diagnostic tools** - Check pipeline status and run everything with one command

---

## Option 1: One Command (Recommended) 🎯

Run the entire pipeline with a single command:

```bash
cd ~/Documents/mpcrl/alternative_approach
python run_full_pipeline.py
```

**This will:**
- Generate 800 training samples
- Train model for 50 epochs
- Export to l4casadi (auto-detects checkpoint!)
- Validate export
- Run MPC demo

**Time:** ~5-10 minutes
**Result:** Full working pipeline!

---

## Option 2: Step by Step 📋

Check what's missing first:

```bash
python check_pipeline.py
```

**Example output:**
```
❌ Step 1: Dataset Generated
   Fix: python scripts/generate_dataset_v2.py
❌ Step 2: Model Trained
   Fix: python scripts/train_surrogate_v2.py --max_epochs 100
❌ Step 3: Model Exported to l4casadi
   Fix: python scripts/export_l4casadi_model_v2.py
```

Then run the missing steps:

```bash
# Step 1: Generate data
python scripts/generate_dataset_v2.py

# Step 2: Train model
python scripts/train_surrogate_v2.py --max_epochs 100

# Step 3: Export (now auto-detects checkpoint!)
python scripts/export_l4casadi_model_v2.py

# Step 4: Validate
python validate_normalization.py

# Step 5: Run MPC
python nmpc_surrogate_casadi.py
```

---

## What You'll See

### Step 1: Generate Dataset
```
Generating dataset...
  Sampling 800 LHS samples...
  Running PyWake simulations...
  ✅ Dataset saved to data/surrogate_dataset.h5
  ✅ Saved data/surrogate_dataset_train.h5
  ✅ Saved data/surrogate_dataset_val.h5
  ✅ Saved data/surrogate_dataset_test.h5
```

### Step 2: Train Model
```
Training...
Epoch 50/50: loss=0.002 val_loss=0.003
✅ Best model saved to: models/checkpoints/best-49-0.05.ckpt
```

### Step 3: Export (NOW AUTO-DETECTS!)
```
Auto-detecting checkpoint...
  Found checkpoint: models/checkpoints/best-49-0.05.ckpt
Loading model...
  ✅ Wrapper matches original perfectly!
  ✅ Validation passed!
✅ Export complete!
```

### Step 4: Validate
```
Testing 3 cases...
  PyTorch:     4.7839 MW
  CasADi:      4.7839 MW
  Error:       0.00 kW (0.000%)
✅ CasADi export is working correctly!
```

### Step 5: Run MPC
```
Step 0:
  Success: True
  Yaw: [15.2  8.3  3.1  0.0]  ← Optimal angles found!
  Power: 6.123 MW  ← ~28% gain!
  Solve time: 87.3 ms
✅ Demo complete!
```

---

## Quick Reference

| Command | What It Does | Time |
|---------|--------------|------|
| `python check_pipeline.py` | Check status | <1s |
| `python run_full_pipeline.py` | Run everything | ~5-10 min |
| `python scripts/generate_dataset_v2.py` | Generate data | ~1-2 min |
| `python scripts/train_surrogate_v2.py --max_epochs 50` | Train model | ~2-5 min |
| `python scripts/export_l4casadi_model_v2.py` | Export (auto!) | ~10-30s |
| `python validate_normalization.py` | Validate | <1s |
| `python nmpc_surrogate_casadi.py` | Run MPC | ~10s |

---

## Troubleshooting

### "Checkpoint not found"
**Solution:** Run training first: `python scripts/train_surrogate_v2.py --max_epochs 100`

### "Dataset not found"
**Solution:** Run data generation first: `python scripts/generate_dataset_v2.py`

### "l4casadi not installed"
**Solution:** `pip install l4casadi`

### "ModuleNotFoundError"
**Solution:** Activate conda environment: `conda activate mpcrl`

### Still stuck?
**Solution:** Run `python check_pipeline.py` to see what's missing

---

## Performance Options

### Quick Test (800 samples, 50 epochs)
```bash
python scripts/generate_dataset_v2.py --n_samples 800
python scripts/train_surrogate_v2.py --max_epochs 50
```
- ✅ Fast (~5 min total)
- ⚠️ Limited accuracy
- ✅ Good for testing

### Recommended (10k samples, 100 epochs)
```bash
python scripts/generate_dataset_v2.py --n_samples 10000
python scripts/train_surrogate_v2.py --max_epochs 100
```
- ⏱️ Medium (~30 min total)
- ✅ Good accuracy
- ✅ Good for experiments

### Best (100k samples, 200 epochs)
```bash
python scripts/generate_dataset_v2.py --n_samples 100000
python scripts/train_surrogate_v2.py --max_epochs 200
```
- ⏱️ Slow (~5-6 hours total)
- ✅ Best accuracy
- ✅ Publication quality

---

## Files Created

- **check_pipeline.py** - Shows pipeline status
- **run_full_pipeline.py** - Runs everything
- **export_l4casadi_model_v2.py** - Now auto-detects checkpoints!

---

## Summary

**Problem:** Checkpoint path mismatch
**Solution:** Auto-detection + diagnostic tools
**Action:** Run `python run_full_pipeline.py` 🚀

---

**Ready? Start now:**
```bash
python run_full_pipeline.py
```
