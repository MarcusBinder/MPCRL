# How to Run Surrogate-Based MPC

Complete guide to training and using the surrogate-based nonlinear MPC.

---

## Prerequisites

### 1. Install Dependencies

```bash
# Core dependencies (should already have these)
pip install numpy torch py_wake casadi

# New dependencies for surrogate approach
pip install l4casadi pytorch-lightning h5py tensorboard scikit-learn

# Optional: For GPU training (10x faster)
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Verify acados Installation

```bash
python -c "from acados_template import AcadosOcp; print('✅ acados OK')"
```

---

## Step-by-Step Guide

### Step 1: Generate Training Dataset (2-4 hours)

Generate 100,000 training samples by running PyWake simulations:

```bash
cd alternative_approach/

python scripts/generate_dataset_v2.py \
    --n_samples 100000 \
    --n_jobs 8 \
    --output data/surrogate_dataset.h5
```

**Options:**
- `--n_samples`: Number of samples (default: 100,000)
- `--n_jobs`: Parallel workers (default: CPU count - 1)
- `--output`: Output file path
- `--seed`: Random seed for reproducibility

**Expected output:**
- `data/surrogate_dataset.h5` - Full dataset
- `data/surrogate_dataset_train.h5` - Training set (80%)
- `data/surrogate_dataset_val.h5` - Validation set (10%)
- `data/surrogate_dataset_test.h5` - Test set (10%)
- `data/surrogate_dataset.json` - Metadata

**Time:** 2-4 hours on 8 cores

---

### Step 2: Train Surrogate Model (1-2 hours)

Train neural network to predict power:

```bash
python scripts/train_surrogate_v2.py \
    --train_dataset data/surrogate_dataset_train.h5 \
    --val_dataset data/surrogate_dataset_val.h5 \
    --test_dataset data/surrogate_dataset_test.h5 \
    --output_dir models \
    --batch_size 256 \
    --max_epochs 1000 \
    --gpus 0
```

**Options:**
- `--batch_size`: Batch size (default: 256)
- `--max_epochs`: Maximum epochs (default: 1000, with early stopping)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--gpus`: Number of GPUs (0 for CPU, 1 for single GPU)

**Expected output:**
- `models/power_surrogate.pth` - Trained model
- `models/training_config.json` - Training configuration
- `models/checkpoints/` - Checkpoint files
- `models/logs/` - TensorBoard logs

**Time:** 1-2 hours on GPU, 4-6 hours on CPU

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir models/logs
```

---

### Step 3: Export to l4casadi (< 1 minute)

Convert PyTorch model to CasADi format for acados:

```bash
python scripts/export_l4casadi_model.py \
    --model models/power_surrogate.pth \
    --output models/power_surrogate_casadi.pkl \
    --validate \
    --benchmark
```

**Options:**
- `--validate`: Run validation tests
- `--benchmark`: Benchmark performance

**Expected output:**
- `models/power_surrogate_casadi.pkl` - l4casadi model
- Validation results (accuracy vs PyTorch)
- Performance benchmark (evaluation time)

**Target metrics:**
- MAE < 1% of mean power
- Evaluation time < 1ms

**Time:** < 1 minute

---

### Step 4: Run Nonlinear MPC

Test the surrogate-based MPC:

```bash
python nmpc_surrogate.py
```

This runs a simple demo showing the MPC optimizing yaw angles using the surrogate model.

**Expected output:**
- MPC solve time: < 10ms per step
- Power gain: Should approach ~15% (close to optimal)
- Solver convergence: Should converge reliably

---

## Quick Test (if already have model)

If you've already trained the model, you can quickly test it:

```bash
# Just run the MPC demo
python nmpc_surrogate.py
```

---

## Validation and Benchmarking

### Validate Surrogate Accuracy

```bash
python tests/test_surrogate_accuracy.py \
    --model models/power_surrogate.pth \
    --test_dataset data/surrogate_dataset_test.h5
```

**Target:** MAE < 1%, R² > 0.99

### Test MPC Performance

```bash
python tests/test_surrogate_mpc_performance.py \
    --model models/power_surrogate_casadi.pkl
```

**Target:** Power gain > 13% (>85% of optimal 15.1%)

### Benchmark Solve Time

```bash
python tests/test_surrogate_mpc_speed.py \
    --model models/power_surrogate_casadi.pkl
```

**Target:** Solve time < 10ms per step

---

## Troubleshooting

### Problem: l4casadi import error

```bash
pip install l4casadi
```

If that fails, install from source:
```bash
git clone https://github.com/Tim-Salzmann/l4casadi.git
cd l4casadi
pip install -e .
```

### Problem: acados not found

Make sure acados is properly installed and environment variables are set.

### Problem: Dataset generation too slow

- Reduce `--n_samples` to 50,000 for faster prototyping
- Increase `--n_jobs` to use more CPU cores
- Use a machine with more cores

### Problem: Training too slow

- Use GPU: `--gpus 1`
- Reduce batch size if out of memory: `--batch_size 128`
- Start with smaller dataset for testing

### Problem: MPC solve time too high

- Reduce horizon length: `N_h=10` instead of `N_h=20`
- Increase solver tolerance: `tol=1e-4` instead of `1e-6`
- Reduce max iterations

### Problem: MPC not finding good solution

- Check surrogate accuracy first (validation script)
- Try warm starting from a good initial guess
- Increase solver iterations
- Check if solver is converging (status=0)

---

## Expected Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Dataset generation | 2-4 hours |
| 2 | Model training | 1-2 hours (GPU) |
| 3 | l4casadi export | < 1 min |
| 4 | MPC testing | < 1 min |
| **Total** | **First-time setup** | **3-6 hours** |

**After first setup:** Just run `python nmpc_surrogate.py` (< 1 min)

---

## File Structure

```
alternative_approach/
├── data/
│   ├── surrogate_dataset.h5           # Generated dataset
│   ├── surrogate_dataset_train.h5     # Training split
│   ├── surrogate_dataset_val.h5       # Validation split
│   └── surrogate_dataset_test.h5      # Test split
│
├── models/
│   ├── power_surrogate.pth            # Trained PyTorch model
│   ├── power_surrogate_casadi.pkl     # l4casadi converted model
│   ├── training_config.json           # Training config
│   └── logs/                          # TensorBoard logs
│
├── scripts/
│   ├── generate_dataset_v2.py         # Step 1
│   ├── train_surrogate_v2.py          # Step 2
│   └── export_l4casadi_model.py       # Step 3
│
├── nmpc_surrogate.py                  # Step 4 - Main MPC
│
└── tests/
    ├── test_surrogate_accuracy.py
    ├── test_surrogate_mpc_performance.py
    └── test_surrogate_mpc_speed.py
```

---

## Performance Targets

### Surrogate Model
- ✅ MAE < 50 kW (< 1% of 5 MW farm)
- ✅ R² > 0.99
- ✅ Gradient agreement > 95% with PyWake
- ✅ Evaluation time < 1ms

### MPC Performance
- ✅ Solve time < 10ms per step
- ✅ Power gain > 13% (>85% of optimal)
- ✅ Convergence rate > 95%
- ✅ Handles constraints reliably

---

## Next Steps After Success

1. **Multi-condition testing**: Test on various wind speeds/directions
2. **Closed-loop simulation**: Run MPC in simulation environment
3. **Compare with hybrid approach**: Benchmark vs lookup table method
4. **Add RL layer** (optional): Use RL to handle model mismatch
5. **Field deployment**: Test on real wind farm

---

## Resources

- **l4casadi**: https://github.com/Tim-Salzmann/l4casadi
- **acados**: https://docs.acados.org/
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/
- **Plan**: `SURROGATE_MPC_PLAN.md` - Detailed implementation plan

---

## Questions?

Check:
1. `SURROGATE_MPC_PLAN.md` - Detailed technical plan
2. `docs/MPC_ALTERNATIVES.md` - Analysis of different approaches
3. `docs/FINAL_RECOMMENDATIONS.md` - Why surrogate approach works

---

**Status:** ✅ All code ready to run
**Next Action:** Start with Step 1 (dataset generation)
