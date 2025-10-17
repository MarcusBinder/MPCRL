# Project Organization

**Last Updated**: 2025-10-17

This document describes the cleaned-up repository structure and organization.

## 📁 Directory Structure

```
mpcrl/
│
├── mpcrl/                      # Core Python package
│   ├── __init__.py
│   ├── mpc.py                  # MPC implementation (WindFarmModel, optimization)
│   ├── environment.py          # RL environment (MPCenv)
│   ├── environment_fast.py     # Optimized environment variant
│   └── config.py               # Configuration utilities
│
├── tests/                      # Test scripts (generate paper results)
│   ├── __init__.py
│   ├── test_mpc.py             # Basic MPC functionality tests
│   ├── test_optimization_quality.py              # ⭐ Main: 100 configs × 3 seeds
│   ├── test_optimization_quality_quick.py        # Quick: 27 configs
│   ├── test_wake_delay_analysis.py               # ⭐ Paper Figure 1
│   ├── test_maxfun_investigation.py              # ⭐ Seed sensitivity study
│   ├── test_single_scenario_debug.py             # Detailed single-case analysis
│   ├── test_wake_steering_benefit.py             # Wind condition sensitivity
│   └── test_nan_handling.py                      # Edge case validation
│
├── docs/                       # Documentation
│   ├── README.md               # Documentation index
│   ├── PAPER_OUTLINE_V2.md     # ⭐ MAIN: Complete paper outline (active)
│   ├── WAKE_DELAY_FIX_SUMMARY.md       # Critical finding #1
│   ├── SEED_BIAS_DISCOVERY.md          # Critical finding #2
│   ├── PERFORMANCE_OPTIMIZATION_GUIDE.md
│   └── PAPER_OUTLINE_V1_ARCHIVED.md    # (superseded by V2)
│
├── results/                    # Experimental results
│   ├── data/                   # CSV data files
│   └── figures/                # Generated figures (PDF/PNG)
│
├── scripts/                    # Utility scripts
│   └── profile_mpc_performance.py      # Performance profiling
│
├── examples/                   # Usage examples and tutorials
│   ├── README.md
│   ├── 01_environment_setup.ipynb
│   ├── 02_mpc_optimization.ipynb
│   └── 03_training_loop.ipynb
│
├── data/                       # Data directory
│   └── README.md               # (Wind field data, if needed)
│
├── sac_MPC_local.py           # ⭐ Main RL training script
├── setup.py                    # Package installation config
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # ⭐ Main project README
└── PROJECT_ORGANIZATION.md     # This file
```

## 🌟 Key Files

### Must-Read First
1. **README.md** - Project overview, quick start, key results
2. **docs/PAPER_OUTLINE_V2.md** - Complete paper plan and roadmap
3. **docs/README.md** - Documentation index

### Critical Findings
4. **docs/WAKE_DELAY_FIX_SUMMARY.md** - Why T_eval ≥ 200s is essential
5. **docs/SEED_BIAS_DISCOVERY.md** - Why multi-seed testing is required

### Implementation
6. **mpcrl/mpc.py** - Core MPC implementation (1000+ lines)
7. **mpcrl/environment.py** - RL environment integration
8. **sac_MPC_local.py** - Main training script (SAC algorithm)

### Testing
9. **tests/test_optimization_quality.py** - Main parameter sweep
10. **tests/test_wake_delay_analysis.py** - Wake delay paper figure

## 📊 File Status

### Complete ✅
- [x] Core MPC implementation
- [x] RL environment wrapper
- [x] Main test scripts (5/7)
- [x] Documentation (wake delay, seed bias, performance)
- [x] Paper outline V2

### In Progress 🚧
- [ ] RL training and evaluation
- [ ] Additional test scripts (heatmaps, variance analysis)
- [ ] Paper figure generation

### Planned 📝
- [ ] Trained model checkpoints
- [ ] Full experimental results
- [ ] Paper manuscript (LaTeX)

## 🗂️ File Purposes

### Core Package (`mpcrl/`)

**mpc.py** (Primary implementation)
- `WindFarmModel`: Physics-based wind farm simulation
- `optimize_farm_back2front()`: Sequential MPC optimization
- `yaw_traj()`, `psi()`: Trajectory basis functions
- `YawCache`: LRU cache with quantization
- `run_farm_delay_loop_optimized()`: Simulation with wake delays

**environment.py** (RL integration)
- `MPCenv`: Gymnasium environment
- `make_config()`: Configuration builder
- Observation/action space definitions
- Reward function

**environment_fast.py** (Optimized variant)
- `MPCenvFast`: Same as MPCenv but with recommended parameters
- Pre-configured for fast training (dt=30, T=300, maxfun=10)

**config.py**
- Configuration utilities
- Default parameter values

### Test Scripts (`tests/`)

**test_optimization_quality.py** ⭐ PRIMARY
- Tests 100 configurations (5 dt × 4 T × 5 maxfun)
- Averages over 3 random seeds per config
- Identifies Pareto frontier
- Generates comprehensive analysis
- **Output**: CSV data, recommendations
- **Runtime**: ~20 minutes

**test_optimization_quality_quick.py**
- Reduced to 27 configurations for fast validation
- Same methodology as full test
- **Runtime**: ~5 minutes

**test_wake_delay_analysis.py** ⭐ PAPER FIGURE
- Varies evaluation horizon from 50s to 1000s
- Shows impact on measured performance
- **Output**: Figure 1 for paper (PDF/PNG)
- **Runtime**: ~5 minutes

**test_maxfun_investigation.py** ⭐ SEED STUDY
- Tests multiple random seeds per configuration
- Demonstrates seed sensitivity
- Compares low vs high maxfun variance
- **Output**: Box plots, variance analysis
- **Runtime**: ~10 minutes

**test_single_scenario_debug.py**
- Detailed analysis of one scenario
- Per-turbine power breakdown
- Yaw trajectory visualization
- **Output**: Diagnostic plots
- **Use**: Debugging and understanding

**test_wake_steering_benefit.py**
- Tests multiple wind conditions
- Shows when wake steering helps vs hurts
- **Output**: Sensitivity analysis
- **Use**: Understanding applicability

**test_nan_handling.py**
- Edge case validation
- NaN detection and handling
- **Output**: Test results
- **Use**: Code validation

### Documentation (`docs/`)

**PAPER_OUTLINE_V2.md** ⭐ ACTIVE
- Complete paper structure (9 sections)
- Balanced MPC-RL narrative
- Detailed experimental plan
- 16-week implementation roadmap
- Expected figures and tables
- **Status**: Current, use this one

**WAKE_DELAY_FIX_SUMMARY.md**
- Documents wake delay bug and fix
- Explains evaluation horizon requirement
- Lists all affected files
- **Status**: Historical record, critical finding

**SEED_BIAS_DISCOVERY.md**
- Documents single-seed bias discovery
- Shows why multi-seed averaging is needed
- Provides corrected methodology
- **Status**: Historical record, critical finding

**PERFORMANCE_OPTIMIZATION_GUIDE.md**
- Comprehensive optimization guide
- Parameter selection rationale
- Caching strategies
- **Status**: Reference guide

**PAPER_OUTLINE_V1_ARCHIVED.md**
- Original MPC-heavy outline
- **Status**: Archived, superseded by V2

### Scripts (`scripts/`)

**profile_mpc_performance.py**
- Benchmarks MPC performance
- Tests different parameter combinations
- Measures cache hit rates
- **Output**: Timing analysis
- **Use**: Performance validation

### Main Training (`sac_MPC_local.py`)
- Soft Actor-Critic (SAC) RL algorithm
- Integrates with MPCenv
- Configurable hyperparameters
- Weights & Biases tracking
- Model checkpointing
- **Use**: Train RL agents

## 🔄 Workflow

### Phase 1: MPC Optimization (✅ Done)
```bash
# Run parameter sweep
python tests/test_optimization_quality.py

# Generate wake delay figure
python tests/test_wake_delay_analysis.py

# Validate seed sensitivity
python tests/test_maxfun_investigation.py
```

### Phase 2: RL Training (🚧 In Progress)
```bash
# Train hybrid MPC-RL agent
python sac_MPC_local.py --num_envs 6 --total_timesteps 100000 --track
```

### Phase 3: Analysis (📝 Planned)
```bash
# Generate all paper figures
python tests/generate_all_paper_figures.py  # (to be created)

# Create LaTeX tables
python tests/generate_paper_tables.py  # (to be created)
```

## 📦 Data Management

### Input Data
- **Wind fields**: `*.nc` files (NetCDF format) - gitignored
- **Turbine specs**: From PyWake library (V80, DTU10MW)
- **Configuration**: `mpcrl/config.py`

### Output Data
- **CSV results**: `results/data/*.csv` - tracked in git
- **Figures**: `results/figures/*.{pdf,png}` - tracked in git
- **Model checkpoints**: `runs/` - gitignored
- **Training logs**: `wandb/` - gitignored

### Size Guidelines
- Track small CSVs (<1MB)
- Track figures (<5MB)
- Gitignore large data files, model checkpoints, logs

## 🧹 Cleanup Actions Taken

### Files Moved
- ✅ `PAPER_OUTLINE_V2.md` → `docs/`
- ✅ `WAKE_DELAY_FIX_SUMMARY.md` → `docs/`
- ✅ `SEED_BIAS_DISCOVERY.md` → `docs/`
- ✅ `PERFORMANCE_OPTIMIZATION_GUIDE.md` → `docs/`
- ✅ `PAPER_OUTLINE.md` → `docs/PAPER_OUTLINE_V1_ARCHIVED.md`
- ✅ `profile_mpc_performance.py` → `scripts/`

### Directories Created
- ✅ `docs/` - Centralized documentation
- ✅ `results/data/` - CSV data files
- ✅ `results/figures/` - Generated figures
- ✅ `scripts/` - Utility scripts

### Files Created
- ✅ `docs/README.md` - Documentation index
- ✅ `README.md` - Updated main README
- ✅ `.gitignore` - Enhanced with project-specific entries
- ✅ `PROJECT_ORGANIZATION.md` - This file

## 🎯 Next Steps

1. **Complete remaining test scripts**
   - [ ] `test_parameter_heatmaps.py`
   - [ ] `test_variance_analysis.py`
   - [ ] `test_wind_sensitivity.py`

2. **Run RL training**
   - [ ] Train baseline policies (greedy, MPC, pure RL)
   - [ ] Train hybrid MPC-RL
   - [ ] Evaluate and compare

3. **Generate paper materials**
   - [ ] All figures (8-10 total)
   - [ ] All tables (5-8 total)
   - [ ] Statistical analyses

4. **Write manuscript**
   - [ ] Sections 3-4 (methodology) - can start now
   - [ ] Sections 5-7 (RL + results) - after training
   - [ ] Sections 1-2, 8 (intro, conclusion) - final

## 📝 Maintenance

### Adding New Files
- **Test scripts**: Place in `tests/`
- **Documentation**: Place in `docs/`, update `docs/README.md`
- **Figures**: Auto-save to `results/figures/`
- **Data**: Auto-save to `results/data/`

### File Naming Conventions
- **Test scripts**: `test_*.py`
- **Documentation**: `UPPERCASE_WITH_UNDERSCORES.md`
- **Figures**: `fig_descriptive_name.{pdf,png}`
- **Data**: `descriptive_name.csv`

### Version Control
- **Commit often**: Small, focused commits
- **Clear messages**: Describe what changed and why
- **Tag milestones**: `v1.0.0-phase1-complete`, etc.
- **Branch strategy**: `main` for stable, `dev` for work in progress

---

**Questions?** See [docs/README.md](docs/README.md) for documentation index or README.md for project overview.
