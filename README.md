# MPCRL - Learning-Enhanced Model Predictive Control for Wind Farms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid control system combining **Model Predictive Control (MPC)** and **Reinforcement Learning (RL)** for wind farm wake steering optimization. This project systematically investigates how to properly optimize MPC baselines before evaluating learning-based improvements.

## 🎯 Project Overview

Wind farms lose 10-40% of downstream turbine power due to wake effects. **Wake steering** - strategically yawing (rotating) upstream turbines to deflect wakes - can recover 10-20% of this lost power. However, optimizing wake steering is challenging due to:

- Complex spatio-temporal dynamics (60-120s wake propagation delays)
- Computational constraints (real-time decisions needed)
- Model uncertainty (physics models ≠ reality)

This project addresses these challenges through:

1. **Systematic MPC optimization** - Finding the best MPC parameters for speed vs quality
2. **Hybrid MPC-RL architecture** - Combining physics-based planning with data-driven learning
3. **Rigorous evaluation methodology** - Proper baselines, multi-seed testing, wake delay physics

## 🔬 Key Results

### MPC Baseline Optimization (Complete ✅)
- **100 configurations tested** across dt_opt, T_opt, maxfun parameters
- **300 total experiments** (3 random seeds per configuration)
- **10.8x speedup** with <1% quality loss vs reference
- **Critical discoveries**:
  - Evaluation horizon must be ≥200s to capture wake effects
  - Single-seed tests can be misleading (±21kW variance)
  - Parameter interactions matter (emergent sweet spots)

### RL Integration (In Progress 🚧)
- **Target**: +2-5% additional gain over optimized MPC baseline
- **Architecture**: MPC provides base policy, RL learns bounded corrections (±5°)
- **Expected total**: 11.8% (MPC) + 2.6% (RL) = ~14.4% gain vs greedy

## 📚 Documentation

**Start here:** [docs/PAPER_OUTLINE_V2.md](docs/PAPER_OUTLINE_V2.md) - Complete paper outline with methodology, experiments, and timeline

**Key findings:**
- [docs/WAKE_DELAY_FIX_SUMMARY.md](docs/WAKE_DELAY_FIX_SUMMARY.md) - Why evaluation horizon is critical
- [docs/SEED_BIAS_DISCOVERY.md](docs/SEED_BIAS_DISCOVERY.md) - Why single-seed tests can be misleading
- [docs/PERFORMANCE_OPTIMIZATION_GUIDE.md](docs/PERFORMANCE_OPTIMIZATION_GUIDE.md) - MPC parameter selection guide

**Full index:** [docs/README.md](docs/README.md)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/[username]/mpcrl.git
cd mpcrl

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

```
numpy >= 1.21
scipy >= 1.7
matplotlib >= 3.5
py_wake >= 2.5
gymnasium >= 0.28
torch >= 2.0
stable-baselines3 >= 2.0
pandas >= 1.5
```

### Basic Usage

#### MPC Optimization

```python
from mpcrl import WindFarmModel, optimize_farm_back2front
from py_wake.examples.data.hornsrev1 import V80
import numpy as np

# Setup wind farm
x_pos = np.array([0, 500, 1000])  # Turbine positions (m)
y_pos = np.array([0, 0, 0])
model = WindFarmModel(x_pos, y_pos, wt=V80(), D=80.0,
                      U_inf=8.0, TI=0.06, wd=270.0)

# Optimize yaw angles
initial_yaws = np.array([0.0, 0.0, 0.0])
optimized_params = optimize_farm_back2front(
    model, initial_yaws,
    dt_opt=30,    # Recommended: fast and high-quality
    T_opt=300,
    maxfun=10,
    r_gamma=0.3,
    t_AH=100.0
)

print(f"Optimized parameters: {optimized_params}")
```

#### RL Training

```python
from mpcrl import MPCenv, make_config
import gymnasium as gym

# Create environment with optimized MPC
env = MPCenv(
    x_pos=x_pos,
    y_pos=y_pos,
    turbine=V80(),
    mpc_dt_opt=30,    # Use optimized parameters
    mpc_T_opt=300,
    mpc_maxfun=10
)

# Train RL agent (example with SAC)
# See sac_MPC_local.py for full training script
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Replace with RL policy
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()
```

## 📊 Running Experiments

### MPC Parameter Investigation

```bash
# Quick test (27 configs, ~5 minutes)
python tests/test_optimization_quality_quick.py

# Comprehensive test (100 configs × 3 seeds, ~20 minutes)
python tests/test_optimization_quality.py

# Wake delay analysis (generates paper figure)
python tests/test_wake_delay_analysis.py

# Seed sensitivity study
python tests/test_maxfun_investigation.py
```

### RL Training

```bash
# Train hybrid MPC-RL agent
python sac_MPC_local.py --num_envs 6 --total_timesteps 100000

# With Weights & Biases tracking
python sac_MPC_local.py --track --wandb_project_name MPC_RL
```

## 📁 Repository Structure

```
mpcrl/
├── mpcrl/                      # Core library
│   ├── mpc.py                  # MPC implementation
│   ├── environment.py          # RL environment
│   ├── environment_fast.py     # Optimized environment
│   └── config.py               # Configuration utilities
│
├── tests/                      # Test scripts for paper results
│   ├── test_optimization_quality.py           # Main parameter sweep
│   ├── test_optimization_quality_quick.py     # Quick validation
│   ├── test_wake_delay_analysis.py            # Wake delay study
│   ├── test_maxfun_investigation.py           # Seed sensitivity
│   ├── test_single_scenario_debug.py          # Detailed diagnostics
│   └── test_wake_steering_benefit.py          # Wind condition study
│
├── docs/                       # Documentation
│   ├── README.md               # Documentation index
│   ├── PAPER_OUTLINE_V2.md     # Complete paper outline
│   ├── WAKE_DELAY_FIX_SUMMARY.md
│   ├── SEED_BIAS_DISCOVERY.md
│   └── PERFORMANCE_OPTIMIZATION_GUIDE.md
│
├── results/                    # Experimental results
│   ├── data/                   # CSV results
│   └── figures/                # Generated figures (PDF/PNG)
│
├── examples/                   # Usage examples
│   ├── 01_environment_setup.ipynb
│   ├── 02_mpc_optimization.ipynb
│   └── 03_training_loop.ipynb
│
├── scripts/                    # Utility scripts
│   └── profile_mpc_performance.py
│
├── sac_MPC_local.py           # Main RL training script
├── setup.py                    # Package installation
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔑 Key Concepts

### MPC Parameterization

**Trajectory basis functions** reduce dimension from K×N to 2×N parameters:

```python
γ_i(t) = γ_i(0) + Δγ_i · ψ(t/T_AH; o1_i, o2_i)

where ψ(s; o1, o2) = (1 - cos(πs^{o1}))^{o2} / 2
```

**Back-to-front optimization** sequentially optimizes from downstream to upstream, accounting for wake delays.

**Time-shifted cost function** properly models delayed wake effects:
```python
J = Σ_k Σ_i P_i(γ_1(k-τ_i1), ..., γ_N(k-τ_iN))
```

### Critical Evaluation Requirements

**Wake delay physics** requires:
```
T_eval ≥ max_delay + T_AH + safety_margin

Example: 3 turbines, 500m spacing, 8 m/s wind
  max_delay = 125s
  T_AH = 100s
  → Need T_eval ≥ 225s minimum
  → We use 1000s for safety
```

**Multi-seed averaging** required for stochastic optimization:
- Low maxfun: ±21kW variance across seeds
- High maxfun: ±9kW variance
- Recommendation: Average ≥3 seeds

### Hybrid MPC-RL Architecture

```python
yaw_final = yaw_mpc + rl_correction

where:
  yaw_mpc:       Physics-based MPC optimization
  rl_correction: Learned adjustments (bounded ±5°)
```

**Design principles:**
- MPC provides foundation (physics-aware)
- RL learns corrections (data-driven)
- Bounded actions for safety
- Decomposable gains (attribution)

## 🧪 Recommended MPC Parameters

Based on 300 experiments:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `dt_opt` | 30s | Coarse discretization simplifies problem |
| `T_opt` | 300s | Medium horizon balances planning vs overfitting |
| `maxfun` | 10 | Sufficient for simple problem, 10x faster |
| **Result** | **0.32s** | **99.78% quality, 10.8x speedup** |

**Alternatives:**
- **Maximum quality**: dt=10, T=500, maxfun=50 (3.4s, 100% quality)
- **Maximum speed**: dt=30, T=200, maxfun=10 (0.29s, 99.6% quality)

## 📈 Performance

### Computational Cost

```
MPC optimization: ~0.32s per call (optimized config)
RL inference:     ~0.01s per call
Environment step: ~0.05s overhead

Total per step:   ~0.4s (6 parallel environments)
```

### Training Time Estimates

```
100k RL steps:
  Steps per env: 100k / 6 = 16,667
  Time per step: 0.5s
  Total time:    ~2.3 hours ✅ (vs ~14 hours with slow MPC)
```

### Scalability

Tested on:
- 3-turbine farm (current)
- Target: 5-20 turbines (future work)
- Computational cost scales as O(N) with back-to-front approach

## 🤝 Contributing

Contributions welcome! Areas of interest:

- **Larger farms**: Test on 10+ turbine layouts
- **Field validation**: Real wind farm data integration
- **Advanced RL**: Meta-learning, multi-agent approaches
- **Forecasting**: Integration with weather predictions

Please see [docs/PAPER_OUTLINE_V2.md](docs/PAPER_OUTLINE_V2.md) Section 7.4.2 for future research directions.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{mpcrl2025,
  title={Learning-Enhanced Model Predictive Control for Wind Farm Wake Steering},
  author={[Your Name]},
  journal={[Venue - In Preparation]},
  year={2025},
  note={Code available at https://github.com/[username]/mpcrl}
}
```

## 📝 License

[Add your license - e.g., MIT License]

## 🙏 Acknowledgments

- **PyWake** for wake modeling framework
- **Stable-Baselines3** for RL implementations
- **CleanRL** for training code inspiration
- [Add funding sources, collaborators, etc.]

## 📧 Contact

- **Author**: [Your Name]
- **Email**: [Your Email]
- **Project Link**: https://github.com/[username]/mpcrl

---

## 🗺️ Project Status & Roadmap

### Phase 1: MPC Baseline Optimization ✅ **COMPLETE**
- [x] Implement parameterized MPC
- [x] Systematic parameter sweep (100 configs)
- [x] Wake delay analysis
- [x] Seed sensitivity study
- [x] Documentation and guides

### Phase 2: RL Integration 🚧 **IN PROGRESS**
- [ ] Implement hybrid MPC-RL environment
- [ ] Train SAC agents (greedy, MPC, pure RL, hybrid)
- [ ] Comprehensive evaluation across wind conditions
- [ ] Ablation studies

### Phase 3: Paper Writing 📝 **PLANNED**
- [ ] Write methodology sections (Sections 3-4)
- [ ] Generate all paper figures
- [ ] Results and analysis (Sections 6-7)
- [ ] Abstract, introduction, conclusions

### Phase 4: Submission 🎯 **TARGET: 16 weeks**
- [ ] Internal review and revision
- [ ] Select target venue
- [ ] Submit manuscript

See [docs/PAPER_OUTLINE_V2.md](docs/PAPER_OUTLINE_V2.md) for detailed timeline.

---

**Last Updated**: 2025-10-17
**Version**: 1.0.0-alpha
**Status**: Active development
