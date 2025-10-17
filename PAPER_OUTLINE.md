# Learning-Enhanced Model Predictive Control for Wind Farm Wake Steering

## Paper Outline and Implementation Plan

---

## ABSTRACT (To be written last)

**Key points to include:**
- Wind farm wake steering can increase total power production
- MPC provides physics-aware optimization but requires tuning
- RL can learn to improve MPC decisions over time
- We systematically investigate MPC parameter trade-offs for RL training
- Novel contributions: wake delay analysis, parameter optimization study, multi-seed robustness testing
- Results: X% power gain with Y% computational speedup

---

## 1. INTRODUCTION

### 1.1 Motivation
- Wind farms: wake effects reduce downstream turbine power by 10-40%
- Wake steering: intentional yaw misalignment to deflect wakes
- Challenge: complex spatio-temporal dynamics with propagation delays
- Real-time optimization needed for varying wind conditions

### 1.2 Problem Statement
- **Control challenge**: Optimize yaw angles for N turbines in real-time
- **Temporal complexity**: Wake changes take 60-120s to propagate through farm
- **Computational constraint**: Decisions needed every 30-60s
- **Uncertainty**: Wind conditions vary, models are imperfect

### 1.3 Proposed Approach
- **Hybrid architecture**: MPC provides base policy, RL learns improvements
- **Parameterized trajectories**: Reduce optimization dimension from 50+ to 2N parameters
- **Back-to-front optimization**: Sequential turbine optimization accounting for delays
- **Systematic investigation**: Identify optimal MPC parameters for RL training

### 1.4 Contributions
1. **Wake delay physics analysis**: Quantify impact of evaluation horizon on performance metrics
2. **Parameter optimization study**: Systematic sweep of dt_opt, T_opt, maxfun with multi-seed robustness
3. **Pareto frontier identification**: Speed vs quality trade-offs for real-time control
4. **Open-source implementation**: Complete codebase for reproducibility

### 1.5 Paper Organization
- Section 2: Background and related work
- Section 3: Problem formulation and methodology
- Section 4: MPC parameterization and optimization
- Section 5: Experimental setup and parameter investigation
- Section 6: Results and analysis
- Section 7: RL integration and learning results (future work)
- Section 8: Conclusions

---

## 2. BACKGROUND AND RELATED WORK

### 2.1 Wind Farm Wake Modeling
- **Engineering wake models**: Jensen, Gaussian, Bastankhah-Porté-Agel
- **PyWake framework**: Blondel-Cathelain 2020 superposition
- **Validation studies**: Comparison with LES and field data
- **Wake propagation delays**: τ = d/U_∞ where d is turbine spacing

**References to include:**
- PyWake documentation
- Wake model validation papers
- Field test data (Hornsrev, WFFC project, etc.)

### 2.2 Wake Steering Control Strategies

#### 2.2.1 Static Optimization
- Lookup tables based on (WS, WD, TI)
- Pros: Fast, reliable
- Cons: No adaptation to transients or model errors

#### 2.2.2 Model Predictive Control
- **Key papers**:
  - Shapiro et al. (2017, 2021): Parameterized MPC
  - Goit & Meyers (2015): Adjoint-based optimization
  - Gebraad et al. (2016): FLORIS-based MPC
- **Challenges**: Computational cost, model uncertainty

#### 2.2.3 Reinforcement Learning
- **Recent work**:
  - Stanfel et al. (2020): RL for wake steering
  - Dong et al. (2022): Multi-agent RL
  - Zhang et al. (2023): Deep RL with FLORIS
- **Challenges**: Sample efficiency, safety, interpretability

#### 2.2.4 Hybrid Approaches
- **Gaps in literature**:
  - Limited work on MPC+RL combination
  - No systematic study of MPC parameter impact on RL training
  - Lack of robustness analysis (sensitivity to random seeds)

### 2.3 Position of This Work
- **Novel contribution**: Systematic investigation of MPC parameterization for RL training
- **Methodological rigor**: Multi-seed averaging, wake delay physics analysis
- **Practical focus**: Real-time computational constraints

---

## 3. PROBLEM FORMULATION

### 3.1 Wind Farm Model

#### 3.1.1 Farm Layout
```
Turbine positions: (x_i, y_i) for i = 1...N
Rotor diameter: D = 80m (Vestas V80)
Hub height: H = 70m
```

#### 3.1.2 Wake Model
- **Model**: Blondel-Cathelain 2020 (PyWake)
- **Wake superposition**: Linear sum
- **Turbulence model**: Crespo-Hernandez
- **Inputs**: (γ, U_∞, θ, TI) → turbine power P_i

#### 3.1.3 Wake Propagation Delays
```
τ_ij = ||r_j - r_i|| / U_∞   (delay from turbine i to j)
```

**Critical insight**: Evaluation horizon must satisfy:
```
T_eval ≥ max_ij(τ_ij) + T_AH + safety_margin
```

### 3.2 Optimization Problem

#### 3.2.1 Continuous-Time Formulation
```
maximize ∫_0^T Σ_i P_i(γ_i(t), u_i^wake(t)) dt

subject to:
  |dγ_i/dt| ≤ γ_max                    (yaw rate limit)
  γ_min ≤ γ_i(t) ≤ γ_max                (yaw bounds)
  u_i^wake(t) = f(γ_j(t - τ_ji))       (wake coupling with delay)
```

#### 3.2.2 Discrete-Time Approximation
- Time discretization: dt_opt
- Horizon length: T_opt
- Decision points: K = T_opt / dt_opt

**Challenge**: High-dimensional (K × N parameters)

### 3.3 Trajectory Parameterization

**Key idea**: Reduce dimension from K×N to 2×N parameters

#### 3.3.1 Basis Function Representation
For turbine i with current yaw γ_i(0):
```
γ_i(t) = γ_i(0) + Δγ_i · ψ(t/T_AH; o1_i, o2_i)
```

Where ψ is the trajectory basis function:
```
ψ(s; o1, o2) = {
  0,                           if s ≤ 0
  (1 - cos(πs^o1))^o2 / 2,    if 0 < s ≤ 1
  1,                           if s > 1
}
```

**Parameters per turbine**: (o1, o2) ∈ [0,1]²

#### 3.3.2 Properties
- **Smooth**: C^0 continuous (C^1 if o1>0)
- **Monotonic**: Guaranteed when o1, o2 ∈ [0,1]
- **Flexible**: Can represent fast/slow, linear/curved transitions
- **Low-dimensional**: 2N parameters instead of K×N

### 3.4 Back-to-Front Optimization

**Algorithm**:
```
1. Sort turbines by upstream → downstream order
2. For i = N down to 1:
     a. Fix parameters for turbines i+1...N
     b. Optimize (o1_i, o2_i) for turbine i
     c. Use time-shifted cost accounting for delays
3. Return optimized parameters for all turbines
```

**Rationale**: Downstream turbines have less influence on others, optimize them first

---

## 4. MPC IMPLEMENTATION AND PARAMETERIZATION

### 4.1 Cost Function Design

#### 4.1.1 Standard Cost (No Delays)
```
J_standard = Σ_k Σ_i P_i(γ_1(k),...,γ_N(k))
```

**Problem**: Ignores propagation delays, optimistic about wake steering benefits

#### 4.1.2 Time-Shifted Cost (With Delays)
```
J_delayed = Σ_k Σ_i P_i(γ_1(k - τ_i1),...,γ_N(k - τ_iN))
```

**Benefit**: Accounts for when wake effects actually arrive at each turbine

### 4.2 Optimization Method

#### 4.2.1 Solver: Dual Annealing
- **Type**: Global optimization with local refinement
- **Pros**: Handles non-convex landscapes, parallel evaluations
- **Cons**: Stochastic (seed-dependent results)

#### 4.2.2 Key Parameters
```
maxfun:  Maximum function evaluations per turbine
         Higher → better convergence, slower
         Lower → faster, more variance
```

### 4.3 Computational Efficiency

#### 4.3.1 Caching Strategy
```
LRU cache with quantization:
- Cache size: 10,000 entries
- Wind quantization: 0.25 m/s bins
- Yaw quantization: 0.25° bins
```

**Typical hit rate**: 70-80%

#### 4.3.2 Evaluation Complexity
```
Per MPC call:
  - N turbines × maxfun evaluations × (1 - cache_hit_rate) PyWake calls
  - Typical: 3 × 20 × 0.3 = 18 PyWake calls per MPC optimization
```

### 4.4 Critical Parameter Investigation

**Research question**: How do MPC parameters affect speed vs quality trade-off?

#### 4.4.1 Parameters Under Study
```
dt_opt:    Optimization time step [10, 15, 20, 25, 30] s
T_opt:     Optimization horizon [200, 300, 400, 500] s
maxfun:    Max function evals [10, 15, 20, 30, 50]
```

**Total combinations**: 5 × 4 × 5 = 100 configurations

#### 4.4.2 Fixed Evaluation Parameters
```
eval_horizon = 1000s  (CRITICAL: must capture full wake propagation!)
T_AH = 100s           (action horizon)
r_gamma = 0.3         (yaw rate limit: 30°/100s)
```

#### 4.4.3 Multi-Seed Robustness
**Critical discovery**: Single seed creates bias!

**Our approach**:
- Test each configuration with n_seeds = 3 different random seeds
- Report mean ± std across seeds
- Ensures results aren't "lucky draws"

---

## 5. EXPERIMENTAL SETUP

### 5.1 Test Farm Configuration

#### 5.1.1 Layout
```
Turbines: 3-turbine aligned array
Positions: x = [0, 500, 1000]m, y = [0, 0, 0]m
Spacing: 6.25D (500m / 80m)
```

#### 5.1.2 Wind Conditions
```
Base scenario:
  Wind speed:    8.0 m/s
  Wind direction: 270° (perfect alignment)
  Turbulence:    0.06 (6%)

Additional tests:
  WS: [8, 10, 12] m/s
  WD: [270, 275, 280]°
  TI: [0.03, 0.06, 0.10]
```

#### 5.1.3 Wake Delays
```
Turbine 0 → 1: 500m / 8m/s = 62.5s
Turbine 1 → 2: 500m / 8m/s = 62.5s
Turbine 0 → 2: 1000m / 8m/s = 125s

→ Need eval_horizon ≥ 200s minimum!
```

### 5.2 Baseline Comparisons

#### 5.2.1 Greedy Baseline
```
Strategy: Hold all yaws at 0° (no wake steering)
Implementation: o1 = o2 = 0.5 for all turbines → Δγ = 0
Evaluation: Run delayed simulation over 1000s
```

#### 5.2.2 Reference Solution
```
Configuration: dt_opt=10, T_opt=500, maxfun=100
Purpose: "Best achievable" quality benchmark
Evaluation: Average over 3 seeds for robustness
```

### 5.3 Performance Metrics

#### 5.3.1 Quality Metrics
```
avg_power:           Total farm power averaged over T_eval
gain_vs_baseline:    (avg_power - baseline) / baseline × 100%
quality_vs_reference: avg_power / reference_power × 100%
```

#### 5.3.2 Speed Metrics
```
optimization_time:  Wall-clock time for MPC optimization
speedup:           reference_time / optimization_time
```

#### 5.3.3 Robustness Metrics
```
avg_power_std:  Standard deviation across random seeds
variance_pct:   (std / mean) × 100%
```

### 5.4 Test Scripts Organization

**Directory structure**:
```
tests/
├── test_optimization_quality.py           # Main parameter sweep (100 configs × 3 seeds)
├── test_optimization_quality_quick.py     # Quick test (27 configs)
├── test_single_scenario_debug.py          # Detailed single scenario analysis
├── test_wake_steering_benefit.py          # Multiple wind conditions
├── test_maxfun_investigation.py           # Seed sensitivity analysis
├── test_nan_handling.py                   # Edge case validation
└── test_wake_delay_analysis.py            # TO BE CREATED: Wake delay impact study
```

---

## 6. RESULTS AND ANALYSIS

### 6.1 Wake Delay Impact Study

**Critical Finding**: Evaluation horizon critically affects measured performance

#### 6.1.1 Experimental Design
```
Test same configuration with varying eval_horizon:
  [50, 100, 150, 200, 300, 500, 1000]s

Expected result:
  - Short horizons underestimate wake steering benefit
  - Benefit plateaus once horizon > max_delay + T_AH
```

**Test script**: `test_wake_delay_analysis.py` (TO BE CREATED)

#### 6.1.2 Expected Figure
```
Plot: Measured power gain vs evaluation horizon
  - X-axis: Evaluation horizon (s)
  - Y-axis: Power gain vs greedy (%)
  - Vertical lines marking wake arrival times
  - Plateau around 200-300s
```

### 6.2 Parameter Sweep Results

#### 6.2.1 Overall Performance Distribution
**Results from test_optimization_quality.py**:
```
Best configuration:  dt_opt=30, T_opt=300, maxfun=10
  Quality: 99.78% of reference
  Speed:   10.8x faster (0.32s vs 3.4s)
  Robust:  Low variance across seeds

Reference: dt_opt=10, T_opt=400, maxfun=50
  Quality: 100% (by definition)
  Speed:   1.0x baseline
```

#### 6.2.2 Pareto Frontier Analysis
**Expected Figure**:
```
Scatter plot: Optimization time vs Average power
  - X-axis: MPC optimization time (s)
  - Y-axis: Average farm power (W)
  - Color by maxfun value
  - Highlight Pareto-optimal points
```

**Key insights**:
- Only 2 configurations are Pareto optimal
- Clear speed/quality tradeoff curve
- Diminishing returns above maxfun=20

#### 6.2.3 Impact of Individual Parameters

**Figure 1: Impact of maxfun** (fixing dt_opt=20, T_opt=300)
```
Plot: Time vs Power for different maxfun values
Expected trend: Higher maxfun → better quality, slower speed
```

**Figure 2: Impact of T_opt** (fixing dt_opt=20, maxfun=20)
```
Plot: Time vs Power for different T_opt values
Expected trend: Longer horizon → slight quality improvement, slower
```

**Figure 3: Impact of dt_opt** (fixing T_opt=300, maxfun=20)
```
Plot: Time vs Power for different dt_opt values
Expected trend: Coarser dt → faster, quality depends on interaction with T_opt
```

**Figure 4: Heatmap** (fixing maxfun=20)
```
Heatmap: dt_opt (x-axis) vs T_opt (y-axis)
Color: Quality (% of reference)
Pattern: "Sweet spot" around dt_opt=20-30, T_opt=300
```

#### 6.2.4 Variance Analysis

**Critical Finding**: Lower maxfun has higher variance

**Expected Figure**:
```
Box plot: Power distribution across 3 seeds for each maxfun value
  - X-axis: maxfun [10, 15, 20, 30, 50]
  - Y-axis: Average power (W)
  - Box shows quartiles, whiskers show range

Expected pattern:
  - maxfun=10: Wide boxes (high variance)
  - maxfun=50: Narrow boxes (low variance)
```

**Test script**: Extend `test_maxfun_investigation.py`

### 6.3 Wind Condition Sensitivity

**Results from test_wake_steering_benefit.py**:

#### 6.3.1 When Does Wake Steering Help?
```
Test matrix:
  WS: [8, 10, 12] m/s
  WD: [270, 275, 280]°
  TI: [0.03, 0.06, 0.10]
```

**Expected patterns**:
- **Low TI** → stronger wakes → more benefit from steering
- **High WS** → more power at stake → larger absolute gains
- **Misaligned WD** → partial wakes → less benefit

**Figure**:
```
Panel plot:
  Left: Benefit vs wind speed (TI=0.06, WD=270°)
  Right: Benefit vs turbulence (WS=8, WD=270°)
```

#### 6.3.2 Optimal Yaw Patterns
```
Analysis of final yaw angles across conditions
Expected: Upstream turbines yaw 15-25°, downstream ~0-5°
```

### 6.4 Computational Performance

#### 6.4.1 Timing Breakdown
**Table**:
```
Configuration          | Opt Time | PyWake Calls | Cache Hit Rate
-----------------------|----------|--------------|---------------
Reference (10,500,50)  | 3.4s     | ~45          | 72%
Recommended (30,300,10)| 0.32s    | ~9           | 78%
Fastest (30,200,10)    | 0.29s    | ~7           | 80%
```

#### 6.4.2 Scaling Analysis
```
Expected RL training time (6 parallel envs, 100k steps):
  - Reference config:     ~500 hours (too slow!)
  - Recommended config:   ~55 hours (acceptable)
  - Fastest config:       ~48 hours (good)
```

### 6.5 Robustness and Reproducibility

#### 6.5.1 Seed Sensitivity Study
**Critical discovery documented in SEED_BIAS_DISCOVERY.md**

**Key finding**:
```
Single-seed test (original):
  "Best" config appeared to beat reference by 10,000W

Multi-seed test (corrected):
  Reference is actually 17,000W better on average!
```

**Figure**:
```
Grouped bar chart:
  X-axis: Seed number [0, 1, 2, 3, 4]
  Y-axis: Average power
  Groups: Reference vs "Best single-seed" config

Shows high variance for low-maxfun, low variance for high-maxfun
```

---

## 7. RL INTEGRATION (Future Work Section)

### 7.1 Hybrid MPC-RL Architecture

#### 7.1.1 Environment Design
```python
class MPCenv(WindFarmEnv):
    """RL environment with MPC base policy"""

    Observation space:
      - Wind conditions: WS, WD, TI (current + history)
      - Current yaw angles
      - MPC recommendations (optional)

    Action space:
      - Option 1: Modify MPC parameters (o1, o2)
      - Option 2: Add correction to MPC yaw commands
      - Option 3: Select from MPC library (different horizons)
```

#### 7.1.2 Reward Design
```python
reward = power_gain
         - action_penalty * ||Δγ||
         - mpc_cost_penalty * mpc_time
```

### 7.2 Learning Objectives

**What should RL learn?**
1. When to use aggressive vs conservative wake steering
2. How to adapt to wind transients faster than MPC reoptimization
3. Model error correction (real turbine ≠ PyWake predictions)
4. Meta-learning: adjust MPC parameters based on wind regime

### 7.3 Proposed Experiments

#### 7.3.1 Baseline Comparisons
```
1. Greedy (no optimization)
2. MPC-only (recommended params)
3. MPC + RL (our approach)
4. Pure RL (no MPC, for comparison)
```

#### 7.3.2 Learning Curves
```
Plot: Episode reward vs training steps
Expected: MPC+RL should bootstrap faster than pure RL
```

#### 7.3.3 Transfer Learning
```
Train on one wind regime, test on another
Expected: MPC provides physics-based generalization
```

---

## 8. DISCUSSION

### 8.1 Key Insights

#### 8.1.1 Evaluation Horizon is Critical
```
Our contribution: Identified that eval_horizon must be:
  T_eval ≥ max_ij(τ_ij) + T_AH + margin

Previous work often used too-short horizons, underestimating benefits
```

#### 8.1.2 Parameter Interactions Matter
```
Not just "higher maxfun is better"
Combination (dt_opt=30, T_opt=300, maxfun=10) is competitive because:
  - Coarse discretization simplifies optimization problem
  - Short horizon prevents overfitting
  - Few iterations sufficient for simpler problem
```

#### 8.1.3 Statistical Robustness is Essential
```
Single-seed tests can be misleading for stochastic optimization
Multi-seed averaging reveals true performance and variance
```

### 8.2 Practical Recommendations

#### 8.2.1 For MPC Implementation
```
Use dt_opt=30, T_opt=300, maxfun=10 for real-time control:
  - 10x faster than high-quality reference
  - <1% quality loss
  - Robust across random seeds
```

#### 8.2.2 For RL Training
```
Start with recommended params for stable MPC baseline
Monitor MPC solve time during training
If RL struggles with variance, increase maxfun to 15-20
```

### 8.3 Limitations

#### 8.3.1 Wake Model Fidelity
```
PyWake is engineering model, not LES
Real-world validation needed
Model uncertainty affects both MPC and RL
```

#### 8.3.2 Small Test Farm
```
3-turbine farm is simplified test case
Scaling to 50+ turbine farms open question
```

#### 8.3.3 Steady Wind Assumptions
```
Tests use constant wind over eval horizon
Real wind is turbulent, transient
Need dynamic wind scenarios for full validation
```

### 8.4 Future Directions

#### 8.4.1 Larger Farms
```
Test on 10, 20, 50+ turbine layouts
Investigate scaling of computational cost
Hierarchical optimization strategies
```

#### 8.4.2 Dynamic Wind
```
Time-varying wind speed, direction
Stochastic wind models
Forecasting integration
```

#### 8.4.3 Field Validation
```
Deploy on real wind farm or high-fidelity simulator
Compare predicted vs actual gains
Model calibration
```

#### 8.4.4 Advanced RL Methods
```
Multi-agent RL (one agent per turbine)
Meta-RL for fast adaptation
Safe RL with constraint handling
```

---

## 9. CONCLUSIONS

### 9.1 Summary
```
This work systematically investigated MPC parameterization for wind farm wake steering:

1. Identified critical role of evaluation horizon (must capture wake delays)
2. Conducted comprehensive parameter sweep (100 configs × 3 seeds)
3. Discovered Pareto-optimal configuration: 10x speedup, <1% quality loss
4. Demonstrated importance of multi-seed robustness testing
5. Provided open-source implementation for reproducibility
```

### 9.2 Impact
```
For practitioners:
  - Clear parameter recommendations for real-time MPC
  - Understanding of speed/quality/robustness tradeoffs

For researchers:
  - Methodology for rigorous MPC evaluation
  - Foundation for MPC+RL hybrid control
  - Reproducible benchmark for future work
```

### 9.3 Open Questions
```
1. How does RL improve upon optimized MPC baseline?
2. Can learned policies transfer across wind farms?
3. What is optimal action space for MPC-RL integration?
```

---

## 10. APPENDICES

### Appendix A: Mathematical Details
- Detailed derivation of time-shifted cost function
- Wake model equations
- Trajectory basis function properties

### Appendix B: Implementation Details
- Full algorithm pseudocode
- Caching strategy details
- Convergence criteria

### Appendix C: Additional Results
- Full parameter sweep tables
- All wind condition scenarios
- Extended robustness analysis

### Appendix D: Reproducibility
- Hardware specifications
- Software versions (Python, PyWake, etc.)
- Random seed values used
- Complete configuration files

---

## TEST SCRIPT IMPLEMENTATION PLAN

### Phase 1: Core Investigation (DONE ✓)
- [x] `test_optimization_quality.py` - Main parameter sweep
- [x] `test_optimization_quality_quick.py` - Quick validation
- [x] `test_single_scenario_debug.py` - Detailed diagnostics
- [x] `test_wake_steering_benefit.py` - Wind condition sensitivity
- [x] `test_maxfun_investigation.py` - Seed sensitivity

### Phase 2: Paper Figure Generation (TO DO)

#### Script 1: `test_wake_delay_analysis.py`
```
Purpose: Figure showing evaluation horizon impact
Experiment: Vary eval_horizon from 50 to 1000s
Output:
  - Plot: Power gain vs eval_horizon
  - CSV: Data for paper table
  - Figure saved as: fig_wake_delay_impact.pdf
```

#### Script 2: `test_parameter_heatmaps.py`
```
Purpose: Generate heatmaps for paper
Experiment: dt_opt vs T_opt for each maxfun
Output:
  - 5 heatmaps (one per maxfun value)
  - Combined figure for paper
  - Figure: fig_parameter_heatmaps.pdf
```

#### Script 3: `test_variance_analysis.py`
```
Purpose: Robustness/variance analysis
Experiment: 10 seeds per key configuration
Output:
  - Box plots showing distribution
  - Statistical tests (ANOVA)
  - Figure: fig_variance_analysis.pdf
```

#### Script 4: `test_scaling_analysis.py`
```
Purpose: Test on larger farms
Experiment: 3, 5, 10, 20 turbine farms
Output:
  - Scaling curves (time vs N)
  - Quality degradation (if any)
  - Figure: fig_scaling_analysis.pdf
```

#### Script 5: `test_wind_sensitivity.py`
```
Purpose: Comprehensive wind condition study
Experiment: Full factorial (WS × WD × TI)
Output:
  - 3D surface plots
  - Conditions where wake steering helps/hurts
  - Figure: fig_wind_sensitivity.pdf
```

#### Script 6: `test_convergence_analysis.py`
```
Purpose: Study optimization convergence
Experiment: Log intermediate solutions during optimization
Output:
  - Convergence curves
  - Optimal maxfun vs problem size
  - Figure: fig_convergence.pdf
```

### Phase 3: RL Integration (FUTURE)

#### Script 7: `train_mpc_baseline.py`
```
Purpose: MPC-only baseline for RL comparison
Output: Performance over many episodes
```

#### Script 8: `train_hybrid_mpc_rl.py`
```
Purpose: Train RL agent with MPC base policy
Output: Learning curves, trained models
```

#### Script 9: `evaluate_policies.py`
```
Purpose: Compare all approaches
Output: Benchmark table for paper
```

### Phase 4: Visualization and Tables

#### Script 10: `generate_all_paper_figures.py`
```
Purpose: One-click generation of all paper figures
Runs all test scripts and compiles results
Output: figures/ directory with publication-ready PDFs
```

#### Script 11: `generate_paper_tables.py`
```
Purpose: Generate LaTeX tables for paper
Output: tables/ directory with .tex files
```

---

## TIMELINE AND MILESTONES

### Milestone 1: Core Investigation (COMPLETED)
**Status**: ✓ Done
- All fundamental test scripts created
- Wake delay bug discovered and fixed
- Seed bias identified and corrected
- Multi-seed robustness implemented

### Milestone 2: Paper Figure Generation (2-3 weeks)
**Tasks**:
- [ ] Create 6 test scripts for paper figures
- [ ] Run comprehensive experiments
- [ ] Generate publication-quality figures
- [ ] Write results section with figures

### Milestone 3: Draft Manuscript (2-3 weeks)
**Tasks**:
- [ ] Write introduction and motivation
- [ ] Complete methodology section
- [ ] Write results with figure references
- [ ] Discussion and conclusions
- [ ] Abstract and references

### Milestone 4: RL Integration (4-6 weeks)
**Tasks**:
- [ ] Implement hybrid MPC-RL environment
- [ ] Train baseline policies
- [ ] Learning experiments
- [ ] Comparison analysis

### Milestone 5: Submission (1-2 weeks)
**Tasks**:
- [ ] Final polishing
- [ ] Co-author review
- [ ] Format for target venue
- [ ] Submit!

---

## TARGET VENUES

### Option 1: Control/Energy Journals
- **IEEE Transactions on Control Systems Technology**
  - Pros: Prestige, control focus
  - Cons: Long review time

- **Applied Energy**
  - Pros: High impact, energy focus
  - Cons: Less control theory depth

### Option 2: AI/ML Conferences
- **ICML / NeurIPS** (RL workshop or main track)
  - Pros: Visibility in ML community
  - Cons: Competitive, may need stronger RL results

- **AAAI**
  - Pros: Applications focus
  - Cons: May want more ML novelty

### Option 3: Wind Energy Conferences
- **Wind Energy Science** (journal)
  - Pros: Domain-specific, open access
  - Cons: Smaller audience

- **TORQUE / NAWEA** (conferences)
  - Pros: Direct wind energy community
  - Cons: May want field validation

### Recommendation
**Start with**: Applied Energy or IEEE TCST
**Reasoning**: Strong methodological contribution + practical application

---

## REPOSITORY ORGANIZATION

```
mpcrl/
├── mpcrl/                          # Core library
│   ├── mpc.py                      # MPC implementation
│   ├── environment.py              # RL environment
│   └── config.py                   # Configuration
│
├── tests/                          # Test scripts
│   ├── test_optimization_quality.py
│   ├── test_wake_delay_analysis.py
│   ├── ...
│   └── generate_all_paper_figures.py
│
├── results/                        # Experimental results
│   ├── data/                       # Raw data (CSV)
│   ├── figures/                    # Generated figures (PDF)
│   └── tables/                     # LaTeX tables
│
├── docs/                           # Documentation
│   ├── WAKE_DELAY_FIX_SUMMARY.md
│   ├── SEED_BIAS_DISCOVERY.md
│   ├── PERFORMANCE_OPTIMIZATION_GUIDE.md
│   └── PAPER_OUTLINE.md            # This file
│
├── examples/                       # Usage examples
│   ├── 01_environment_setup.ipynb
│   ├── 02_mpc_optimization.ipynb
│   └── 03_training_loop.ipynb
│
└── paper/                          # Manuscript (LaTeX)
    ├── main.tex
    ├── sections/
    ├── figures/                    # Symlink to results/figures
    └── tables/                     # Symlink to results/tables
```

---

## NOTES AND IDEAS

### Potential Additional Contributions
- **Benchmark dataset**: Release parameter sweep results as benchmark
- **Interactive visualization**: Web tool to explore tradeoffs
- **Tutorial**: Step-by-step guide for practitioners

### Collaboration Opportunities
- **Field validation**: Partner with wind farm operator
- **LES validation**: Collaborate with CFD group
- **Multi-farm**: Extend to wind farm clusters

### Software Engineering
- **CI/CD**: Automated testing of all scripts
- **Documentation**: Sphinx-based API docs
- **Packaging**: PyPI release for easy installation

---

**Last Updated**: 2025-10-17
**Status**: Phase 1 Complete, Phase 2 Planning
**Next Steps**: Create wake delay analysis script, begin figure generation
