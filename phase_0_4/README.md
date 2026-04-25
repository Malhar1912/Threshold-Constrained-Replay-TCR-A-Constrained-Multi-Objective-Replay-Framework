# Phase 0.4: Proxy Approximation Validity Gate

## Overview

This directory contains the complete implementation of **Phase 0.4**, the prerequisite validation gate for TCR (Threshold-Constrained Replay).

**Purpose:** Rigorously validate that the proxy approximation $w^\top U(\tau)$ meaningfully predicts true empirical learning gains $\widehat{\Delta\mathcal{L}}(\tau)$.

**Success Criterion:** $\text{corr}(w^\top U(\tau), \widehat{\Delta\mathcal{L}}(\tau)) > 0.3$ with $p < 0.05$ across ≥80% of tested environments.

**Timeline:** 1–2 weeks, 2–4 GPU-weeks of compute

---

## Directory Structure

```
phase_0_4/
├── config.yaml                 # Complete experiment configuration
├── src/
│   ├── __init__.py
│   ├── utility_computer.py     # Compute utility vectors U(τ)
│   ├── gain_measurer.py        # Measure learning gains ΔL̂(τ)
│   ├── correlation_validator.py # Compute & validate correlations
│   ├── visualizer.py           # Generate publication-quality figures
│   └── main.py                 # Main orchestrator & entry point
├── README.md                   # This file
└── results/                    # Output directory (auto-created)
    ├── figures/
    │   ├── 01_proxy_vs_gain_scatter.png
    │   ├── 02_correlation_barplot.png
    │   ├── 03_component_ablation_heatmap.png
    │   └── 04_summary_statistics.png
    ├── results.csv             # Per-environment results
    ├── results.json            # JSON summary
    └── summary.txt             # Human-readable report
```

---

## Core Modules

### 1. `utility_computer.py`
Computes multi-objective utility vectors $U(\tau) = [\hat{R}, \hat{\mathcal{N}}, \widehat{\Delta V}, \hat{G}]^\top$ from existing forward passes.

**Key Classes:**
- `UtilityComputer`: Main computation engine with EMA-based normalization
- `AggregateUtilityScore`: Computes proxy scores $w^\top U(\tau)$

**Components:**
- **Reward:** Discounted episodic return
- **Novelty:** World model prediction error
- **TD Error:** Value function uncertainty
- **Goal Proximity:** Distance to goal state (sparse tasks only)

### 2. `gain_measurer.py`
Measures empirical learning gains $\widehat{\Delta\mathcal{L}}(\tau)$ via stratified SGD sampling.

**Key Classes:**
- `StratifiedGainSampler`: Stratified sampling across utility quantiles
- `GainAnalyzer`: Statistics and filtering

**Efficiency:**
- Stratified sampling: measures 500 trajectories (not all 10K)
- Per-quantile stratification: ensures coverage across utility ranges
- Model checkpointing: avoids parameter drift

### 3. `correlation_validator.py`
Computes comprehensive correlation statistics with rigorous significance testing.

**Key Classes:**
- `CorrelationValidator`: Main correlation analysis engine
- `CorrelationReporter`: Aggregation & summary reporting

**Metrics Computed:**
- Pearson r (primary)
- Spearman ρ_s (robustness)
- Kendall τ (robustness)
- Bootstrap 95% CI
- Regression slope + p-value
- Cohen's q effect size
- Per-component ablations

**Acceptance Criteria:**
- ✓ $\rho > 0.3$ with $p < 0.05$
- ✓ $|(\text{Pearson} - \text{Spearman})| < 0.05$ (robustness)
- ✓ Positive slope: $\beta > 0$ (p < 0.05)
- ✓ No single component dominance: $\rho_i < 0.95$

### 4. `visualizer.py`
Generates 4 publication-quality figures:

**Figure 1:** Scatter plot grid (2×3)
- Per-environment proxy vs. learning gain scatter plots
- Linear regression fit with 95% CI band
- Density coloring

**Figure 2:** Correlation barplot
- Per-environment Pearson r with 95% CI errorbars
- Pass/fail coloring (green/red)
- Threshold line

**Figure 3:** Component ablation heatmap
- Rows: Environments
- Cols: Utility components
- Values: Per-component correlations

**Figure 4:** Summary statistics
- Correlation distribution
- Slope distribution
- Pass/fail breakdown (pie chart)
- Statistical summary table

### 5. `main.py`
Main orchestrator coordinating the full pipeline.

**Key Class:**
- `Phase04Orchestrator`: Coordinates all submodules

**Workflow:**
1. Load configuration
2. Initialize all components
3. Run validation per environment
4. Aggregate results
5. Generate visualizations
6. Save results (CSV, JSON, TXT)
7. Determine gate pass/fail

---

## Configuration

All experiment parameters are defined in `config.yaml`:

### Key Sections

**Benchmarks:**
```yaml
benchmarks:
  atari:
    games: [Pong, Breakout, Seaquest, SpaceInvaders, MsPacman]
    episodes_per_game: 200
  dmc:
    tasks: [walker-walk, cheetah-run, finger-spin]
    episodes_per_task: 100
```

**Utility Components:**
```yaml
utility:
  components:
    - {name: reward, enabled: true}
    - {name: novelty, enabled: true}
    - {name: td_error, enabled: true}
    - {name: goal_proximity, enabled: true, task_specific: true}
```

**Acceptance Criteria:**
```yaml
acceptance_criteria:
  correlation_threshold: 0.3
  pvalue_threshold: 0.05
  min_env_pass_rate: 0.8  # ≥80% of envs must pass
  spearman_tolerance: 0.05
```

---

## Usage

### Quick Start (Mock Data)

```python
from src.main import Phase04Orchestrator, create_mock_experiment_data

# Generate mock data
trajectories, utilities, learning_gains = create_mock_experiment_data(
    n_envs=8, n_trajectories=500, true_correlation=0.42
)

# Run experiment
orchestrator = Phase04Orchestrator(config_path="config.yaml")
results = orchestrator.run_validation_pipeline(trajectories, utilities, learning_gains)

# Check gate status
if results['gate_status']['pass']:
    print("✓ GATE PASSED: Proceed to Phases 1-7")
else:
    print("✗ GATE FAILED: Re-engineer utilities")
```

### Real Data (DreamerV3 Integration)

```python
from src.main import Phase04Orchestrator
from src.utility_computer import UtilityComputer
from real_env_loader import load_dreamer_data  # custom loader

# Load pre-collected data from DreamerV3 agent
trajectories, embeddings, values = load_dreamer_data(
    env_list=["Pong", "Breakout", ...],
    buffer_size=1000000,
)

# Compute utilities
utility_computer = UtilityComputer(gamma=0.99)
utilities_dict = {}
for env_name, traj_list in trajectories.items():
    utilities = np.array([
        utility_computer.compute_trajectory_utilities(t)
        for t in traj_list
    ])
    utilities_dict[env_name] = utilities

# Measure gains (requires model access)
from src.gain_measurer import StratifiedGainSampler
gains_dict = measure_learning_gains_per_env(...)

# Run validation
orchestrator = Phase04Orchestrator(config_path="config.yaml")
results = orchestrator.run_validation_pipeline(trajectories, utilities_dict, gains_dict)
```

---

## Output

### Formats

1. **CSV** (`results.csv`)
   - Per-environment results with all metrics

2. **JSON** (`results.json`)
   - Structured summary with aggregate statistics

3. **Text** (`summary.txt`)
   - Human-readable report with full result details

4. **Figures** (`figures/`)
   - 4 publication-quality PNG plots at 300 DPI

### Interpretation

**Pass Example:**
```
✓ PASS Environment: Pong
──────────────────────────────────
Sample Size:              n = 500
Pearson r:                ρ = 0.42 (95% CI: [0.37, 0.48])
P-value:                  p < 0.001 ***
Spearman ρ_s:             ρ_s = 0.41 (CI: [0.36, 0.47])

Regression:
  Slope (β):              β = 0.35 (p < 0.001)
  R²:                     R² = 0.18

Decision: All criteria pass
```

**Fail Example:**
```
✗ FAIL Environment: Seaquest
──────────────────────────────────
Sample Size:              n = 500
Pearson r:                ρ = 0.21 (95% CI: [0.14, 0.28])
P-value:                  p < 0.001

Decision: r=0.21 ≤ 0.3; failed minimum correlation
```

---

## Timeline & Phases

| Phase | Task | Duration | Dependencies |
|:------|:-----|:---------|:-------------|
| **1** | Data collection | 2–3 days | Environment + agent setup |
| **2** | Utility computation | 2–4 hrs | Phase 1 data |
| **3** | Learning gain measurement | 3–7 days | Phase 1 data + model |
| **4** | Correlation analysis | 1 hr | Phases 2–3 data |
| **5** | Visualization | 4–8 hrs | Phase 4 results |
| **Total** | **1–2 weeks** | **2–4 GPU-weeks** | |

---

## Requirements

### Python Dependencies

```yaml
packages:
  - numpy >= 1.20
  - scipy >= 1.7
  - matplotlib >= 3.4
  - seaborn >= 0.11
  - pandas >= 1.3
  - pyyaml >= 5.4
```

### Compute Resources

- **1x GPU** for data collection & gain measurement
- **1x CPU** for utility computation & analysis
- **~9 GB disk** for 1M trajectories + results

---

## Failure Modes & Recovery

### Scenario A: $\rho < 0.3$ Globally

**Diagnosis:**
- Utility components don't capture learning dynamics
- Ground truth $\widehat{\Delta\mathcal{L}}$ measurement is noisy

**Recovery:**
1. Audit $\widehat{\Delta\mathcal{L}}$ computation (verify loss calculation)
2. Increase gain measurement sample size (batch > 1)
3. Re-engineer utility components:
   - Add higher-order statistics (trajectory variance, skewness)
   - Include state visitation frequency
   - Consider trajectory length as feature
4. Optimize $w$ via regression instead of equal weights

### Scenario B: High $\rho$ for 1–2 Envs, Low for Others

**Diagnosis:**
- Task-specific effects (utilities sensitive to environment structure)

**Recovery:**
1. Stratify analysis by environment type (discrete vs. continuous)
2. Learn per-environment weight vectors
3. Analyze per-component importance via SHAP/permutation

### Scenario C: Spearman divergence from Pearson ($|r - \rho_s| > 0.05$)

**Diagnosis:**
- Outlier sensitivities; nonlinear effects

**Recovery:**
1. Filter outliers (trim 5%) and re-measure
2. Consider robust correlation metrics (Winsorized correlation)
3. Investigate outlier trajectories manually

---

## References

- **Correlation Testing:** Wald et al. (1943) "Tests of Statistical Hypotheses"
- **Bootstrap CI:** Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
- **Effect Sizes:** Cohen (1988) "Statistical Power Analysis for Behavioral Sciences"
- **Importance Weighting:** Sugiyama et al. (2012) "Machine Learning from Weak Learners"

---

## Contact & Support

For issues or questions, refer to parent framework documentation or contact TCR development team.

**Status:** ✓ Ready for implementation (May 2026)

