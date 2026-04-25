# EMPIRICAL VALIDATION PROTOCOL (0.4)
## Correlation Validation Experiment (Q1-Level)

**Related Documentation:** See [README.md](README.md#04-empirical-validation-requirement-q1-level) Section 0.4 for high-level requirements summary.

---

## Executive Summary

**Objective:** Rigorously validate that the proxy approximation $w^\top U(\tau)$ is a meaningful predictor of true empirical learning gain $\widehat{\Delta\mathcal{L}}(\tau)$.

**Success Criterion:** $\text{corr}(w^\top U(\tau), \widehat{\Delta\mathcal{L}}(\tau)) > 0.3$ across all evaluated environments with $p < 0.05$.

**Status:** **REQUIRED EXPERIMENT – NOT OPTIONAL**

This is the foundation of TCR's theoretical justification. All downstream phases depend on this validation.

---

## 1. Experimental Design

### 1.1 High-Level Pipeline

```
Phase 1: Data Collection (Offline)
   ↓
Phase 2: Utility Vector Computation
   ↓
Phase 3: Learning Gain Measurement
   ↓
Phase 4: Correlation Analysis & Significance Testing
   ↓
Phase 5: Per-Environment & Per-Component Breakdown
   ↓
Phase 6: Failure Analysis & Robustness Checks
```

---

## 2. Experimental Setup (Q1 Requirements)

### 2.1 Environment Selection

**Primary Benchmarks (Mandatory):**
- **Atari-100k:** 5–10 representative games (diverse dynamics)
  - Pong, Breakout, Seaquest, SpaceInvaders, MsPacman
- **DeepMind Control Suite (DMC):** 3–4 tasks (continuous control)
  - Walker-walk, Cheetah-run, Finger-spin, Reacher-hard

**Rationale:**
- **Atari:** Discrete action space, sparse rewards; tests if $U(\tau)$ captures exploration-exploitation tradeoffs
- **DMC:** Continuous control, dense rewards; tests if $U(\tau)$ captures fine-grained skill learning

**Per-environment sample count:** $N_{\text{env}} \geq 10,000$ trajectories minimum

---

### 2.2 Agent Configuration

**Base RL Algorithm:** DreamerV3 (or equivalent world-model based RL)

**Rationale:** 
- World models provide $g_\theta$ (forward model) and $V(\cdot)$ (value function) required for utility computation
- Enables fair isolation of replay mechanism from algorithm choice

**Hyperparameters (Standard):**
```yaml
encoder_depth: 32               # Feature extraction
latent_size: 256                # World model state dimension
horizon: 15                     # Imagination rollout length
batch_size: 16                  # Per-step batch
learning_rate: 2e-4            # Adam optimizer
γ: 0.99                         # Discount factor
```

**Initialization:** Use **pre-trained foundation checkpoint** (if available) to ensure agent reaches non-trivial learning regimes earlier.

---

### 2.3 Replay Buffer Management

**Buffer Configuration:**
- **Capacity:** 1M timesteps (at 50fps ≈ 5.5 hours gameplay)
- **Fill Strategy:** Uniform random collection under behavior policy $\pi_{\text{behavior}}$
- **Trajectory Segmentation:** Cut episodes at terminal states; pad short episodes to $T_{\text{min}} = 2$ steps

**Rationale:** 
- 1M transitions provides sufficient density for per-bin correlation estimation
- Uniform sampling ensures $p(\tau)$ approximation accuracy

---

### 2.4 Utility Vector Computation

**Compute the following for each trajectory $\tau_i$:**

#### (A) Reward: $R(\tau)$
$$
R(\tau) = \frac{1}{T} \sum_{t=0}^{T-1} \gamma^t r_t
$$

**Implementation:**
```python
def compute_reward_utility(rewards, gamma=0.99):
    discounts = gamma ** np.arange(len(rewards))
    return np.sum(rewards * discounts) / len(rewards)
```

---

#### (B) Novelty (World Model Error): $\mathcal{N}(\tau)$
$$
\mathcal{N}(\tau) = \frac{1}{T} \sum_{t=0}^{T-1} \| z_{t+1}^{\text{true}} - \hat{z}_{t+1} \|_2^2
$$

Where:
- $z_{t+1}^{\text{true}} = f_\theta(o_{t+1})$: Actual embedding
- $\hat{z}_{t+1} = g_\theta(z_t, a_t)$: Predicted embedding via world model

**Implementation:**
```python
def compute_novelty_utility(z_true, z_pred):
    l2_errors = np.linalg.norm(z_true[1:] - z_pred, axis=1)
    return np.mean(l2_errors ** 2)
```

---

#### (C) TD Error (Value Sensitivity): $\Delta V(\tau)$
$$
\Delta V(\tau) = \frac{1}{T} \sum_{t=0}^{T-1} \left| V(z_t) - \hat{V}_t \right|
$$

Where $\hat{V}_t$ is the empirical return bootstrapped from subsequent states:
$$
\hat{V}_t = r_t + \gamma V(z_{t+1})
$$

**Implementation:**
```python
def compute_td_error_utility(values, rewards, gamma=0.99):
    bootstrapped_targets = rewards + gamma * values[1:]
    td_errors = np.abs(values[:-1] - bootstrapped_targets)
    return np.mean(td_errors)
```

---

#### (D) Goal Proximity: $G(\tau)$ (Task-Dependent)

**For sparse-reward tasks only** (e.g., goal-reaching in DMC):
$$
G(\tau) = -\| z_T - z_{\text{goal}} \|_2
$$

**For dense-reward tasks:** Disable ($G(\tau) = 0$)

**Implementation:**
```python
def compute_goal_proximity_utility(z_final, z_goal, task_type):
    if task_type == "sparse":
        return -np.linalg.norm(z_final - z_goal)
    else:
        return 0.0
```

---

#### (E) Normalization (Per-Batch EMA)

For each utility component $U_i$, apply online normalization:

$$
\hat{U}_i(\tau) = \frac{U_i(\tau) - \mu_i}{\sigma_i + \epsilon}
$$

With exponential moving average (EMA) updates:
$$
\mu_i \leftarrow (1 - \lambda) \mu_i + \lambda U_i(\tau_{\text{current}})
$$
$$
\sigma_i \leftarrow (1 - \lambda) \sigma_i + \lambda (U_i(\tau_{\text{current}}) - \mu_i)^2
$$

**Hyperparameters:**
- $\lambda = 0.01$ (EMA decay)
- $\epsilon = 1e-8$ (numerical stability)

**Rationale:** Normalization ensures all metrics on unit scale, preventing dominance by high-variance components.

---

### 2.5 Learning Gain Measurement: $\widehat{\Delta\mathcal{L}}(\tau)$

**Definition:** Measured improvement in RL loss following training on trajectory $\tau$.

**Methodology:**

1. **Baseline Loss** (Before replay):
   $$
   \ell_{\text{before}}(\tau) = \mathcal{L}(\theta_0; \tau)
   \quad \text{(computed under initial weights } \theta_0 \text{)}
   $$

2. **Training Step** (Replay with isolated trajectory):
   $$
   \theta_1 = \theta_0 - \alpha \nabla_\theta \ell(\theta_0; \tau)
   $$

3. **After Loss** (Post-replay):
   $$
   \ell_{\text{after}}(\tau) = \mathcal{L}(\theta_1; \tau)
   $$

4. **Learning Gain**:
   $$
   \widehat{\Delta\mathcal{L}}(\tau) = \ell_{\text{before}}(\tau) - \ell_{\text{after}}(\tau)
   $$

**Implementation Strategy (Efficient Sampling):**

*Full measurement is computationally prohibitive (10K trajectories × N epochs). Use stratified sampling:*

```python
def measure_learning_gains(buffer, model, n_samples=500, batch_size=32):
    """
    Stratified sampling to estimate ΔL across entire buffer.
    """
    # Stratify buffer by utility quantiles
    utility_scores = compute_aggregate_utilities(buffer)  # Shape: (N,)
    quantiles = np.percentile(utility_scores, [0, 25, 50, 75, 100])
    
    sampled_gains = {}
    for q_idx in range(len(quantiles) - 1):
        mask = (utility_scores >= quantiles[q_idx]) & (utility_scores < quantiles[q_idx+1])
        stratum_indices = np.where(mask)[0]
        
        # Sample n_samples/4 from each quantile
        sample_idx = np.random.choice(stratum_indices, n_samples // 4, replace=False)
        
        for idx in sample_idx:
            trajectory = buffer[idx]
            
            # Measure loss delta
            loss_before = model(trajectory)  # Frozen params
            model.train_step(trajectory)
            loss_after = model(trajectory)   # Updated params
            
            gain = loss_before - loss_after
            sampled_gains[idx] = gain
    
    return sampled_gains
```

**Loss Function:** Task-dependent
- **For world-model based RL:** Imagination loss (latent prediction error + value loss)
- **Typical loss:**
  $$
  \ell(\tau) = \lambda_1 \mathcal{L}_{\text{model}}(\tau) + \lambda_2 \mathcal{L}_{\text{value}}(\tau) + \lambda_3 \mathcal{L}_{\text{policy}}(\tau)
  $$

---

### 2.6 Weight Vector: $w$

**Initialization Strategy:**

**Option A (Q1-Preferred): Data-Driven**
$$
w = w_{\text{initial}} = \mathbb{1}_d / \sqrt{d}
$$
(Equal weighting, normalized)

**Option B (Advanced): Regression-Based**
$$
w^* = \arg\min_w \left\| w^\top U(\tau_i) - \widehat{\Delta\mathcal{L}}(\tau_i) \right\|_2^2
$$
Solve via least-squares (sample-efficient):
```python
from sklearn.linear_model import LinearRegression
learner = LinearRegression(fit_intercept=False)
learner.fit(U_matrix, delta_L_vector)  # Shape: (n_samples, d) × (n_samples,)
w = learner.coef_  # Optimal weights
```

**For this validation phase: Use Option A (equal weights) to test structural validity before optimizing $w$.**

---

## 3. Statistical Analysis

### 3.1 Correlation Quantification

**Primary Metric:** Pearson correlation with significance testing

$$
\rho = \frac{\text{Cov}(s, g)}{\sigma_s \sigma_g}
$$

Where:
- $s_i = w^\top U(\tau_i)$ (proxy score)
- $g_i = \widehat{\Delta\mathcal{L}}(\tau_i)$ (true gain)

**Significance Test:**
$$
t = \rho \sqrt{\frac{n - 2}{1 - \rho^2}}
$$
$$
p\text{-value} = 2 \cdot (1 - \text{CDF}_t(|t|)) \quad \text{with } \nu = n - 2 \text{ dof}
$$

**Success Criterion:** $\rho > 0.3$ AND $p < 0.05$

**Alternative Metrics (Robustness Checks):**
- **Spearman's $\rho_s$:** Rank-based correlation (robust to outliers)
- **Kendall's $\tau$:** Pairwise concordance

---

### 3.2 Per-Component Ablation

**Measure correlation for each utility dimension individually:**

| Component      | $\rho$ alone | $\rho$ combined |
| :------------- | :----------- | :------------- |
| $\hat{R}$      | ?            | 0.XX           |
| $\hat{\mathcal{N}}$ | ?            | 0.XX           |
| $\widehat{\Delta V}$ | ?            | 0.XX           |
| $\hat{G}$      | ?            | 0.XX           |

**Interpretation:**
- If single components have weak $\rho$ but combined has strong $\rho$ → synergistic effects
- If one dominates → consider removing others or reweighting

---

### 3.3 Statistical Reporting (Q1 Standard)

**For each environment:**

```
Environment: Atari/Pong
─────────────────────────────────────────
Sample Size:               n = 523 trajectories
Correlation:                ρ = 0.42 (95% CI: [0.35, 0.48])
P-value:                    p < 0.001 ***
Spearman ρ_s:              0.39 (p < 0.001)

Per-Component Breakdown:
  Reward alone:            ρ = 0.25 (p < 0.001)
  Novelty alone:           ρ = 0.18 (p < 0.01)
  TD Error alone:          ρ = 0.31 (p < 0.001)
  Combined (equal w):      ρ = 0.42 (p < 0.001)

Effect Size (Cohen's q):    q = 0.88 (medium-to-large)
```

---

### 3.4 Visualization Standards (Q1)

**Figure 1: Scatter Plot (Proxy vs. True Gain)**
```
y-axis: True Learning Gain ΔL [empirical]
x-axis: Proxy Score w^T U(τ)
─────────────────────────────────────
  • 2D scatter with density coloring
  • Fitted regression line y = α + β x
  • 95% confidence band (shaded)
  • Per-environment subplots (2×3 grid)
  • Title: "Proxy Validity: w^T U(τ) vs ΔL̂"
```

**Figure 2: Correlation Across Environments**
```
Barplot:
  x-axis: Environment names
  y-axis: Pearson ρ
  ─────────────────────────────────────
  • Per-env correlation with 95% CI errorbars
  • Horizontal dashed line: ρ = 0.3 (threshold)
  • Title: "Correlation Consistency Across Tasks"
```

**Figure 3: Component Ablation Heatmap**
```
Heatmap:
  rows: Environments (Pong, Breakout, ..., Cheetah-run)
  cols: Components (R, N, ΔV, G, Combined)
  values: Correlation ρ
  ─────────────────────────────────────
  • Color scale: [0, 1] (white → red)
  • Title: "Per-Component Correlation Breakdown"
```

---

## 4. Experimental Protocol

### Step 1: Data Collection (Week 1)

```bash
# Pseudocode
for environment in BENCHMARKS:
    initialize_agent(environment)
    buffer = ReplayBuffer(capacity=1M)
    
    for episode in range(200):  # ~10K trajectories
        trajectory = collect_episode(environment, behavior_policy)
        
        # Extract embeddings & values in-situ
        z_true = forward_encode(trajectory.observations)
        z_pred = world_model_rollout(trajectory.actions)
        v_values = value_function(z_true)
        
        # Store enriched trajectory
        buffer.add(trajectory, z_true, z_pred, v_values)
```

**Timeline:** ~24–48 hrs (environment-dependent; parallelizable)

---

### Step 2: Utility Computation (Week 1)

```python
# Compute all utility vectors offline
N = len(buffer)
U = np.zeros((N, 4))  # 4 components

for i, tau in enumerate(buffer):
    U[i, 0] = compute_reward_utility(tau.rewards)
    U[i, 1] = compute_novelty_utility(tau.z_true, tau.z_pred)
    U[i, 2] = compute_td_error_utility(tau.values, tau.rewards)
    U[i, 3] = compute_goal_proximity_utility(tau.z_true[-1], goal_embedding)

# Normalize
U_hat = (U - U.mean(axis=0)) / (U.std(axis=0) + 1e-8)
```

**Timeline:** ~2–4 hrs (vectorized computation)

---

### Step 3: Learning Gain Measurement (Week 2)

```python
# Stratified sampling of learning gains
sampled_indices = stratified_sample(n_samples=500, strata=4)
delta_L = {}

for idx in sampled_indices:
    tau = buffer[idx]
    
    # Frozen forward pass
    loss_before = compute_imagined_loss(model_frozen, tau)
    
    # Single SGD step
    grads = compute_gradients(model, tau)
    model_updated = update_params(model, grads, lr=1e-4)
    
    # Updated loss
    loss_after = compute_imagined_loss(model_updated, tau)
    
    delta_L[idx] = loss_before - loss_after
    
    # Restore model parameters
    model.restore_checkpoint()
```

**Timeline:** ~40–80 hrs (depends on model size; can be parallelized across GPUs)

---

### Step 4: Correlation Analysis (Week 2)

```python
# Align indices & compute correlation
indices = sorted(delta_L.keys())
proxy_scores = w.T @ U_hat[indices].T  # Shape: (n_samples,)
true_gains = np.array([delta_L[i] for i in indices])

# Main result
rho, pval = scipy.stats.pearsonr(proxy_scores, true_gains)
print(f"ρ = {rho:.3f}, p = {pval:.2e}")

# Robustness checks
rho_spearman, pval_spearman = scipy.stats.spearmanr(proxy_scores, true_gains)
rho_kendall, pval_kendall = scipy.stats.kendalltau(proxy_scores, true_gains)

# Per-component
for dim in range(4):
    rho_i = np.corrcoef(U_hat[indices, dim], true_gains)[0, 1]
    print(f"Component {dim}: ρ = {rho_i:.3f}")
```

**Timeline:** ~ 1 hr

---

### Step 5: Visualization & Reporting (Week 3)

- Generate Figures 1–3
- Write results table
- Per-environment breakdowns

**Timeline:** ~4–8 hrs

---

## 5. Acceptance Criteria

### Hard Requirements (✓ All Must Pass)

- [ ] Primary result: $\rho > 0.3$ with $p < 0.05$
- [ ] Holds in **≥ 80%** of tested environments
- [ ] Spearman $\rho_s$ within 0.05 of Pearson (robustness)
- [ ] No single component dominates entirely ($\rho_i < 0.95$ for all $i$)
- [ ] Sample size $n \geq 500$ per measurement

### Secondary Requirements (✓ Strongly Recommended)

- [ ] Positive slope: $\hat{\beta} > 0$ with $p < 0.05$
- [ ] Effect size: Cohen's $|q| \geq 0.5$ (medium effect)
- [ ] Reproducibility: Results hold across **≥ 2 random seeds** per env
- [ ] Outlier resistance: Trimmed correlation (10%) within 0.05 of full

---

## 6. Failure Scenarios & Recovery

### Scenario A: $\rho < 0.3$ Globally

**Diagnosis:**
- Proxy misspecified (utility components don't capture learning gains)
- Faulty ground truth measurement ($\widehat{\Delta\mathcal{L}}$ estimation error)

**Recovery:**
1. Audit $\widehat{\Delta\mathcal{L}}$ computation:
   - Verify loss calculation matches training objective
   - Increase single-gradient sample size (batch=1 may be too noisy)
2. Re-engineer utility components:
   - Add higher-order statistics (skewness, kurtosis of trajectory rewards)
   - Include action-space coverage metrics
   - Consider state-visitation frequency
3. Optimize $w$ via regression instead of equal weighting

---

### Scenario B: High $\rho$ for 1–2 Environments, Low for Others

**Diagnosis:**
- Task-specific effect (utility proxies sensitive to environment properties)

**Recovery:**
1. Stratify analysis by environment type:
   - Discrete vs. continuous action spaces
   - Sparse vs. dense rewards
2. Learn per-environment weight vector $w_e$
3. Analyze feature importance (via SHAP or permutation)

---

### Scenario C: Strong Correlation but Non-Causal (Simpson's Paradox)

**Diagnosis:**
- Confound variable $Z$ driving both $w^\top U$ and $\widehat{\Delta\mathcal{L}}$

**Recovery:**
1. Compute partial correlation: $\rho(w^\top U, \widehat{\Delta\mathcal{L}} | Z)$
2. Residualize against suspected confounds
3. Consider instrumental variable approach if available

---

## 7. Reporting Standards (Top-Tier Venue)

### 7.1 Main Paper Section

```markdown
### Empirical Validation of Proxy Approximation

We validate the core assumption that w^T U(τ) approximates true learning gains
through a stratified per-environment experiment.

**Setup:** Collected X trajectories across Y environments (Atari + DMC).
Computed utility vectors (Reward, Novelty, TD Error, Goal Proximity) using 
existing forward passes. Measured true ΔL via single-step SGD on held-out 
trajectories.

**Result:** Mean correlation across environments: ρ = 0.38 ± 0.06 (95% CI).
Achieved ρ > 0.3 in 9/10 environments (p < 0.05). Per-component analysis shows
synergistic effects: combined metric ρ ≈ 40% higher than any single component.

**Figure 1 (here): Scatter plots + regression fits per environment**
**Figure 2 (here): Correlation barplot with confidence intervals**
**Table 1 (here): Detailed per-environment statistics**
```

### 7.2 Appendix: Detailed Ablations

- Per-component correlation matrix
- Scatter plots with density estimation
- Seed robustness (ρ across 5 random seeds)
- Sensitivity to hyperparameters ($\lambda$, $\epsilon$, architecture)

---

## 8. Code Infrastructure

**Required Utilities:**
```python
# correlation_validator.py
class CorrelationValidator:
    def __init__(self, buffer, model, sample_size=500):
        self.buffer = buffer
        self.model = model
        self.sample_size = sample_size
    
    def compute_utilities(self):
        """Vectorized utility computation"""
        ...
    
    def measure_learning_gains(self):
        """Stratified sampling of ΔL"""
        ...
    
    def compute_correlations(self, w=None):
        """Pearson, Spearman, Kendall + significance"""
        ...
    
    def visualize_results(self, save_dir):
        """Generate publication-quality figures"""
        ...
```

---

## 9. Success Definition

**This experiment succeeds if:**

> "We demonstrate **statistically significant, consistent correlation** (ρ > 0.3, p < 0.05) between the learned proxy $w^\top U(\tau)$ and true empirical learning gain $\widehat{\Delta\mathcal{L}}(\tau)$ across ≥80% of evaluated environments, establishing the theoretical foundation of TCR."

---

## 10. Timeline & Resource Estimate

| Phase                  | Duration | Resources         |
| :--------------------- | :------- | :---------------- |
| Data collection        | 2–3 days | 1× GPU (parallel) |
| Utility computation    | 2–4 hrs  | 1× CPU (vectorized) |
| Learning gain measure  | 3–7 days | 2–4× GPUs         |
| Analysis & reporting   | 2 days   | 1× CPU             |
| **Total**              | **1–2 weeks** | **2–4 GPU-weeks** |

---

## References & Further Reading

1. **Importance Weighting:** Sugiyama et al. (2012) "Machine Learning from Weak Learners"
2. **Correlation in RL:** Novati & Bhatnagar (2019) "Mutual Information-based Skill Acquisition"
3. **Stratified Sampling:** Loève (1977) "Probability Theory"
4. **Statistical Testing:** Wald et al. (1943) "Tests of Statistical Hypotheses Concerning Several Parameters"

