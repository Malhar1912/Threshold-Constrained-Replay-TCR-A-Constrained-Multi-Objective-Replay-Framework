# CONSTRAINED OPTIMIZATION MODULE (0.2 - 0.3)

## Overview
This module formalizes the constrained stochastic resource allocation problem for replay distribution optimization.

---

## Core Mathematical Formulation

### Problem Definition (Section 0.2)

**Objective:** Maximize expected learning gain subject to distributional constraints

$$
\max_{q \in \Delta} \mathbb{E}_{\tau \sim q}[\Delta \mathcal{L}(\tau)]
\quad \text{s.t.} \quad
\text{KL}(q \| p) \le \epsilon, \quad |D^*| \le B
$$

**Where:**
- $q(\tau)$: Target replay distribution (decision variable)
- $p(\tau)$: Behavior distribution (empirical from buffer)
- $\Delta \mathcal{L}(\tau)$: Empirical learning gain per trajectory
- $\mathcal{L}(\tau)$: Supervised learning loss reduction
- $\epsilon$: KL divergence budget (trust region)
- $B$: Maximum buffer utilization (batch constraint)
- $\Delta$: Simplex of valid probability distributions

---

## Proxy Approximation (Section 0.3)

### Motivation
True learning gain $\Delta \mathcal{L}(\tau)$ is **not computable pre-selection** (requires post-replay measurement).

### Linear Approximation
$$
\Delta \mathcal{L}(\tau) \approx w^\top U(\tau)
$$

**Where:**
- $w \in \mathbb{R}^d$: Weight vector (learned or fixed)
- $U(\tau) = [U_1(\tau), U_2(\tau), \ldots, U_d(\tau)]^\top$: Multi-objective utility vector
- $d$: Number of utility objectives

### Optimal Distribution (Gibbs Form)
By substituting the proxy into the constraint maximization:

$$
q^*(\tau) \propto p(\tau)\exp(\beta w^\top U(\tau))
$$

**Where:**
- $\beta$: Inverse temperature (traded off against KL constraint)
- The form naturally emerges from Lagrangian duality

---

## Data Structures

### 1. Trajectory Representation

```
Trajectory τ = {
  "id": uint64,
  "timesteps": T,
  "states": {o_0, o_1, ..., o_T},
  "actions": {a_0, a_1, ..., a_{T-1}},
  "rewards": {r_0, r_1, ..., r_T},
  "embeddings": {z_0, z_1, ..., z_T},  // f_θ(o_t)
  "utility_vector": U(τ) ∈ ℝ^d,
  "behavior_prob": p(τ),
  "feasible": bool,  // τ ∈ D*?
}
```

### 2. Utility Vector Components

```
U(τ) = [
  U_1(τ): Normalized Reward,
  U_2(τ): Normalized Novelty (World Model Error),
  U_3(τ): Normalized TD Error,
  U_4(τ): Normalized Goal Proximity (sparse tasks only),
]
```

### 3. Constraint State

```
ConstraintState = {
  "kl_budget": ε,           // Hard KL divergence limit
  "buffer_capacity": B,      // Max |D*| size
  "current_feasible_size": |D*|,
  "thresholds": {θ_1, θ_2, θ_3, θ_4},
  "feasible_mask": bool[N],  // N = buffer size
}
```

### 4. Optimization Parameters

```
OptimizationParams = {
  "beta": β,                 // Inverse temperature
  "learning_rate": α,        // For weight updates
  "ema_decay": λ,            // For running statistics
  "weight_vector": w ∈ ℝ^d,
  "importance_weight_clip": (w_min, w_max),
}
```

---

## Algorithmic Pipeline

### Phase 0.2-0.3: Optimization Setup

**Input:**
- Buffer $D = \{\tau_1, \tau_2, \ldots, \tau_N\}$ with utility vectors
- Behavior distribution estimate $\hat{p}(\tau)$
- Feasibility constraint state

**Output:**
- Target sampling distribution $q^*(\tau)$
- Feasible subset $D^* \subseteq D$
- Per-trajectory importance weights $w_{IS}(\tau)$

---

## Implementation Steps

### Step 1: Validate Approximation Quality
**Required Experiment:**
$$
\text{corr}(w^\top U(\tau), \widehat{\Delta \mathcal{L}}(\tau)) > 0
$$

Measure correlation between:
- Proxy scores: $s(\tau) = w^\top U(\tau)$ (computable pre-replay)
- Actual gains: $\Delta\hat{\mathcal{L}}(\tau)$ (measured post-training)

**Success criterion:** $\rho > 0.3$ (min positive correlation required)

---

### Step 2: Feasibility Computation

$$
D^* = \{ \tau \in D \mid U_i(\tau) \ge \theta_i, \quad \forall i \in \{1, 2, 3, 4\} \}
$$

**Algorithm:**
```
feasible_set = []
for τ in D:
    if all(τ.U[i] >= θ[i] for i in range(d)):
        feasible_set.append(τ)
        feasible_mask[τ.id] = True
    else:
        feasible_mask[τ.id] = False
        
D* = feasible_set
```

**Constraint check:**
$$
\text{If } |D^*| < \alpha |D| \text{: RELAX thresholds by } \delta
$$

---

### Step 3: Distribution Computation

**Compute replay probabilities:**

$$
q(\tau) = \frac{\exp(\beta w^\top U(\tau))}{\sum_{\tau' \in D^*} \exp(\beta w^\top U(\tau'))}
$$

**Algorithm:**
```
log_scores = zeros(|D*|)
for i, τ in enumerate(D*):
    log_scores[i] = β * (w @ τ.U)

# Numerical stability: subtract max
log_scores = log_scores - max(log_scores)
scores = exp(log_scores)
q = scores / sum(scores)
```

---

### Step 4: Importance Sampling Weights

$$
w_{IS}(\tau) = \text{clip}\left(\frac{p(\tau)}{q(\tau)}, w_{min}, w_{max}\right)
$$

**Algorithm:**
```
p_values = behavior_probabilities(D*)  // From buffer statistics
q_values = q(τ)                        // From Step 3

IS_weights = clip(p_values / q_values, w_min, w_max)
```

---

### Step 5: Sanity Checks

**KL Divergence Validation:**

$$
\text{KL}(q \| p) = \sum_{\tau \in D^*} q(\tau) \log\left(\frac{q(\tau)}{p(\tau)}\right)
$$

Verify: $\text{KL}(q \| p) \le \epsilon$

**Buffer Constraint Validation:**

Verify: $|D^*| \le B$

---

## Key Hyperparameters

| Parameter      | Symbol | Range       | Notes                              |
| :------------- | :----- | :---------- | :--------------------------------- |
| KL Budget      | $\epsilon$    | [0.1, 1.0]  | Smaller = stricter trust region    |
| Buffer Limit   | $B$    | [0.5N, N]   | Fraction of buffer size            |
| Temperature    | $\beta$ | [0.1, 10.0] | Higher = peakier distribution      |
| Min Correlation| $\rho_{min}$ | > 0.3      | REQUIRED validation threshold      |

---

## Failure Mode Mitigations

### Mode 1: Infeasible Thresholds
**Problem:** No trajectories satisfy all thresholds → $D^* = \emptyset$

**Solution:**
```
if |D*| == 0:
    for i in range(d):
        θ[i] -= δ  // Symmetric relaxation
    recompute D*
```

### Mode 2: Divergence Explosion
**Problem:** $q \gg p$ causes numerical instability

**Solution:**
```
if KL(q||p) > ε:
    β = β * reduction_factor  // Reduce temperature
    recompute q
```

### Mode 3: Proxy Correlation Failure
**Problem:** $\text{corr}(w^\top U, \Delta\mathcal{L}) < 0.3$

**Solution:**
- Re-weight or retune $w$
- Augment utility components
- Consider task-specific feature engineering

---

## Integration Points

### With Phase 1 (Utility Computation)
- Input: $U(\tau)$ vectors (provided by Phase 1)
- Output: Feasibility classification $D^*$

### With Phase 2 (Threshold Adaptation)
- Coordinate: Adaptive $\theta$ updates via percentile statistics
- Ensure: Consistency between $D^*$ definitions

### With Phase 4 (Sampling Pipeline)
- Output: $q(\tau)$ + $w_{IS}$ passed to actor/learner
- Use alias method for efficient $O(1)$ sampling from $q$

---

## Validation Checklist

- [ ] Correlation test passes ($\rho > 0.3$)
- [ ] Feasible set non-empty for all buffer states
- [ ] KL divergence within budget
- [ ] Importance weights bounded and clipped
- [ ] Distribution normalized correctly
- [ ] Numerical stability at extreme values
- [ ] Edge cases: empty buffer, single trajectory, uniform weights

