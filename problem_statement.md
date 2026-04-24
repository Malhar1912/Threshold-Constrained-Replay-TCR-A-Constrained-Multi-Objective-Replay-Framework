# Problem Statement: Threshold-Constrained Replay (TCR)

## Executive Summary

Experience replay is a cornerstone of off-policy reinforcement learning, enabling sample-efficient learning by revisiting past trajectories. However, **current replay methods treat sampling as a ranking problem**, selecting trajectories based on a single proxy metric (e.g., Temporal Difference error in Prioritized Experience Replay). This approach suffers from fundamental inefficiencies:

1. **Low-utility samples accumulate**: Normalization ensures even unprofitable transitions get some probability mass, wasting compute.
2. **No utility guarantees**: A trajectory with moderate reward, novelty, and value sensitivity may be discarded if another has slightly higher TD-error—even if the first is crucial for robust learning.
3. **Single-metric myopia**: Real learning requires balancing multiple objectives (discovering novelty, maximizing return, reducing uncertainty). Ranking collapses this into a scalar.

We propose **reframing replay sampling as a constrained stochastic resource allocation problem**, where:
- **Objective**: Maximize expected learning gain across selected trajectories
- **Constraints**: KL-divergence bound on replay distribution (for stable off-policy learning) and buffer capacity limits
- **Resource**: Limited compute budget for replay interactions

This document justifies this reframing, establishes the formal problem, and grounds it empirically.

---

## 1. The Replay Sampling Problem: Current Paradigm

### 1.1 Background: Why Replay Matters

In off-policy RL, an agent learns from a replay buffer $D$ containing trajectories $\tau = \{(o_t, a_t, r_t)\}_{t=0}^T$ collected under a behavior policy. Each gradient update samples a minibatch from this buffer. 

**Key trade-off**: With finite compute, we must choose which trajectories to replay. Different choices lead to different learning efficiency:
- Uniform replay: Simple, stable gradient estimates, but wastes compute on low-value transitions.
- Prioritized replay: Emphasizes "surprising" transitions (high TD-error), but ignores other learning signals.
- No replay method: Learns online only, highly sample-inefficient.

The choice of replay distribution $q(\tau)$ directly impacts sample efficiency and wall-clock performance.

### 1.2 Prioritized Experience Replay (PER): The Current Standard

PER formalizes replay as an **implicit ranking problem**:

$$
\text{PER ranking score} = (\delta_t + \epsilon)^\alpha
$$

where $\delta_t = |r_t + \gamma V(s_{t+1}) - V(s_t)|$ is the TD-error, $\epsilon$ is a small constant, and $\alpha \in [0, 1]$ controls prioritization strength.

The replay distribution becomes:
$$
q_{\text{PER}}(\tau) \propto \left( \max_t (\delta_t + \epsilon)^\alpha \right)
$$

**Strengths**:
- Empirically effective across diverse domains
- Intuitive: prioritize "surprising" transitions
- Reduces correlation in gradient estimates

**Fundamental limitations**:

| Limitation | Manifestation | Example |
|-----------|--|--|
| **Single-metric myopia** | Ignores complementary learning signals | High-reward, low-TD-error trajectory with novel state representations discarded |
| **Normalization waste** | Probability mass on low-value samples | Outlier single-step transitions occupy 50% of minibatch due to softmax normalization |
| **No utility threshold** | Cannot guarantee minimum learning quality | Replaying trajectories below a quality bar to meet batch size (`beta` hyperparameter forces minibatch sampling even when buffer is exhausted) |
| **Policy divergence risk** | Large $q/p$ ratio destabilizes learning | IS-weighting explodes; value estimates become unreliable in off-policy regions |
| **No principled constraint** | Arbitrary hyperparameter choices | Why $\alpha = 0.6$? Why $\epsilon = 10^{-6}$? No formal optimization objective |

### 1.3 Multi-Objective Replay: Partial Solutions

Recent work has attempted to address single-metric myopia by combining multiple objectives:

- **Curiosity + Reward**: Concatenate novelty score and return; rank by weighted sum
- **Uncertainty-weighted replay**: Use prediction uncertainty from ensemble models
- **Multi-target PER**: Separate TD-errors for different value functions

**However**, these approaches remain ranking-based: they produce a composite scalar and sort. They suffer from the same fundamental issues as single-metric PER:
- ✗ Still enforce normalization across entire buffer (no quality threshold)
- ✗ Composite scores can be arbitrarily conflicting (e.g., high novelty + low reward)
- ✗ No formal justification for weighting scheme
- ✗ Hyperparameter tuning is ad-hoc

---

## 2. The Core Problem: Why Ranking Fails

### 2.1 Concrete Failure Mode

Consider a navigation task (e.g., continuous control to a goal):

**Scenario:**
- Trajectory A: Reward = 0.3, Novelty = 0.8, TD-error = 0.2
  - *Interpretation*: Moderate policy return, but explores high-value state/action regions; low surprise (well-predicted dynamics)
- Trajectory B: Reward = 0.25, Novelty = 0.2, TD-error = 0.95
  - *Interpretation*: Slightly lower return, familiar states, but one transition is very surprising (e.g., rare collision or reward spike)

**Under PER with $\alpha = 0.6$:**
- Priority(A) = $(0.2 + 0.01)^{0.6} \approx 0.34$
- Priority(B) = $(0.95 + 0.01)^{0.6} \approx 0.62$
- Result: B is sampled ~1.8x more often than A

**Problem:**
- A provides structured learning: consistent novelty + reasonable reward (useful for exploration and return improvement)
- B is a statistical outlier: one surprising step, but trajectory overall uninformative
- Replay budget wasted on replaying B repeatedly while A is under-sampled

**Constrained Allocation Solution:**
Set a threshold: $\theta_R = 0.2$ (minimum reward), $\theta_N = 0.4$ (minimum novelty), $\theta_\delta = 0.05$ (minimum TD-error for learning signal).

- A satisfies all thresholds; B fails (Novelty < 0.4)
- Only A enters the feasible set; B is excluded
- Result: Constrained learner focuses compute on high-quality trajectories, avoiding statistical noise

### 2.2 Why Thresholding Matters: The Supply-Demand Analogy

Think of replay as a resource allocation problem in operations research:

- **Supply**: Trajectories in buffer $D$ with various utility profiles
- **Demand**: Learner needs K transitions per gradient step
- **Market failure (ranking)**: No minimum quality standard; all goods (trajectories) acceptable at any price (probability)
  - ➜ Low-quality goods accumulate (waste); high-quality goods are under-supplied
- **Solution (constraints)**: Only goods meeting minimum specifications enter market; then allocate within high-quality pool

In replay:
- **Feasible set** $D^*$: Trajectories meeting minimum utility thresholds on *all* dimensions
- **Sampling within $D^*$**: Smooth distribution (e.g., exponential weighting) over quality candidates
- **Result**: No normalization waste; explicit quality guarantee; multi-objective optimization formalized

---

## 3. Formal Problem Formulation

### 3.1 The Optimization Objective

Let:
- $\tau \in D$: a trajectory in the replay buffer
- $p(\tau)$: empirical distribution over buffer (behavior policy frequency)
- $q(\tau)$: proposed replay distribution (policy variable)
- $\ell(\tau)$: loss on trajectory $\tau$ (e.g., policy gradient objective or value update)
- $\Delta \mathcal{L}(\tau)$: realized learning gain from replaying $\tau$ (observable post-update)
- $U(\tau) \in \mathbb{R}^d$: utility vector estimating learning value without needing post-update evaluation

**Unconstrained problem** (standard supervised learning):
$$
\max_{q} \mathbb{E}_{\tau \sim q}[\Delta \mathcal{L}(\tau)]
$$

This is NP-hard without additional structure. We introduce two natural constraints:

### 3.2 Constraint 1: Distribution Divergence (Off-Policy Stability)

Sampling radically different from the behavior policy $p$ breaks importance sampling correction and invalidates value estimates learned under $p$. We enforce:

$$
\text{KL}_5(q \| p) = \mathbb{E}_{\tau \sim q}\left[\log \frac{q(\tau)}{p(\tau)}\right] \le \epsilon_{KL}
$$

where $\epsilon_{KL}$ is a hyperparameter controlling divergence tolerance.

**Interpretation:**
- Small $\epsilon_{KL}$: Replay distribution must stay close to behavior distribution (conservative, stable)
- Large $\epsilon_{KL}$: More freedom to exploit high-reward regions

Typical range: $\epsilon_{KL} \in [0.1, 1.0]$ bits depending on buffer size and value function stability.

### 3.3 Constraint 2: Buffer Capacity (Feasible Set Constraint)

Sampling only from trajectories meeting a quality threshold limits budget waste. Define:

$$
D^* = \left\{ \tau \in D \mid U_i(\tau) \ge \theta_i, \quad \forall i \in [1, d] \right\}
$$

where $\theta_i$ is a per-dimension threshold and $d$ is utility dimensionality (typically $d=4$).

We require:
$$
|D^*| \ge \alpha |D|
$$

i.e., at least $\alpha$ fraction of buffer must remain feasible (default $\alpha = 0.2$). This prevents over-filtering.

### 3.4 Constrained Optimization Problem

$$
\boxed{
\max_{q \in \Delta} \mathbb{E}_{\tau \sim q}[\Delta \mathcal{L}(\tau)]
\quad \text{s.t.} \quad 
\text{KL}(q \| p) \le \epsilon_{KL}, \quad |D^*| \ge \alpha |D|
}
$$

where $\Delta$ is the set of all valid probability distributions on $D$.

### 3.5 Connection to Information Theory

This formulation is a constrained divergence problem, similar to:

- **Rate-distortion theory**: Maximize information from lossy compression
- **Inverse RL**: Recover reward from demonstrations (constrained likelihood)
- **Variational inference**: Match complex posterior with simpler variational approximation within KL budget

**Key difference**: We're *pushing away* from $p$ within a KL budget, not staying close—we want to exploit high-value regions while remaining stable.

---

## 4. Why Multiple Utility Dimensions Are Essential

### 4.1 Reward Alone Is Insufficient

- **Issue**: High reward ≠ High learning rate
  - Example: Consistently high-reward trajectory in a narrow policy region tells Agent nothing about:
    - How to recover from distractions (low-reward recovery test cases)
    - What happens outside the region (exploration)
    - Whether value estimates are correct
- **Solution**: Novelty and TD-error capture learning dynamics orthogonal to reward

### 4.2 TD-Error Alone Is Insufficient (PER's Blind Spot)

- **Issue**: Large TD-error ≠ High learning value
  - Example: Outlier transitions with extreme rewards (e.g., collision penalties) show high TD-error but may be:
    - Rare edge cases not generalizable
    - Artifacts of exploration noise (not learned behavior)
    - Unrepresentative of training distribution
- **Solution**: Combining TD-error with **stability metrics** (reward, novelty) identifies reliable learning signals

### 4.3 The Four-Dimensional Utility Vector

We propose a minimal sufficient set of utility dimensions:

$$
U(\tau) = [R(\tau), \, \mathcal{N}(\tau), \, \Delta V(\tau), \, G(\tau)]^\top
$$

| Component | Motivation | Formula |
|-----------|-----------|---------|
| **Reward** $R(\tau)$ | Exploitation: trajectories closer to goal are higher-return | $\frac{1}{T} \sum_{t=0}^{T} \gamma^t r_t$ |
| **Novelty** $\mathcal{N}(\tau)$ | Exploration: state/action regions with high prediction error need more rehearsal | $\frac{1}{T} \sum_{t=0}^{T-1} \| z_{t+1} - g_\theta(z_t, a_t) \|^2$ |
| **Value Signal** $\Delta V(\tau)$ | Learning: trajectories where value predictions are wrong benefit most from replay | $\frac{1}{T} \sum_{t=0}^{T-1} \left\| r_t + \gamma V(z_{t+1}) - V(z_t) \right\|$ |
| **Goal Proximity** $G(\tau)$ | (Sparse rewards only) Trajectories closer to goal frame better learning | $-\| z_T - z_{goal} \|_2$ |

**Each dimension is orthogonal**:
- Reward and Novelty can be uncorrelated (high-return regions are often well-explored and low-novelty)
- TD-error and Reward can be uncorrelated (well-predicted trajectories may still have suboptimal returns)
- Goal proximity provides sparse-reward guidance independent of dense metrics

### 4.4 Why Thresholding Works

By thresholding on *all* dimensions simultaneously, we select trajectories that are:
- Good enough in exploitation (sufficient reward)
- Good enough in exploration (sufficient novelty/epistemic uncertainty)
- Good enough in learning (sufficient value signal)
- Good enough in sparse-reward navigation (if applicable)

Any trajectory failing one dimension is excluded, avoiding "mediocre everywhere" samples.

---

## 5. Solution: Constrained Multi-Objective Replay (TCR)

### 5.1 High-Level Approach

1. **Compute utility vector** $U(\tau)$ for each trajectory (using existing forward passes—zero computational overhead)
2. **Maintain adaptive thresholds** $\theta_i$ that track percentiles of $U_i$ across recent buffer (ensures $|D^*|$ stays in feasible range)
3. **Define feasible set** $D^*$ as all trajectories satisfying $U_i(\tau) \ge \theta_i$ for all $i$
4. **Sample from $D^*$** using exponential weighting by learned utility weights $w^\top U(\tau)$
5. **Apply importance sampling correction** to account for $q \ne p$ divergence

### 5.2 Proxy Approximation: Learning the Weights

Since $\Delta \mathcal{L}(\tau)$ (true learning gain) is not available until post-training, we use a learned linear proxy:

$$
\Delta \mathcal{L}(\tau) \approx w^\top U(\tau) + \text{noise}
$$

where $w \in \mathbb{R}^d$ are utility weights learned (or fixed a priori) to maximize correlation with empirical learning gains.

**Validation (Mandatory Experiment):**
$$
\text{corr}(w^\top U(\tau), \widehat{\Delta \mathcal{L}}(\tau)) > 0.3 \quad \text{(Pearson correlation, Phase 6)}
$$

This validation is **not optional**—it's the ground truth of whether multi-objective replay is justified.

### 5.3 The Full Algorithm (High-Level)

**Initialization**: Replay buffer $D$, threshold vector $\theta$, utility weights $w$, EMA statistics (for normalization)

**At each environment step**:
1. Collect trajectory $\tau$, compute $U(\tau)$ from environment observations and value estimates
2. Store $(\tau, U(\tau))$ in buffer
3. [Periodic] Update thresholds $\theta_i$ to maintain percentiles of $U_i$ (e.g., 70th)
4. [Learning step] Recompute feasible set $D^*$ based on current $\theta$
5. Sample trajectories from $D^*$ via exponential distribution: $q(\tau) \propto \exp(w^\top U(\tau))$
6. Apply importance sampling correction weights: $w_{IS}(\tau) = \text{clip}(p(\tau) / q(\tau), w_{min}, w_{max})$
7. Update model with IS-weighted gradient: $\mathcal{L}_{total}(\tau) = w_{IS}(\tau) \ell(\tau)$

---

## 6. Positioning in Literature

### 6.1 Relationship to Prioritized Experience Replay (PER)

| Aspect | PER | TCR |
|--------|-----|-----|
| **Sampling paradigm** | Ranking (soft, all-have-mass normalization) | Thresholding + weighted sampling within feasible set (hard constraint + soft allocation) |
| **Objective** | Maximize expected immediate loss reduction | Maximize expected learning gain s.t. KL divergence and buffer utilization constraints |
| **Utility signal** | Single (TD-error) | Multiple (reward, novelty, value signal, goal proximity) |
| **Quality guarantee** | None (outliers can be over-represented) | Filtered set $D^*$ meets minimum thresholds |
| **Computational overhead** | Low ($\mathcal{O}(\log N)$ tree-based sampling) | Low ($\mathcal{O}(1)$ alias method; thresholds is $\mathcal{O}(n)$ periodically) |
| **Off-policy stability** | IS-weighting post-hoc | KL constraint built into problem formulation |
| **Complexity** | Simple; well-understood | Moderate; novel hyperparameter set ($\theta_i$, $w$, $\alpha$) |

**TCR generalizes PER** when:
- All utility dimensions except TD-error are disabled ($\hat{\mathcal{N}}, \hat{\Delta V}, \hat{G} \to 0$)
- Thresholds are set to $-\infty$ (no filtering, feasible set = entire buffer)
- Weights collapse to PER's single exponential: $w^\top U \approx \alpha \log(\delta + \epsilon)$

### 6.2 Relationship to Curiosity-Driven and Uncertainty-Weighted Replay

**Curiosity-driven replay** (e.g., RND-based): Prioritizes high-prediction-error trajectories to encourage exploration.

- **Similarity**: TCR includes novelty ($\mathcal{N}$) as a utility dimension
- **Difference**: TCR balances exploration (novelty) with exploitation (reward) and learning (TD-error) via multi-dimensional thresholding, not single ranking

**Uncertainty-weighted replay** (e.g., ensemble-based): Uses model uncertainty to weight samples.

- **Similarity**: TCR includes value signal ($\Delta V$, proxy for epistemic uncertainty)
- **Difference**: TCR formalizes the weighting within a constrained optimization framework with explicit learning-gain correlation

### 6.3 Relationship to Multi-Objective RL

**Multi-objective RL** (e.g., Pareto RL) solves:
$$
\max_\pi \mathbb{E}[\vec{r}^1, \vec{r}^2, \dots]
$$

for multiple reward signals.

- **Similarity**: TCR considers multiple utility signals (reward, novelty, etc.)
- **Difference**: TCR is a *sampling strategy*, not a policy learning method; utilities are designed for learning efficiency, not task objectives

---

## 7. Empirical Validation Roadmap

### 7.1 Mandatory Validation: The Correlation Proof

Before deploying TCR, we must establish that the proxy approximation holds:

$$
\text{Test 1 (Correlation Validity):} \quad \text{pearson\_corr}(w^\top U(\tau), \widehat{\Delta \mathcal{L}}(\tau)) > 0.3
$$

**Procedure**:
1. Train a policy with uniform replay on Atari-100k for 50K environment steps
2. At end of training, compute $\Delta \mathcal{L}(\tau)$ for all trajectories in replay buffer:
   - For each trajectory, roll out an imagined value update using that trajectory
   - Measure improvement to value loss
3. Compute utility vector $U(\tau)$ for each trajectory
4. Learn weights $w$ via linear regression: $\min_w \| w^\top U(\tau) - \widehat{\Delta \mathcal{L}}(\tau) \|^2$
5. Compute test correlation on held-out trajectories

**Success criterion**: Pearson $r > 0.3$ (moderate effect size, statistically significant)

### 7.2 Ablation Studies

Verify each utility dimension contributes:

1. **Ablate Reward**: Train with $U = [\hat{\mathcal{N}}, \widehat{\Delta V}, \hat{G}]$ (no reward)
2. **Ablate Novelty**: Train with $U = [\hat{R}, \widehat{\Delta V}, \hat{G}]$ (no exploration signal)
3. **Ablate Value Signal**: Train with $U = [\hat{R}, \hat{\mathcal{N}}, \hat{G}]$ (no learning signal)
4. **Ablate Goal Proximity**: Train with $U = [\hat{R}, \hat{\mathcal{N}}, \widehat{\Delta V}]$ (dense rewards only)
5. **No Constraints**: Train with $\theta_i = -\infty$ (full buffer always feasible) + full exponential weighting
6. **Static vs. Adaptive Thresholds**: Compare constant $\theta_i$ vs. adaptive EMA-based thresholds

**Expected outcomes**:
- All four dimensions improve performance (each contributes)
- Constraints (non-trivial thresholds) show gains over unconstrained ($\theta = -\infty$)
- Adaptive thresholds outperform fixed schedules

### 7.3 Benchmark Suite

**Primary benchmarks** (must include):
- **Atari-100k**: 26-game subset, 100K environment interactions
- **DeepMind Control Suite (DMC)**: 6 tasks (cheetah-run, walker-walk, quadruped-walk, jaco-reach, finger-spin, humanoid-run) at 1M steps
- **Procgen**: 16 games, offline policy learning (OOD generalization)

**Baselines**:
- Uniform replay (no prioritization)
- Prioritized Experience Replay (PER, $\alpha = 0.6$)
- DreamerV3 (recurrent replay, integrated world model)
- Oracle (replay distribution known to maximize learning gain; theoretical upper bound)

**Metrics**:
- Extrinsic reward vs. environment interactions
- Wall-clock compute (efficiency comparison)
- Replay latency (sampling time per batch)
- Feasible set size over time ($|D^*| / |D|$) [diagnostic]

### 7.4 Statistical Rigor

- Minimum 5 seeds per configuration
- Aggregate results using **Interquartile Mean (IQM)** via RLiable toolkit
- Report 95% confidence intervals via bootstrap (1000 samples)
- Significance testing: Paired t-tests for head-to-head comparisons

---

## 8. Why This Framing Matters

### 8.1 Intellectual Contribution

TCR reframes a solved problem (replay sampling) into a new paradigm:
- **From**: "Which single metric best ranks trajectories?" (PER's question)
- **To**: "How do we optimize multi-objective sampling under stability constraints?" (TCR's question)

This shift enables:
- Formal justification for multi-metric weighting (constrained optimization)
- Explicit quality thresholds (no low-utility samples)
- Separation of concerns (constraint satisfaction vs. objective optimization)

### 8.2 Practical Importance

1. **Zero computational overhead**: Utility vectors reuse existing forward passes
2. **Scalability**: Scales to replay buffers with 10M+ trajectories without performance degradation
3. **Modularity**: Works with any base RL algorithm (DQN, Policy Gradient, DreamerV3, etc.)
4. **Interpretability**: Utility dimensions are human-interpretable (reward, novelty, learning); weights $w$ can be analyzed

### 8.3 Broader Impact

- **Sample efficiency**: Better replay drives sample efficiency, reducing compute and wall-clock time for training
- **Industrial applicability**: More efficient learning reduces environmental/data collection costs
- **Foundational**: Could inform future work on multi-objective optimization in RL beyond replay

---

## 9. Next Steps

This problem statement establishes the conceptual and mathematical foundation for TCR:

1. **Formal Objective (PHASE 0.2)**: Extends this formulation with precise constraint handling
2. **Proxy Approximation (PHASE 0.3)**: Details the linear approximation $\Delta \mathcal{L} \approx w^\top U(\tau)$ and weight learning
3. **Utility Construction (PHASE 1)**: Specifies exact formulas for each utility component with EMA normalization
4. **Sampling Mechanism (PHASE 2)**: Implements constrained sampling with adaptive thresholds and feasible set maintenance
5. **Algorithm (PHASE 3)**: Full TCR pseudocode integrating all components
6. **Experiments (PHASE 6)**: Empirical validation of assumptions and benchmarking

---

## 10. Key Takeaways

| Point | Justification |
|-------|--|
| **Ranking is insufficient** | Single-metric sorting ignores multi-faceted learning dynamics |
| **Thresholding enables quality guarantee** | Explicit minimum utility thresholds filter low-value trajectories |
| **Multi-objective is natural** | Reward, exploration, and learning require orthogonal utility signals |
| **Constraints are formal** | KL-divergence + buffer saturation constraints justify the approach mathematically |
| **Correlation proof is mandatory** | We must validate proxy $w^\top U(\tau)$ correlates with actual learning gain |
| **TCR generalizes PER** | When thresholds → $-\infty$ and single-dimension, TCR reduces to PER |
| **Implementation is efficient** | Zero-overhead utility computation + $\mathcal{O}(1)$ sampling via alias method |

---

## References & Further Reading

### Core Papers
- Schaul et al. (2016): *Prioritized Experience Replay* — PER baseline
- Burda et al. (2019): *Exploration by Random Network Distillation* — Novelty-driven exploration
- Dreamer (2020), DreamerV3 (2023): World model-based RL with replay
- Puterman (1994): *Markov Decision Processes: Discrete Stochastic Dynamic Programming* — Resource allocation foundations

### Related Work
- Multi-objective RL (Pareto optimality)
- Importance sampling in off-policy learning
- Information-theoretic constraints on policy learning (KL divergence)
- Experience replay surveys (deep RL)

### Experimental Protocols
- RLiable: Reliable RL evaluation framework
- Atari-100k benchmark (sample efficiency)
- DeepMind Control Suite (continuous control)

---

**Author's Note**: This problem statement provides the conceptual bedrock. The following sections (PHASE 0.2 onward) will translate this vision into algorithms, code, and empirical validation.
