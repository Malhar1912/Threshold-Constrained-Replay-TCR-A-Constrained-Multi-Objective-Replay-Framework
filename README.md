# 📄 IMPLEMENTATION.md

# **Threshold-Constrained Replay (TCR): A Constrained Multi-Objective Replay Framework**

---

# 🧠 PHASE 0: Objective & Problem Formulation

## 0.1 Problem Reframing

Standard replay methods (e.g., Prioritized Experience Replay) treat sampling as a **ranking problem** based on a single proxy (e.g., TD-error). This leads to inefficient compute usage because:

- Low-utility samples are still replayed due to normalization.
- No minimum utility guarantee exists.

We instead model replay as a:

> **Constrained stochastic resource allocation problem**

## 0.2 Formal Objective

Let:
- $p(\tau)$: behavior distribution (buffer)
- $q(\tau)$: replay distribution
- $\Delta \mathcal{L}(\tau)$: empirical learning gain

We aim to solve:

$$
\max_{q \in \Delta} \mathbb{E}_{\tau \sim q}[\Delta \mathcal{L}(\tau)]
\quad \text{s.t.} \quad
\text{KL}(q \| p) \le \epsilon, \quad |D^*| \le B
$$

## 0.3 Proxy Approximation

Since $\Delta \mathcal{L}$ is not computable pre-selection:

$$
\Delta \mathcal{L}(\tau) \approx w^\top U(\tau)
$$

Thus:

$$
q^*(\tau) \propto p(\tau)\exp(w^\top U(\tau))
$$

## 0.4 Empirical Validation Requirement

We **must validate**:

$$
\text{corr}(w^\top U(\tau), \widehat{\Delta \mathcal{L}}(\tau)) > 0
$$

*This is a required experiment (not optional).*

---

# 🧭 PHASE 1: Utility Vector Construction (Zero Extra Compute)

Trajectory:

$$
\tau = \{(o_t, a_t, r_t)\}_{t=0}^T, \quad z_t = f_\theta(o_t)
$$

*All components are computed **using existing forward passes only**.*

## 1.1 Normalization

For each metric $X$:

$$
\hat{X} = \frac{X - \mu_X}{\sigma_X + \epsilon}
$$

with Exponential Moving Average (EMA) updates.

## 1.2 Utility Components

### (1) Reward
$$
R(\tau) = \frac{1}{T} \sum_{t=0}^T \gamma^t r_t
$$

### (2) Novelty (World Model Error)
$$
\mathcal{N}(\tau) = \frac{1}{T} \sum_{t=0}^{T-1} \| z_{t+1} - g_\theta(z_t, a_t) \|^2
$$

### (3) Value Sensitivity (TD Error)
$$
\Delta V(\tau) = \frac{1}{T} \sum_{t=0}^{T-1} \left| V(z_t) - (r_t + \gamma V(z_{t+1})) \right|
$$

### (4) Goal Proximity (Sparse Only)
$$
G(\tau) = -\| z_T - z_{goal} \|_2
$$
*(Disabled in dense reward settings)*

## 1.3 Final Utility Vector

$$
U(\tau) = [\hat{R}, \hat{\mathcal{N}}, \widehat{\Delta V}, \hat{G}]^\top
$$

---

# ⚙️ PHASE 2: Constrained Sampling Mechanism

## 2.1 Thresholding (Feasible Set)

$$
D^* = \{ \tau \mid U_i(\tau) \ge \theta_i, \quad \forall i \}
$$

## 2.2 Adaptive Threshold Update

$$
\theta_i \leftarrow (1-\eta)\theta_i + \eta \cdot \text{Percentile}_p(U_i)
$$

Typical hyperparameters:
- $p \in [60, 80]$
- $\eta \in [0.01, 0.05]$

## 2.3 Safety Constraint

If buffer constraints fail:

$$
|D^*| < \alpha |D|
$$

Then symmetrically relax thresholds:

$$
\theta_i \leftarrow \theta_i - \delta
$$

## 2.4 Sampling Distribution

$$
q(\tau) = \frac{\exp(w^\top U(\tau))}{\sum_{\tau' \in D^*} \exp(w^\top U(\tau'))}
$$

## 2.5 Importance Sampling Correction

$$
w_{IS}(\tau) = \text{clip}\left(\frac{p(\tau)}{q(\tau)}, w_{min}, w_{max}\right)
$$

Final loss application:

$$
\mathcal{L}_{total} = w_{IS}(\tau) \cdot \ell(\tau)
$$

---

# 🧮 PHASE 3: Algorithm (TCR)

### **Algorithm 1:** Threshold-Constrained Replay

**Initialize:** replay buffer $D$, thresholds $\theta$, weights $w$

**Loop over environments:**
1. **Collect trajectory $\tau$**
2. Compute utility vector $U(\tau)$ using available inferences.
3. Store transition/trajectory tuple $(\tau, U(\tau))$ in $D$.
4. **Periodic Threshold Update:**
   - Update $\theta_i$ using EMA and recent empirical percentiles.
5. Compute current feasible set $D^*$.
6. **Replay Phase:**
   - Sample trajectories from $D^*$ via target distribution $q(\tau)$.
7. **Model Update:**
   - Apply IS-weighted gradients to base model parameters.

---

# ⚡ PHASE 4: Systems Architecture (Compute Efficiency)

## 4.1 Asynchronous Pipeline
- **Actor:** collects trajectories + computes offline statistics $U(\tau)$
- **Learner:** dedicated exclusively to sampling ($q$) + training ($\nabla \theta$)

## 4.2 Memory Footprint

At 1,000,000 transitions:

| Component         | Precision | Size footprint |
| :---------------- | :-------- | :------------- |
| Utility Vectors   | `float16` | ~8 MB          |
| Feasibility Masks | `bool`    | ~1 MB          |

## 4.3 Sampling Efficiency
Because sets are dynamically cached, sampling requires minimal compute bounds:
- **Alias method:** $\mathcal{O}(1)$ time complexity per sample.
- **Sum-tree alternative:** $\mathcal{O}(\log N)$

---

# 🧬 PHASE 5: Latent Replay Mechanisms
*(Explicitly aligned with frameworks like DreamerV3)*

## 5.1 Forward Rollout

$$
z_{t+1} = g_\phi(z_t, a_t)
$$

## 5.2 Time Compression

Subsampling step interval ($k$):

$$
t \in \{0, k, 2k, \dots \}
$$

Alternatively, deploy multi-step jump models $g_k$.

## 5.3 Reverse Replay
Backward value propagation through generated latents:

$$
y_t = r_t + \gamma y_{t+1}
$$

---

# 📊 PHASE 6: Empirical Validation Protocol

## 6.1 Core Experiments

**MUST Include:**
1. Correlation Proof: $w^\top U(\tau)$ **vs** $\widehat{\Delta \mathcal{L}}(\tau)$
2. Sub-metric Ablations:
   - Remove individual utility axes ($\hat{R}$, $\hat{\mathcal{N}}$, $\widehat{\Delta V}$).
   - Remove global constraints (thresholds = 0).
   - Static threshold boundaries **vs** adaptive boundaries.

## 6.2 Target Benchmarks
- **Atari-100k:** (full suite or standard 26 subset)
- **DeepMind Control (DMC):** (6+ robust tasks)
- **Procgen:** (Evaluate OOD generalization capabilities)

## 6.3 Standard Baselines
- Uniform replay
- Prioritized Experience Replay (PER)
- DreamerV3 Native (Recurrent Replay)
- Multi-metric routing (no feasible bounds / no constraints)

## 6.4 Primary Metrics
- Extrinsic reward vs. environment interactions
- Extrinsic reward vs. wall-clock compute (efficiency benchmark)
- Cache insertion/Replay efficiency latency logs

## 6.5 Statistical Rigor Framework
- Minimal $n \ge 5$ varying seeds
- Evaluate using Interquartile Means (IQM, via `RLiable`)
- Confidence intervals aggregated through bootstrap bounds (95% CI)

---

# 🛡️ PHASE 7: Failure Modes & Safeguards

## 7.1 Over-filtering
- **Symptom:** Disconnected learning curve; $D^*$ starves batch processing.
- **Correction:** Implement a unified safety distribution limit:
$q' = \lambda q + (1 - \lambda)\mathcal{U}$  
*(where $\mathcal{U}$ is the standard Uniform distribution).*

## 7.2 Mode Collapse
- **Symptom:** Buffer exploits very few hyper-rare behaviors.
- **Correction:** Establish hard thresholding boundaries on absolute coverage limitations; continuous Entropy monitoring across sampling layers.

## 7.3 Unaccounted Policy Bias
- **Symptom:** $q \gg p$ divergence shatters the trusted value manifold.
- **Correction:** Rely directly on calculated Importance Sampling constraint weights.

## 7.4 Metric Redundancy & Collapse
- **Symptom:** Covariant utilities blow up vector space ranges.
- **Correction:** Implement mandatory $N(0,1)$ Normalization directly routed via ongoing Exponential Moving Averages (EMA). Optionally prune $G(\tau)$ directly if base settings naturally include density returns.

---

# 🏁 FINAL POSITIONING

## Expected Contribution Line
> "A mathematically robust constrained multi-objective replay framework equipped with adaptively scaled thresholds alongside formally validated utility proxies."

## Paper's Core Strengths (Sell Sheet)
1. Demands strictly **Zero extra computational latency overhead**.
2. Scales asymptotically and painlessly toward extremely large structural bounds ($N > 1e7$).
3. Establishes itself gracefully across distinct architectural configurations (off-policy vs recurrence/latent).
4. Employs direct, computationally quantifiable and verifiable objective validations (Correlative validity).

## Core Requirements for Top-Tier Acceptance
- Strong observable metric correlation between predicted and empirical $\Delta \mathcal{L}$.
- Unquestionably prominent graphical benefits via Ablation derivations.
- Top percentile performance markers placed beside deeply integrated base SOTAs like Dreamer.
