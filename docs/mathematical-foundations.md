# Mathematical Foundations

This document specifies the exact mathematical formulations implemented in `src/prism_gate/` and verified via `Hypothesis` property-based tests in `tests/`.

---

## 1. TCP-Vegas / Netflix Gradient2 Adaptive Concurrency Control

Static concurrency limits fail for LLM inference: set too low, GPU/API throughput is wasted; set too high during token-generation slowdowns, Little's Law ($L = \lambda W$) induces unbounded standing queues and cascading `p99` timeouts.

For each upstream endpoint $p$, let $\text{RTT}_{\min}$ denote the baseline no-queue round-trip latency and $\text{RTT}_t$ denote the latest observed sample at concurrency $I_t = \max(1, \text{inflight}_t)$.

The **RTT Gradient** $g_t \in (0, 1]$ and **Standing Queue Estimate** $\hat{Q}_t \ge 0$ are:

$$
g_t = \frac{\text{RTT}_{\min}}{\text{RTT}_t}, \qquad \hat{Q}_t = I_t \left(1 - \frac{\text{RTT}_{\min}}{\text{RTT}_t}\right)
$$

The concurrency ceiling $L_t \in [L_{\min}, L_{\max}]$ evolves according to:

$$
L_{t+1} = \text{clip}_{[L_{\min}, L_{\max}]} \begin{cases}
L_t \cdot \gamma & \text{if HTTP } 429 \text{ or } \ge 500 \\
L_t + \delta & \text{if } \hat{Q}_t < \alpha \text{ (under-queued)} \\
L_t & \text{if } \alpha \le \hat{Q}_t \le \beta \text{ (Little's Law knee)} \\
L_t \cdot \max(\gamma, \min(0.95, g_t)) & \text{if } \hat{Q}_t > \beta \text{ (standing queue buildup)}
\end{cases}
$$

where $\alpha = 2.5$, $\beta = 5.5$, $\delta = 1.0$, and $\gamma = 0.75$. Every $K = 64$ low-concurrency samples ($I_t \le L_{\min}$), $\text{RTT}_{\min} \leftarrow 1.02 \, \text{RTT}_{\min}$ relaxes upward to prevent stale transient micro-RTTs from permanently depressing $g_t$.

---

## 2. Weinberger Signed Feature-Hashing Semantic Cache

To achieve sub-100µs semantic similarity lookup without shipping multi-hundred-megabyte neural embedding weights, `TenantSemanticCache` implements **Signed Random Projections / Feature Hashing** (Weinberger et al., ICML 2009).

Given normalized content unigrams, sorted adjacent word bigrams, and character 4-grams $\{(f_i, w_i)\}_{i=1}^m$, two universal hash slices derived from `BLAKE2b-64` define:
- Bucket index $h(f) \in \{0, \dots, D-1\}$ ($D = 256$)
- Rademacher sign $\xi(f) \in \{-1, +1\}$

$$
v_j(x) = \sum_{i : h(f_i) = j} \xi(f_i)\, w_i, \qquad \hat{\mathbf{v}}(x) = \frac{\mathbf{v}(x)}{\|\mathbf{v}(x)\|_2}
$$

Because $\mathbb{E}[\xi(f)\xi(f')] = \mathbb{I}[f = f']$, the inner product is an unbiased estimator $\mathbb{E}[\langle \mathbf{v}(x), \mathbf{v}(y) \rangle] = \langle \mathbf{\phi}(x), \mathbf{\phi}(y) \rangle$ with variance $O(1/D)$, guarded by exact numeric set equality $\mathcal{N}(x) = \mathcal{N}(y)$ and negation parity $\pi_{\text{neg}}(x) = \pi_{\text{neg}}(y)$.

---

## 3. Structural Complexity & Speculative Cascade FinOps

Each prompt $x$ is mapped deterministically to $C(x) \in [0, 1]$:

$$
C(x) = \text{clip}_{[0, 1]}\Big(0.34\, s_{\text{reason}}(x) + 0.28\, s_{\text{code}}(x) + 0.20\, s_{\text{const}}(x) + 0.18\, s_{\text{hop}}(x)\Big)
$$

When $C(x) < \tau_c$, PrismGate executes speculatively on the fast tier ($c_{\text{fast}}$) and escalates to the frontier tier ($c_{\text{front}}$) iff confidence $\kappa < \kappa_{\min}$ or RAG faithfulness $F < 0.75$. Honest FinOps accounting charges **both** invocations on escalation:

$$
\text{Cost}_{\text{actual}}(r) = \begin{cases}
0 & \text{if semantic cache hit} \\
c_{\text{fast}}(r) & \text{if } C(x) < \tau_c \land \neg \text{escalated} \\
c_{\text{fast}}(r) + c_{\text{front}}(r) & \text{if } C(x) < \tau_c \land \text{escalated} \\
c_{\text{front}}(r) & \text{if } C(x) \ge \tau_c
\end{cases}
$$

---

## 4. Jensen-Shannon Divergence & Population Stability Index

### Base-2 Jensen-Shannon Divergence (`JSD`)
For discrete token or topic probability distributions $P, Q$ over joint support $\Omega$ with mixture $M = \frac{1}{2}(P + Q)$:

$$
\text{JSD}_2(P \| Q) = \frac{1}{2} \sum_{x : P(x) > 0} P(x) \log_2 \frac{P(x)}{M(x)} + \frac{1}{2} \sum_{x : Q(x) > 0} Q(x) \log_2 \frac{Q(x)}{M(x)}
$$

Proven invariants enforced by Hypothesis tests:
1. **Identity**: $\text{JSD}_2(P \| P) = 0$.
2. **Exact Symmetry**: $\text{JSD}_2(P \| Q) = \text{JSD}_2(Q \| P)$.
3. **Strict Boundedness**: $0 \le \text{JSD}_2(P \| Q) \le 1$ bit (attaining $1$ iff $\text{supp}(P) \cap \text{supp}(Q) = \emptyset$).

### Population Stability Index (`PSI`)
Given baseline empirical decile proportions $\hat{p}_k$ and live window proportions $\hat{q}_k$ smoothed via $\tilde{p}_k = \frac{\hat{p}_k + \epsilon}{1 + K\epsilon}$ ($\epsilon = 10^{-4}$):

$$
\text{PSI}(P, Q) = \sum_{k=1}^{K} (\tilde{q}_k - \tilde{p}_k) \ln\left(\frac{\tilde{q}_k}{\tilde{p}_k}\right) \ge 0
$$

Since $(a - b)\ln(a / b) \ge 0$ for all $a, b > 0$ with equality iff $a = b$, $\text{PSI}(P, Q) \ge 0$ and $\text{PSI}(P, P) = 0$ identically, triggering `MODERATE_DRIFT` at $\text{PSI} \ge 0.10$ and `CRITICAL_DRIFT` at $\text{PSI} \ge 0.25$.
