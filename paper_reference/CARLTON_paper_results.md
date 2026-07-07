# CARLTON Paper — Experimental Results Reference

**Paper:** SINR-Aware Deep Reinforcement Learning for Distributed Dynamic Channel Allocation in Cognitive Interference Networks  
**Authors:** Yaniv Cohen, Tomer Gafni, Ronen Greenberg, Kobi Cohen  
**Source:** [arXiv:2402.17773](https://arxiv.org/abs/2402.17773) | [PDF](https://arxiv.org/pdf/2402.17773)  
**Algorithm:** CARLTON (Channel Allocation RL To Overlapped Networks)

> **Purpose of this file:** Baseline reference for comparing future runs of this repository (including the PyTorch Federated RL refactor) against the published CARLTON results.

---

## 1. Headline claims (Abstract & Section I-B)

| Claim | Reported value | Notes |
|-------|----------------|-------|
| Superiority over Random Agent (RA) | **~45%** margin | Approximate performance margin |
| Superiority over JAR (state-of-the-art baseline) | **~20%** margin | Approximate performance margin |
| Gap vs centralized graph-coloring upper bound | **~2.5%** only | In-sample domain (#Networks < 7) |
| Training paradigm (paper) | **CTDE** | Centralized Training, Decentralized Execution |
| Learner (paper) | **DeepMellow** | Value-based DRL with MellowMax backup |
| Generalization | Strong **out-of-sample** (#Networks > 7) | Trained up to 7 networks, tested 2–15 |

---

## 2. Simulation setup (Section II-B)

| Setting | Paper value |
|---------|-------------|
| Networks per scenario `N` | Randomly sampled each episode |
| Users per network `M^n` | Uniform **2–22** in text; Table II lists **{1,…,15}** for experiments |
| User spatial spread | Multivariate Gaussian around network center |
| Covariance | Diagonal `[50², 50²]` m² |
| Network center placement | Algorithm 1 (random polar offset from existing center) |
| Center range parameter `u₁` | **400 m** (Table II) |
| Radius range `x₁, x₂` | **50–500 m** (Table II) |
| Network manager | User minimizing total intra-network Euclidean distance |
| Scenario horizon | **`T × N`** steps; each network gets **`T = 20`** decision points |
| Training episodes `B` | **1000** |
| Evaluation sweep | **420 games** total (30 games × scenarios with **2–15 networks**) |
| In-sample training domain | Up to **7 networks** |
| Out-of-sample test domain | **8–15 networks** |

---

## 3. Physical / wireless parameters (Section II-A, Table I–II)

### Channels & RF

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Number of channels | `K` | **10** |
| Carrier frequencies | `{f₁,…,f_K}` | **208–226 MHz**, step **2 MHz** |
| Channel bandwidth | `B` | **2 MHz** (implied by thermal noise example) |
| Antenna gains | `G_R, G_T` | **1** |
| Antenna heights | `H_R, H_T` | **1 m** |
| Transmit power | `PT` | **2 dBW** |
| SINR QoS threshold | `SINR*` | **4 dBm** |
| Thermal noise example | `I_T` | **−104.9 dBm** (room temp, 2 MHz, NF=6 dB) |

### Spectral attenuation (Table I)

| Spectral distance `\|k − k̃\|` | Attenuation (dBm) |
|------------------------------|-------------------|
| 0 | 0 |
| 1 | 20 |
| 2 | 40 |
| 3 | 50 |
| 4 | 60 |
| ≥ 5 and `\|f_k − f_k̃\|/f_k ≤ 0.05` | 95 |
| else | 110 |

---

## 4. RL / CARLTON hyperparameters (Table II)

| Parameter | Value |
|-----------|-------|
| Personal reward weight | `ρ = 0.7` |
| Desired reward | `r_desired = 4` |
| Stay bonus constant | `c₁ = 1.1` |
| Quality threshold | `ζ = 0.9` |
| Neighbor distance threshold | `Γ = 500 m` |
| Policy mixing | `α = 0`, `β = 1` |
| Exploration `ε_b` | **0.5 → 0.01** over first **B/2** episodes, then fixed |
| Huber loss `δ` | **1** |
| Discount `γ` | **0.9** |
| MellowMax `ω` | **0.02** if episode `i ≤ N/2`, else **0.2** |
| Learning rate | **0.00025** if `i ≤ N/2`, else **0.0001** |
| Optimizer | Adam (`β₁=0.9`, `β₂=0.999`, `ε=10⁻⁷`) |
| Global replay size | `S_z = 10⁵` |
| Local replay / decisions per net | `T = 20` |
| Training gradient steps per episode | `N_E = 40` |
| Batch size | `bz = 32` |
| NN hidden layers | **3** × **128** nodes |
| Skip connections | **2** |
| Hidden activation | Leaky-ReLU slope **0.2** |
| Output activation | Identity |
| Weight init | Glorot-Uniform; bias **0** |

---

## 5. Reward design (Section III-C)

| Component | Definition |
|-----------|------------|
| Personal reward | Algorithm 2 — rank-based on `QV` of chosen channel; `r_desired` if `v ≥ ζ`; stay bonus `× c₁` |
| Social welfare | Algorithm 3 — mean personal reward of neighbors within `Γ` |
| Total reward | **Eq. (13):** `r = ρ·r_p + (1−ρ)·r_sw` |

**Note for this repo:** Code uses **`ρ = 0.7`** per Table II / Eq. (13) (`RHO` in `Coordinator.py`).

---

## 6. Evaluation metrics (Section IV-A, Eq. 18–22)

Use these definitions when comparing against paper figures.

### Channel quality (Eq. 18)

At final scenario time `T`:

- **CQ** = `[QV¹_{c¹(T)}, …, QV^N_{c^N(T)}]` — quality on each network's final channel
- **CQ_mean** = `E[CQ]` (mean over networks / scenarios)
- **CQ_median** = `Median(CQ)`
- **min_CQ** = `min(CQ)`

### Convergence & mobility

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **ANCC** | Avg channel changes until convergence per network | Lower is better |
| **ANCCS** | `1 − ANCC/T` (Eq. 19) | Higher is better |
| **CT** | Time index of last channel change in scenario | Lower is faster convergence |
| **CTS** | `1 − CT/(T·N)` (Eq. 20) | Higher is better |

### Spectrum efficiency (Eq. 21)

- **Ψ^n** = `( Σ_k QV^n_k(T)² )^0.5 / K^0.5`
- **SES** = `E[Ψ^n]` averaged over networks

### Weighted score (Eq. 22) — primary composite metric

```
WS = 0.4·CQ_mean + 0.1·ANCCS + 0.4·CTS + 0.1·SES
```

Secondary composite used in several figures:

```
E[(CQ + min_CQ) / 2]
```

### Training reward benchmark

- **Perfect-game accumulated reward ceiling:** **88** (Figure 3 caption)

---

## 7. Section IV-A — CARLTON training results

### 7.1 Action masking (Figure 3)

| Condition | Finding |
|-----------|---------|
| With vs without masking | Masking gives **higher accumulated reward early** (episodes ≤ 400) |
| Late training (≥ 500) | **Comparable** final performance with/without masking |
| Perfect game reference | Max accumulated reward ≈ **88** |

### 7.2 Channel quality during training (Figure 4)

| Metric | Reported trend |
|--------|----------------|
| CQ_mean | Improves during training |
| CQ_median | Improves during training |
| min_CQ | Improves; at convergence some agents stay **> 0.95** |

### 7.3 Physical scores during training (Figure 5)

| Metric | Reported trend |
|--------|----------------|
| ANCCS | Improves (reward encourages stability) |
| CTS | Improves |
| WS | Improves |
| SES | ~**0.8** average at convergence (not directly optimized) |

### 7.4 Sensitivity — personal reward weight ρ (Figures 6–7)

Test: **420 games**, 30 per network-count scenario, **N ∈ {2,…,15}**

| ρ | WS (generality) | CQ / min_CQ |
|---|-----------------|-------------|
| Higher ρ | Better WS, faster/aggressive | — |
| Lower ρ | — | Better fairness/cooperation |
| **ρ = 0.7** | **Best balance** (paper choice) | Good trade-off |

### 7.5 Sensitivity — neighbor threshold Γ (Figures 8–9)

| Γ (m) | WS performance | E[(CQ+min_CQ)/2] |
|-------|----------------|------------------|
| **500** | **Best average WS** (~50% mutual QV influence probability) | Strong |
| **400** | — | **Best average** E[(CQ+min_CQ)/2] (~86% influence probability) |
| **0** (selfish) | **Worst** across scenarios | Worst |

Reference probabilities cited in paper:
- ED = **700 m** → ~**3%** mutual QV influence
- ED ≤ **500 m** → ~**50%** influence
- ED ≤ **400 m** → ~**86%** influence

### 7.6 Post-processing threshold φ (Figures 10–11)

Rule: keep current channel unless new channel CQ exceeds current by at least **φ%**.

| φ | WS | Channel quality |
|---|-----|-----------------|
| Higher φ | **Increases WS** (fewer useless switches) | **Decreases** CQ |
| φ = None (baseline) | Lower WS | Higher CQ |
| Trade-off | Faster convergence vs peak quality | User/system dependent |

---

## 8. Section IV-B — Algorithm comparison (Figures 12–15)

### Baselines compared

| Algorithm | Description |
|-----------|-------------|
| **CARLTON** | Proposed MARL (with optional post-processing φ) |
| **JAR** | Jamming Avoidance Response — switches ±2 MHz only if CQ gain ≥ **0.05** |
| **RA** | Random Agent — random initial channel, no further switches |
| **Graph coloring** | Centralized upper bound (not distributed) |

### Reported comparative findings

| Comparison | Result |
|------------|--------|
| CARLTON vs RA | CARLTON **wins** on WS and E[(CQ+min_CQ)/2] (~**45%** margin, Section I-B) |
| CARLTON vs JAR | CARLTON **wins** (~**20%** margin, Section I-B) |
| CARLTON vs graph coloring | Only **~2%** gap in-sample (#Networks < 7) |
| CARLTON CTS vs baselines | **Lower CTS** without post-processing (φ=None) — better CQ costs longer convergence |
| Out-of-sample (#Networks > 7) | CARLTON still strong; cooperates mainly with **neighbors (<7 effective)** |

Figure 15 reports average **E[(CQ+min_CQ)/2]** across all 420 scenarios per algorithm (CARLTON shown with **φ=0**).

> **Note:** Exact numeric curves are in paper figures 3–15; this file captures **reported trends and headline numbers**. Reproduce plots from logged metrics for pixel-level comparison.

---

## 9. Mapping paper metrics → this repository

| Paper metric | Repo function / field |
|--------------|----------------------|
| CQ_mean, min_CQ, median | `Utils/ScenarioExamination.channelQuality()` via `get_game_performamce()` |
| ANCC, ANCCS | `clacANCC()` |
| CT, CTS | convergence time helpers in `ScenarioExamination.py` |
| SE, SES | spectrum efficiency in `get_game_performamce()` |
| WS | weighted score in `get_game_performamce()` (verify weights match Eq. 22) |
| Accumulated episode reward | `Coordinator` → `average_accumulated_reward_val` |
| Channel changes | `Agent.change_channel_counter` |

**Repo vs paper training difference:**

| Aspect | Paper CARLTON | This repo (FRL refactor) |
|--------|---------------|---------------------------|
| Aggregation | Global replay buffer (GRM) + centralized NN updates | **FedAvg** over local `state_dict()` |
| ρ in reward | 0.7 | **0.6** in `Coordinator.py` |
| Episodes / rounds | 1000 episodes | Configurable `communication_rounds` in `main_v3.py` |

---

## 10. Future comparison template

Fill this table when running experiments on this codebase.

### 10.1 Run configuration

| Field | Paper | Our run |
|-------|-------|---------|
| Date | — | |
| Commit / branch | — | |
| Training mode | CTDE + GRM | FRL + FedAvg |
| Episodes / comm. rounds | 1000 | |
| Networks trained | ≤ 7 | |
| Networks tested | 2–15 | |
| Games per setting | 30 | |
| K channels | 10 | |
| ρ | 0.7 | |
| Γ (m) | 500 | |
| Masking | yes/no | |
| Post-process φ | 0 / None | |

### 10.2 Primary metrics (mean over 420 games or equivalent)

| Metric | Paper (qualitative / headline) | Our result | Δ |
|--------|--------------------------------|------------|---|
| WS | Best among distributed baselines | | |
| E[(CQ+min_CQ)/2] | Best among distributed baselines | | |
| CQ_mean | High at convergence | | |
| min_CQ | > 0.95 some agents at convergence | | |
| ANCCS | Improves with training | | |
| CTS | Lower than JAR/RA without φ | | |
| SES | ~0.8 | | |
| Accumulated reward (train) | → ~88 perfect game | | |

### 10.3 Baseline comparison (Section IV-B style)

| Algorithm | WS | E[(CQ+min_CQ)/2] | CTS |
|-----------|-----|------------------|-----|
| CARLTON (paper) | ref | ref | ref |
| CARLTON (our FRL) | | | |
| JAR | lower | lower | higher |
| RA | lowest | lowest | — |
| Graph coloring | ~+2% vs CARLTON in-sample | upper bound | — |

### 10.4 Acceptance criteria (suggested)

- [ ] WS within **±X%** of paper CARLTON on same scenario set  
- [ ] min_CQ > **0.90** at convergence (paper reports > 0.95 for some agents)  
- [ ] Outperforms RA and JAR on WS in ≥ **80%** of network counts  
- [ ] In-sample gap to graph coloring **≤ 5%** (paper ~2.5%)  

---

## 11. Paper figures index

| Figure | Content |
|--------|---------|
| Fig. 3 | Accumulated training reward — masking vs no masking |
| Fig. 4 | CQ_mean, CQ_median, min_CQ vs episode |
| Fig. 5 | ANCC, CT, SE, WS vs episode |
| Fig. 6 | WS vs #networks for different ρ |
| Fig. 7 | E[CQ], E[min_CQ] vs #networks for different ρ |
| Fig. 8 | WS vs #networks for different Γ |
| Fig. 9 | E[(CQ+min_CQ)/2] vs #networks for different Γ |
| Fig. 10 | WS vs #networks for different φ |
| Fig. 11 | E[(CQ+min_CQ)/2] vs #networks for different φ |
| Fig. 12 | WS — algorithm comparison |
| Fig. 13 | E[CQ], E[min_CQ] — algorithm comparison |
| Fig. 14 | CTS — algorithm comparison |
| Fig. 15 | Bar: average E[(CQ+min_CQ)/2] per algorithm |

---

## 12. Key equations quick index

| Eq. | Topic |
|-----|-------|
| (2)–(7) | SINR, path loss, interference, thermal noise |
| (10) | Optimization objective |
| (11)–(12) | BSINR, Quality Vector QV |
| (13) | Total reward |
| (14)–(16) | Policy, masking, Huber + MellowMax loss |
| (17) | State = concat(CBR, QV) |
| (18)–(22) | CQ, ANCCS, CTS, SES, WS |

---

*Last updated from arXiv:2402.17773v1 HTML source. Use this file as the canonical benchmark when validating the federated PyTorch implementation.*
