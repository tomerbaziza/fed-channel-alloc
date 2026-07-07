# Hyperparameter Alignment — Table II (Paper) vs Code

| Parameter | Paper (Table II) | Before fix | After fix |
|-----------|------------------|------------|-----------|
| ρ (personal reward) | **0.7** | 0.6 | **0.7** ✓ |
| r_desired | 4 | 4 | ✓ |
| ζ (quality threshold) | 0.9 | 0.9 | ✓ |
| c₁ (stay bonus) | 1.1 (+10%) | 1.1 | ✓ |
| Γ (neighbor dist) | 500 m | 500 m | ✓ |
| B (episodes) | 1000 | 1000 | ✓ |
| T (decisions/net) | 20 | N×20+N steps | ✓ |
| S_z (global replay) | 10⁵ | 10⁵ | ✓ |
| N_E (train steps/ep) | 40 | 40 | ✓ |
| bz (batch) | 32 | 64 default | **32** ✓ |
| γ | 0.9 | 0.9 | ✓ |
| Huber δ | 1 | 1 | ✓ |
| ω (MellowMax) | 0.02 / 0.2 @ B/2 | bug → **2.0** | **0.02 / 0.2** ✓ |
| LR | 0.00025 / 0.0001 @ B/2 | same intent | ✓ |
| Adam ε | 10⁻⁷ | 10⁻⁸ default | **10⁻⁷** ✓ |
| ε exploration | 0.5→0.01 @ B/2 | 0.5→0.01 | ✓ |
| K channels | 10 | 10 | ✓ |
| M users/net | {1,…,15} | {2,…,21} | **{1,…,15}** ✓ |
| NN | 3×128, skip×2, LeakyReLU 0.2 | same | ✓ |

**Completed runs** (centralized PyTorch + TF baseline) used **ρ=0.6** and **ω=2.0** after ep 500. Re-run with current code for strict paper parity.
