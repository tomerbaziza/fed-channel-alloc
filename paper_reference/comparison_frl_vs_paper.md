# FRL Run vs CARLTON Paper — Comparison Report

Generated: 2026-05-24T15:40:48

## 1. Setup differences (important)

| Aspect | Paper CARLTON | This FRL run |
|--------|---------------|--------------|
| Training length | 1000 episodes | 20 federated rounds |
| Aggregation | Global replay + centralized NN update | FedAvg over local `state_dict` |
| Reward ρ | 0.7 | 0.6 (code default) |
| Eval games | 420 | 10 post-train scenarios |

## 2. Training run (logged by main_v3)

| Metric | Our run | Paper reference |
|--------|---------|-----------------|
| Mean accumulated reward / round | 49.85 | Perfect game ≈ 88 (Fig. 3) |
| Reward range | 17.87 – 73.58 | — |
| Mean reward (rounds 1–10) | 43.12 | — |
| Mean reward (rounds 11–20) | 56.57 | — |
| % of paper perfect-game ceiling | 56.6% | 100% at convergence |
| Mean channel changes / agent | 8.60 | Lower ANCC is better; ANCCS = 1−ANCC/T |
| Channel changes (1st half) | 13.01 | — |
| Channel changes (2nd half) | 4.20 | Paper: ANCCS improves with training |

## 3. Post-train evaluation (paper-aligned metrics)

Computed via `get_game_performamce()` (WS uses repo weights 0.4 CQ + 0.4 CT + 0.1 ANCC + 0.1 SE).

| Metric | Our eval (mean) | Paper target / trend | Assessment |
|--------|-----------------|----------------------|------------|
| WS | 0.658 | Best among distributed baselines (Fig. 12) | Qualitative only — no numeric paper WS in text |
| CQ_mean | 0.665 | Improves to high values at convergence (Fig. 4) | Below typical converged CQ |
| min_CQ | 0.312 | > 0.95 for some agents at convergence | Below paper (>0.95) |
| ANCCS | 0.837 | Improves during training (Eq. 19) | Improving |
| CTS | 0.572 | Improves (Eq. 20); lower CTS vs JAR without φ | — |
| SES (SE) | 0.792 | ~0.8 at convergence (Sec. IV-A) | Near paper |
| E[(CQ+min_CQ)/2] | 0.488 | Primary comparison metric (Figs 7, 13, 15) | — |

## 4. Headline paper claims (not directly reproduced here)

| Claim | Paper | This run |
|-------|-------|----------|
| vs Random Agent | ~+45% | **Not tested** (no RA baseline run) |
| vs JAR | ~+20% | **Not tested** (no JAR baseline run) |
| vs graph coloring | ~−2.5% gap | **Not tested** |

## 5. Per-scenario eval detail

| N nets | Reward | CQ_mean | min_CQ | WS | ANCCS | CTS | SE |
|--------|--------|---------|--------|-----|-------|-----|-----|
| 6 | 47.2 | 0.787 | 0.222 | 0.852 | 0.984 | 0.904 | 0.772 |
| 4 | 31.8 | 0.688 | 0.250 | 0.838 | 0.976 | 0.928 | 0.948 |
| 2 | -12.0 | 0.583 | 0.500 | 0.820 | 1.000 | 0.976 | 0.967 |
| 2 | 73.3 | 1.000 | 1.000 | 0.985 | 1.000 | 0.976 | 0.949 |
| 7 | 46.6 | 0.662 | 0.000 | 0.368 | 0.469 | 0.000 | 0.565 |
| 3 | 24.0 | 0.535 | 0.273 | 0.765 | 0.984 | 0.952 | 0.715 |
| 6 | 36.7 | 0.726 | 0.190 | 0.426 | 0.627 | 0.000 | 0.728 |
| 7 | 38.2 | 0.710 | 0.333 | 0.389 | 0.531 | 0.000 | 0.518 |
| 7 | 43.5 | 0.694 | 0.250 | 0.442 | 0.803 | 0.007 | 0.810 |
| 2 | -12.0 | 0.264 | 0.100 | 0.691 | 1.000 | 0.976 | 0.954 |

## 6. Summary verdict

- **Spectrum mobility:** Second-half channel changes lower than first half — consistent with paper trend (fewer switches as policy stabilizes).
- **Reward scale:** Mean reward is 57% of paper perfect-game ceiling (88) — expected with only 20 rounds vs paper 1000 episodes.
- **min_CQ:** Below paper's reported >0.95 at full convergence; more training/eval games needed.
- **Baselines:** To match paper Section IV-B, run RA, JAR, and graph-coloring on the same 420-game protocol.
- **Protocol:** Paper uses CTDE+GRM; this repo uses FRL+FedAvg — not apples-to-apples without aligning training length and ρ=0.7.
