# Paper Baseline vs Our Implementation

## Baseline (original authors' code)

| Item | Value |
|------|-------|
| **Folder** | `C:\Users\User\carlton-paper-baseline` |
| **GitHub** | https://github.com/Kobi-Cohen-researchlab/fed-channel-alloc |
| **Commit** | `aebf6ff` (main — TensorFlow/Keras, centralized CTDE) |
| **Python** | **3.11** (TensorFlow 2.16 does not support 3.14) |
| **Entry** | `main_v3.py` (1000 episodes) or `run_baseline_smoke.py` (20 episodes) |

### Setup

```powershell
cd C:\Users\User\carlton-paper-baseline
py -3.11 -m pip install -r requirements.txt
py -3.11 -u run_baseline_smoke.py          # quick validation (20 ep)
py -3.11 -u main_v3.py                     # full paper training (1000 ep)
```

### Results

- Smoke test report: `carlton-paper-baseline/baseline_results/smoke_report.json`
- Smoke log: `carlton-paper-baseline/baseline_smoke_log.txt`
- Full training: `train_info.pk`, `Train_weights_/`, plots via `create_images`

**Smoke test (20 ep, 2026-06-30):** reward 39.9 → 72.6, last-10 mean **65.6**, cc last-10 **3.6** — learning OK.

### Environment notes

- Set `TF_USE_LEGACY_KERAS=1` (see `run_smoke.bat`) — required for `Nets_keras.py` on TF 2.16+
- Use **Python 3.11** (`py -3.11`), not 3.14

## Our fork (modified)

| Item | Value |
|------|-------|
| **Folder** | `C:\Users\User\fed-channel-alloc` |
| **Centralized** | `main_centralized.py` (PyTorch CTDE) |
| **Federated** | `main_v3.py` (FedAvg FRL) |
| **Python** | 3.14 + PyTorch |

## Comparison metrics (from paper Section IV)

Use the same scenarios from `scenarios_for_test/` and `compare_to_paper.py` / `Inference_step_on_test_cases`.

| Metric | Paper target |
|--------|----------------|
| Perfect-game reward ceiling | ~88 |
| ANCC, CT, CQ, SES | see `paper_reference/CARLTON_paper_results.md` |

## Patches applied to baseline clone

Minimal fixes so the repo runs on current Windows/Python (not algorithm changes):

1. `SimulationEnvironments/Egli.py` — `pdist` scalar fix
2. `BuildingBlocks/Coordinator.py` — `np.reshape` without deprecated `newshape` kwarg
3. Added `requirements.txt`, `run_baseline_smoke.py`, `run_smoke.bat`
4. `Nets_keras.py` — residual add via `tf.keras.layers.Add` (Keras 3 compat)
5. `TF_USE_LEGACY_KERAS=1` for TensorFlow 2.16+
