# Centralized Training: PyTorch Fork vs Paper TensorFlow Baseline

Both runs: 1000 episodes, CTDE, 10 channels, global replay + `train_model`.

| Metric | Paper TF (`carlton-paper-baseline`) | Our PyTorch (`fed-channel-alloc`) |
|--------|-------------------------------------|-------------------------------------|
| Last-20 mean reward | **71.05** | **71.92** |
| Last-20 mean channel changes | 2.85 | **1.90** |
| % of paper ceiling (88) | ~81% | ~80% |
| Final episode reward | 72.58 (@ep 950) | 72.80 (@ep 1000) |
| Runtime | ~2.3 h | ~2.5 h |

## Conclusion

Results are **very close** — PyTorch refactor matches original TensorFlow behavior for centralized CARLTON within noise. Slightly lower channel mobility (cc) on PyTorch is a minor positive.

## Artifacts

| Run | Weights | Log |
|-----|---------|-----|
| Paper TF | `carlton-paper-baseline/Train_weights_/` | `baseline_training_log.txt` |
| PyTorch | `fed-channel-alloc/Train_weights_/` | `training_log_centralized.txt` |

Next: evaluate both on `scenarios_for_test/` with `Inference_step_on_test_cases` for paper metrics (ANCC, CT, CQ, SES).
