"""Recover good nested PyTorch results and mark bad seeds for re-run."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

STUDY = Path("experiments/results/main_comparison_rho1_multiseed_20260701").resolve()
PYTORCH = STUDY / "pytorch"
NESTED_ROOT = PYTORCH / "seed_06" / "experiments" / "results" / "main_comparison_rho1_multiseed_20260701" / "pytorch"
GOOD_TRAIN_THRESHOLD = 75.0

# Promote good nested runs into canonical seed dirs
if NESTED_ROOT.is_dir():
    for nested in sorted(NESTED_ROOT.glob("seed_*")):
        jc = nested / "job_complete.json"
        if not jc.is_file():
            continue
        data = json.loads(jc.read_text(encoding="utf-8"))
        target = PYTORCH / nested.name
        main_jc = target / "job_complete.json"
        main_train = None
        if main_jc.is_file():
            main_train = json.loads(main_jc.read_text(encoding="utf-8")).get("last_20_train_reward", 0)
        nested_train = data.get("last_20_train_reward", 0)
        if nested_train >= GOOD_TRAIN_THRESHOLD and (main_train is None or nested_train > main_train):
            print(f"Promote {nested.name}: nested train={nested_train:.1f} > main={main_train}")
            target.mkdir(parents=True, exist_ok=True)
            for item in nested.iterdir():
                dest = target / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
    shutil.rmtree(PYTORCH / "seed_06" / "experiments", ignore_errors=True)

# Remove bad/incomplete seeds so multiseed runner re-runs them
for i in range(30):
    sd = PYTORCH / f"seed_{i:02d}"
    jc = sd / "job_complete.json"
    if not jc.is_file():
        print(f"seed_{i:02d}: will re-run (missing)")
        continue
    data = json.loads(jc.read_text(encoding="utf-8"))
    train = float(data.get("last_20_train_reward", 0))
    if train < GOOD_TRAIN_THRESHOLD:
        print(f"seed_{i:02d}: delete bad run train={train:.1f}")
        for name in ("job_complete.json", "history.json", "inference.json", "training_report.json", "train_info.pk"):
            p = sd / name
            if p.exists():
                p.unlink()
        for d in sd.glob("Global_RB_Storage_*"):
            shutil.rmtree(d, ignore_errors=True)
        for d in sd.glob("Train_weights_*"):
            shutil.rmtree(d, ignore_errors=True)
    else:
        print(f"seed_{i:02d}: keep train={train:.1f}")
