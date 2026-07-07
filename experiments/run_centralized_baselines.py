"""Run PyTorch + TF centralized baselines (1000 ep), then refresh campaign report."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT.parent / "carlton-paper-baseline"
CAMPAIGN_DIR = REPO_ROOT / "experiments" / "results" / "campaign_20260630"


def _run_pytorch():
    log_path = REPO_ROOT / "training_log_centralized_v3.txt"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            [sys.executable, "-u", str(REPO_ROOT / "main_centralized.py"), "--episodes", "1000", "--fresh"],
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {"name": "pytorch", "exit_code": proc.returncode, "log": str(log_path)}


def _run_tf():
    log_path = BASELINE_ROOT / "baseline_training_log_v3.txt"
    env = os.environ.copy()
    env["TF_USE_LEGACY_KERAS"] = "1"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            ["py", "-3.11", "-u", "main_v3.py"],
            cwd=str(BASELINE_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return {"name": "tf", "exit_code": proc.returncode, "log": str(log_path)}


def main():
    print("Starting centralized baselines (1000 episodes each) in parallel...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run_pytorch), pool.submit(_run_tf)]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            print(f"  {result['name']} finished exit={result['exit_code']} log={result['log']}", flush=True)

    failed = [r for r in results if r["exit_code"] != 0]
    if failed:
        print("Baseline run(s) failed:", failed, flush=True)
        sys.exit(1)

    print("Rebuilding campaign report...", flush=True)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from experiments.build_html_report import build_report

    report = build_report(CAMPAIGN_DIR)
    print(f"Report updated: {report}", flush=True)


if __name__ == "__main__":
    main()
