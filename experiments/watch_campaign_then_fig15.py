"""Wait for the private-buffer main comparison campaign to finish, then run
the paper Fig. 15 / Fig. 13 evaluation (WS, CQ, min_CQ, RA, JAR).

This script is intentionally long-lived: it polls for job_complete.json files
and only starts eval_paper_fig15.py once every method has N_SEEDS completions.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "experiments" / "results" / "main_comparison_private_buffers_20260801"
METHODS = ("pytorch", "tf", "fedavg", "fedprox")
N_SEEDS = 30
POLL_SECONDS = 300  # 5 minutes
PY = sys.executable
LOG_PATH = STUDY_DIR.parent / "main_comparison_private_buffers_20260801_watcher.log"


def _log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _counts() -> dict[str, int]:
    out = {}
    for method in METHODS:
        root = STUDY_DIR / method
        if not root.is_dir():
            out[method] = 0
            continue
        out[method] = sum(
            1 for p in root.glob("seed_*/job_complete.json") if p.is_file()
        )
    return out


def _campaign_done(counts: dict[str, int]) -> bool:
    return all(counts.get(m, 0) >= N_SEEDS for m in METHODS)


def _run_fig15() -> int:
    out_dir = STUDY_DIR / "paper_fig15"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(REPO_ROOT / "experiments" / "eval_paper_fig15.py"),
        "--study-dir",
        str(STUDY_DIR),
        "--games-per-n",
        "30",
        "--max-workers",
        "4",
        "--base-seed",
        "424242",
    ]
    _log(f"Starting Fig.15 eval: {' '.join(cmd)}")
    log_file = out_dir / "eval_stdout.log"
    with log_file.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    _log(f"Fig.15 eval finished with exit_code={proc.returncode} (log={log_file})")
    summary = out_dir / "summary.json"
    if summary.is_file():
        data = json.loads(summary.read_text(encoding="utf-8"))
        _log("=== Fig. 15 summary: E[(CQ+minCQ)/2] ===")
        for method, s in (data.get("methods") or {}).items():
            _log(
                f"  {s.get('label', method):20s} "
                f"in={s.get('composite_in'):.3f} "
                f"out={s.get('composite_out'):.3f} "
                f"all={s.get('composite_all'):.3f} "
                f"WS={s.get('ws_all', 'n/a')}"
            )
    return int(proc.returncode)


def main() -> int:
    _log(f"Watcher started for {STUDY_DIR}")
    if not STUDY_DIR.is_dir():
        _log(f"ERROR: study dir missing: {STUDY_DIR}")
        return 2

    while True:
        counts = _counts()
        total = sum(counts.values())
        _log(
            "progress: "
            + ", ".join(f"{m}={counts[m]}/{N_SEEDS}" for m in METHODS)
            + f" | total={total}/{N_SEEDS * len(METHODS)}"
        )
        if _campaign_done(counts):
            _log("Campaign complete — launching paper Fig.15 evaluation.")
            return _run_fig15()
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
