"""Overnight helper: keep the private-buffer campaign moving until 120/120.

- If the orchestrator is dead and jobs remain, relaunch with --max-workers 10.
- If all methods are complete and Fig.15 summary is missing, launch eval_paper_fig15.
- Safe to run repeatedly (idempotent checks).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "experiments" / "results" / "main_comparison_private_buffers_20260801"
METHODS = ("pytorch", "tf", "fedavg", "fedprox")
N_SEEDS = 30
PY = str(Path(r"C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe"))
LOG = STUDY.parent / "overnight_helper.log"
RESUME_LOG = STUDY.parent / "main_comparison_private_buffers_20260801_resume10_stdout.log"
FIG15_DIR = STUDY / "paper_fig15"
POLL = 900  # 15 minutes


def _log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _counts() -> dict[str, int]:
    out = {}
    for m in METHODS:
        root = STUDY / m
        out[m] = (
            sum(1 for p in root.glob("seed_*/job_complete.json") if p.is_file())
            if root.is_dir()
            else 0
        )
    return out


def _orch_alive() -> bool:
    try:
        import psutil  # type: ignore
    except ImportError:
        # Fallback: tasklist scan
        out = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
            capture_output=True,
            text=True,
            cwd=str(REPO),
        )
        return "run_main_comparison_multiseed.py" in (out.stdout or "")
    for p in psutil.process_iter(["cmdline"]):
        cmd = " ".join(p.info.get("cmdline") or [])
        if "run_main_comparison_multiseed.py" in cmd and "private_buffers_20260801" in cmd:
            return True
    return False


def _campaign_done(counts: dict[str, int]) -> bool:
    return all(counts.get(m, 0) >= N_SEEDS for m in METHODS)


def _fig15_done() -> bool:
    return (FIG15_DIR / "summary.json").is_file()


def _launch_resume(workers: int = 10) -> None:
    cmd = [
        PY,
        str(REPO / "experiments" / "run_main_comparison_multiseed.py"),
        "--study-dir",
        str(STUDY),
        "--n-seeds",
        str(N_SEEDS),
        "--max-workers",
        str(workers),
    ]
    _log(f"Launching resume: {' '.join(cmd)}")
    RESUME_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RESUME_LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"\n--- resume launch {datetime.now().isoformat()} workers={workers} ---\n")
        subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=fh,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )


def _launch_fig15() -> None:
    FIG15_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(REPO / "experiments" / "eval_paper_fig15.py"),
        "--study-dir",
        str(STUDY),
        "--games-per-n",
        "30",
        "--max-workers",
        "4",
        "--base-seed",
        "424242",
    ]
    _log(f"Launching Fig.15: {' '.join(cmd)}")
    log_file = FIG15_DIR / "eval_stdout.log"
    with log_file.open("w", encoding="utf-8") as fh:
        subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=fh,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )


def main() -> int:
    _log(f"Overnight helper started for {STUDY}")
    while True:
        counts = _counts()
        total = sum(counts.values())
        alive = _orch_alive()
        _log(
            "progress: "
            + ", ".join(f"{m}={counts[m]}/{N_SEEDS}" for m in METHODS)
            + f" | total={total}/{N_SEEDS * len(METHODS)} | orch_alive={alive}"
        )
        if _campaign_done(counts):
            _log("Campaign complete.")
            if _fig15_done():
                _log("Fig.15 already present — helper exiting.")
                return 0
            # Prefer existing watcher; if summary still missing after a bit, launch ourselves.
            _log("Waiting up to 30 min for watcher Fig.15...")
            for _ in range(6):
                time.sleep(300)
                if _fig15_done():
                    _log("Fig.15 appeared — helper exiting.")
                    return 0
            if not _fig15_done():
                _launch_fig15()
                _log("Fig.15 launched by helper — exiting (eval runs independently).")
                return 0
        if not alive:
            remaining = N_SEEDS * len(METHODS) - total
            if remaining > 0:
                _log(f"Orchestrator dead with {remaining} jobs left — relaunching with 10 workers.")
                _launch_resume(10)
                time.sleep(60)
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
