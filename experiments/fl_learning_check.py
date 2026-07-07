"""Learning-signal checks for short federated CARLTON runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _mean_window(values, start, end):
    chunk = values[start:end]
    return float(np.mean(chunk)) if chunk else float("nan")


def assess_learning(rewards, channel_changes, min_rounds=20):
    """Return a pass/fail signal for a short FL smoke run.

    Reward traces are noisy in CARLTON, so either higher late reward or lower
    channel switching is treated as enough evidence to continue.
    """
    rewards = [float(v) for v in rewards]
    channel_changes = [float(v) for v in channel_changes]
    n = min(len(rewards), len(channel_changes))
    if n < min_rounds:
        return {
            "status": "insufficient_data",
            "rounds": n,
            "message": f"Need at least {min_rounds} rounds.",
        }

    window = min(10, n // 2)
    r_early = _mean_window(rewards, 0, window)
    r_late = _mean_window(rewards, n - window, n)
    cc_early = _mean_window(channel_changes, 0, window)
    cc_late = _mean_window(channel_changes, n - window, n)

    reward_improved = r_late > r_early * 1.03
    mobility_improved = cc_late < cc_early * 0.90
    passed = reward_improved or mobility_improved

    return {
        "status": "pass" if passed else "fail",
        "rounds": n,
        "reward_early": r_early,
        "reward_late": r_late,
        "reward_improved": reward_improved,
        "channel_changes_early": cc_early,
        "channel_changes_late": cc_late,
        "mobility_improved": mobility_improved,
        "pct_of_paper_ceiling_88": 100.0 * r_late / 88.0,
    }


def assess_history(history, min_rounds=20):
    return assess_learning(
        history.get("round_rewards", []),
        history.get("round_channel_changes", []),
        min_rounds=min_rounds,
    )


def write_assessment(history, output_path, min_rounds=20):
    assessment = assess_history(history, min_rounds=min_rounds)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assessment, indent=2), encoding="utf-8")
    return assessment
