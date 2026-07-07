"""Load centralized logs and federated histories for benchmark plots."""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT.parent / "carlton-paper-baseline"

REQUIRED_EPISODES = 1000
MIN_ACCEPTED_EPISODES = 900
PYTORCH_CENTRALIZED_LOG = REPO_ROOT / "training_log_centralized_v3.txt"
TF_CENTRALIZED_LOG = BASELINE_ROOT / "baseline_training_log_v3.txt"
PYTORCH_TRAIN_INFO = REPO_ROOT / "train_info.pk"
TF_TRAIN_INFO = BASELINE_ROOT / "train_info.pk"

REWARD_PATTERNS = (
    re.compile(r"(?:^|\s)(?:ep|Round)=?\s*(\d+).*?reward=([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(
        r"Episode:\s*(\d+).*?average_accumulated_reward_val:\s*([-+]?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
)
EPISODE_RE = re.compile(r"Episode:\s*(\d+)", re.IGNORECASE)
AVERAGE_REWARD_RE = re.compile(r"average_accumulated_reward_val:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_reward_log(path):
    """Parse reward traces from CARLTON text logs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    episodes = []
    rewards = []
    for line in _read_text(path).splitlines():
        match = None
        for pattern in REWARD_PATTERNS:
            match = pattern.search(line)
            if match:
                break
        if match:
            episodes.append(int(match.group(1)))
            rewards.append(float(match.group(2)))
            continue

        episode_match = EPISODE_RE.search(line)
        reward_match = AVERAGE_REWARD_RE.search(line)
        if episode_match and reward_match:
            episodes.append(int(episode_match.group(1)))
            rewards.append(float(reward_match.group(1)))

    if not rewards:
        raise ValueError(f"No rewards found in {path}")

    return {"label": path.stem, "episodes": episodes, "rewards": rewards, "path": str(path)}


def _read_text(path):
    for encoding in ("utf-8", "utf-16", "utf-16-le"):
        try:
            text = Path(path).read_text(encoding=encoding)
            if "reward" in text.lower() or "episode" in text.lower():
                return text
        except UnicodeError:
            continue
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def load_federated_history(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    rewards = [float(v) for v in history.get("round_rewards", [])]
    channel_changes = [float(v) for v in history.get("round_channel_changes", [])]
    if not rewards:
        raise ValueError(f"No round_rewards found in {path}")
    return {
        "label": history.get("run_name", path.parent.name),
        "episodes": list(range(1, len(rewards) + 1)),
        "rewards": rewards,
        "channel_changes": channel_changes,
        "path": str(path),
        "config": history.get("config", {}),
    }


def validate_reward_trace(trace, min_points=20, reward_min=0.0, reward_max=120.0):
    rewards = np.asarray(trace["rewards"], dtype=float)
    if len(rewards) < min_points:
        raise ValueError(f"{trace['label']} has only {len(rewards)} reward points.")
    if np.isnan(rewards).any():
        raise ValueError(f"{trace['label']} contains NaN rewards.")
    if np.min(rewards) < reward_min or np.max(rewards) > reward_max:
        raise ValueError(
            f"{trace['label']} rewards outside expected range "
            f"[{reward_min}, {reward_max}]: min={np.min(rewards)}, max={np.max(rewards)}"
        )
    return True


def load_train_info_rewards(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    repo_root = path.parent
    for candidate in (repo_root, BASELINE_ROOT, REPO_ROOT):
        candidate = str(candidate)
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    with path.open("rb") as f:
        train_info = pickle.load(f)
    rewards = [float(v) for v in train_info.average_accumulated_reward_vec]
    channel_changes = [float(v) for v in getattr(train_info, "average_changed_channels_vec", [])]
    if len(rewards) < MIN_ACCEPTED_EPISODES:
        raise ValueError(
            f"{label}: need at least {MIN_ACCEPTED_EPISODES} episodes, got {len(rewards)} in {path}"
        )
    return {
        "label": label,
        "episodes": list(range(1, len(rewards) + 1)),
        "rewards": rewards,
        "channel_changes": channel_changes,
        "path": str(path),
    }


def is_complete_federated_history(path):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    rewards = history.get("round_rewards", [])
    cfg_rounds = history.get("config", {}).get("communication_rounds")
    expected = int(cfg_rounds) if cfg_rounds else REQUIRED_EPISODES
    return len(rewards) >= min(MIN_ACCEPTED_EPISODES, expected)


def load_centralized_traces(min_points=None, require_full=True):
    min_points = min_points or REQUIRED_EPISODES
    traces = {}
    errors = []

    for label, pk_path, log_path in (
        ("PyTorch without FL", PYTORCH_TRAIN_INFO, PYTORCH_CENTRALIZED_LOG),
        ("TF without FL", TF_TRAIN_INFO, TF_CENTRALIZED_LOG),
    ):
        try:
            traces[label] = load_train_info_rewards(pk_path, label)
        except (FileNotFoundError, ValueError, AttributeError) as exc:
            errors.append(str(exc))
            try:
                trace = parse_reward_log(log_path)
                trace["label"] = label
                if require_full and max(trace["episodes"]) < MIN_ACCEPTED_EPISODES:
                    raise ValueError(
                        f"{label}: log {log_path} only reaches episode {max(trace['episodes'])} "
                        f"(need at least {MIN_ACCEPTED_EPISODES})"
                    )
                traces[label] = trace
            except (FileNotFoundError, ValueError) as log_exc:
                errors.append(str(log_exc))

    if not traces:
        raise FileNotFoundError("Missing centralized baselines: " + "; ".join(errors))

    for label, trace in traces.items():
        validate_reward_trace(trace, min_points=min(min_points, len(trace["rewards"])), reward_min=20.0, reward_max=95.0)
    return traces


def run_parser_preflight():
    traces = load_centralized_traces(min_points=10)
    return {
        label: {
            "points": len(trace["rewards"]),
            "first_reward": trace["rewards"][0],
            "last_reward": trace["rewards"][-1],
            "path": trace["path"],
        }
        for label, trace in traces.items()
    }


if __name__ == "__main__":
    print(json.dumps(run_parser_preflight(), indent=2))
