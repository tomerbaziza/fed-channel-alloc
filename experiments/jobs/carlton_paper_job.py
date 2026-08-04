"""Faithful CARLTON CTDE reimplementation from arXiv:2402.17773 (Table II).

Differences from `main_centralized.single_training_run`, both taken straight
from the published specification:

* Glorot-Uniform weight initialization with zero biases (Section III-D), instead
  of PyTorch's default Kaiming-uniform.
* Training scenarios draw N from {2,...,7}; the legacy loop used
  ``np.random.randint(2, 7)`` and therefore never saw a 7-network scenario.

Everything else follows Table II: K=10 channels, gamma=0.9, Huber delta=1,
Adam(0.9, 0.999, 1e-7), 40 gradient steps of batch 32 per episode on a global
replay memory of 1e5 transitions, epsilon annealed 0.5 -> 0.01 over B/2
episodes, and (omega, lr) switching from (0.02, 2.5e-4) to (0.2, 1e-4) at B/2.

Usage:
  py -3 experiments/jobs/carlton_paper_job.py --save-dir <dir> --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import DeepMellow_Single_agent.DeepMellow_no_epsilon as deepmellow

MAX_TRAIN_NETS = 7
NUMBER_OF_CHANNELS = 10
RHO = 0.7


def apply_glorot_init():
    """Table II: Glorot-Uniform weights, zero biases."""
    base_init = deepmellow.QResNet.__init__

    def glorot_init(self, *args, **kwargs):
        base_init(self, *args, **kwargs)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    deepmellow.QResNet.__init__ = glorot_init


def train(save_dir: Path, seed: int, episodes: int, inference_episodes: int):
    from BuildingBlocks.TrainBlock import train_model
    from BuildingBlocks.Worker import worker
    from SimulationEnvironments.Pythonic_Environment import python_env
    from Utils.RandomLocationOfNetworks import set_random_location_of_networks
    from Utils.ScenarioExamination import get_game_performamce
    from Utils.save_to_df_csv import wrrape_game_history_do_df

    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))

    tag = f"carlton_{seed:02d}"
    base_dir = str(save_dir)
    lr = 0.00025
    epsilon = 0.5
    mellowmax_constant = 0.02
    half = episodes // 2

    rewards, channel_changes, ws_vec, cq_mean_vec, cq_min_vec = [], [], [], [], []

    for episode in range(episodes):
        # Paper: training is limited to scenarios of at most 7 networks.
        number_of_nets = int(np.random.randint(2, MAX_TRAIN_NETS + 1))
        users, centers = set_random_location_of_networks(number_of_nets)
        scenario = python_env(
            number_of_nets=number_of_nets,
            number_of_users_in_each_net=users,
            net_center_location_and_std=centers,
            possible_channels=NUMBER_OF_CHANNELS,
            add_noise=False,
            training=True,
        )

        avg_reward, avg_cc, _, game_history = worker(
            address_scen="",
            scenario=scenario,
            address_algo=base_dir,
            history_length=1,
            training=True,
            epsilon=epsilon,
            save_to_global_rb=True,
            local_train_steps=0,
            batch_size=32,
            replay_memory_size=100000,
            i_d_folder=tag,
            rho=RHO,
            verbose=False,
        )

        epsilon = max(0.01, epsilon - (0.5 - 0.01) / max(1, half))
        if episode > half:
            mellowmax_constant, lr = 0.2, 0.0001
        else:
            mellowmax_constant, lr = 0.02, 0.00025

        train_model(
            trainig_iterations=40,
            action_space=NUMBER_OF_CHANNELS,
            batch_size=32,
            learning_rate=lr,
            history=1,
            mellowmax_constant=mellowmax_constant,
            i_d_folder=tag,
            base_dir=base_dir,
            verbose=False,
        )

        perf = get_game_performamce(
            game_history=wrrape_game_history_do_df(game_history, NUMBER_OF_CHANNELS),
            number_of_channels=NUMBER_OF_CHANNELS,
            save_file=False,
        )
        _ancc, _ct, cq_mean, _cq_median, _cq_max, cq_min = perf[0], perf[1], perf[2], perf[3], perf[4], perf[5]
        ws = perf[11]

        rewards.append(float(avg_reward))
        channel_changes.append(float(avg_cc))
        cq_mean_vec.append(float(cq_mean))
        cq_min_vec.append(float(cq_min))
        ws_vec.append(float(ws))

        if episode % 50 == 0 or episode == episodes - 1:
            print(
                f"[seed {seed}] episode {episode}/{episodes} "
                f"reward={avg_reward:.2f} ws={ws:.3f}",
                flush=True,
            )

    history = {
        "method": "CARLTON CTDE (paper-faithful)",
        "config": {
            "episodes": episodes,
            "seed": seed,
            "rho": RHO,
            "max_train_nets": MAX_TRAIN_NETS,
            "init": "glorot_uniform",
        },
        "round_rewards": rewards,
        "round_channel_changes": channel_changes,
        "cq_mean": cq_mean_vec,
        "cq_min": cq_min_vec,
        "ws": ws_vec,
    }

    if inference_episodes:
        from main_centralized import run_centralized_inference

        history["inference"] = run_centralized_inference(
            n_episodes=inference_episodes,
            base_dir=base_dir,
            i_d_folder=tag,
            rho=RHO,
            seed=seed + 10000,
        )
        (save_dir / "inference.json").write_text(
            json.dumps(history["inference"], indent=2), encoding="utf-8"
        )

    (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (save_dir / "job_complete.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "last_20_train_reward": float(np.mean(rewards[-20:])),
                "inference_reward_mean": (history.get("inference") or {}).get(
                    "inference_reward_mean"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return history


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--inference-episodes", type=int, default=100)
    args = parser.parse_args(argv)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    apply_glorot_init()
    torch.set_num_threads(1)
    train(save_dir, args.seed, args.episodes, args.inference_episodes)


if __name__ == "__main__":
    main()
