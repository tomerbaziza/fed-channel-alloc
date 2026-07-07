"""One TF centralized training run + inference (baseline repo, rho=0.7 unchanged)."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = REPO_ROOT.parent / "carlton-paper-baseline"


def _setup_baseline_import_paths():
    """Baseline modules use relative sys.path.append from repo root."""
    root = str(BASELINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    for sub in ("SimulationEnvironments", "DeepMellow_Single_agent", "BuildingBlocks", "Utils"):
        path = str(BASELINE_ROOT / sub)
        if path not in sys.path:
            sys.path.insert(0, path)


def _run_training(n_episodes, seed, i_d_folder=""):
    _setup_baseline_import_paths()
    from BuildingBlocks.TrainBlock import train_model
    from BuildingBlocks.Worker import worker
    from SimulationEnvironments.Pythonic_Environment import python_env
    from Utils.RandomLocationOfNetworks import set_random_location_of_networks
    from Utils.ScenarioExamination import get_game_performamce
    from Utils.dotdict import dotdict
    from Utils.get_adress_scen_and_adress_algo import get_adress_scen_and_adress_algo
    from Utils.save_to_df_csv import wrrape_game_history_do_df

    np.random.seed(int(seed))
    random.seed(int(seed))

    _, address_algo = get_adress_scen_and_adress_algo(script_path=str(BASELINE_ROOT))
    address_scen = ""
    number_of_possible_nets = 7
    number_of_channels = 10
    lr = 0.00025
    epsilon = 0.5
    mellowmax_constant = 0.02

    train_info = dotdict(
        {
            "average_accumulated_reward_vec": [],
            "average_changed_channels_vec": [],
        }
    )

    for j in range(int(n_episodes)):
        number_of_nets = np.random.randint(2, number_of_possible_nets)
        users, centers = set_random_location_of_networks(number_of_nets)
        scenario = python_env(
            number_of_nets=number_of_nets,
            number_of_users_in_each_net=users,
            net_center_location_and_std=centers,
            possible_channels=number_of_channels,
            add_noise=False,
            training=True,
        )
        avg_reward, avg_cc, _, game_history = worker(
            address_scen=address_scen,
            scenario=scenario,
            address_algo=address_algo,
            history_length=1,
            training=True,
            epsilon=epsilon,
            i_d_folder=i_d_folder,
            verbose=False,
        )
        epsilon = max(0.01, epsilon - (0.5 - 0.01) / max(1, n_episodes // 2))
        if j > n_episodes // 2:
            mellowmax_constant = 0.2
            lr = 0.0001
        else:
            mellowmax_constant = 0.02
            lr = 0.00025

        train_model(
            trainig_iterations=40,
            action_space=number_of_channels,
            batch_size=32,
            learning_rate=lr,
            history=1,
            mellowmax_constant=mellowmax_constant,
            i_d_folder=i_d_folder,
            verbose=False,
        )

        gh_df = wrrape_game_history_do_df(game_history, number_of_channels)
        _ = get_game_performamce(
            game_history=gh_df,
            number_of_channels=number_of_channels,
            save_file=False,
        )

        train_info.average_accumulated_reward_vec.append(float(avg_reward))
        train_info.average_changed_channels_vec.append(float(avg_cc))

    return train_info


def _run_inference(n_episodes, seed, i_d_folder=""):
    _setup_baseline_import_paths()
    from BuildingBlocks.Worker import worker
    from SimulationEnvironments.Pythonic_Environment import python_env
    from Utils.RandomLocationOfNetworks import set_random_location_of_networks
    from Utils.get_adress_scen_and_adress_algo import get_adress_scen_and_adress_algo

    np.random.seed(int(seed))
    random.seed(int(seed))

    _, address_algo = get_adress_scen_and_adress_algo(script_path=str(BASELINE_ROOT))
    address_scen = ""
    number_of_possible_nets = 7
    number_of_channels = 10
    rewards = []
    channel_changes = []

    for _ in range(int(n_episodes)):
        number_of_nets = np.random.randint(2, number_of_possible_nets)
        users, centers = set_random_location_of_networks(number_of_nets)
        scenario = python_env(
            number_of_nets=number_of_nets,
            number_of_users_in_each_net=users,
            net_center_location_and_std=centers,
            possible_channels=number_of_channels,
            add_noise=False,
            training=True,
        )
        avg_reward, avg_cc, _, _ = worker(
            address_scen=address_scen,
            scenario=scenario,
            address_algo=address_algo,
            history_length=1,
            training=False,
            epsilon=0.0,
            i_d_folder=i_d_folder,
            verbose=False,
        )
        rewards.append(float(avg_reward))
        channel_changes.append(float(avg_cc))

    return {
        "inference_rewards": rewards,
        "inference_channel_changes": channel_changes,
        "inference_reward_mean": float(np.mean(rewards)) if rewards else None,
        "inference_reward_last_20": float(np.mean(rewards[-20:])) if rewards else None,
        "inference_cc_mean": float(np.mean(channel_changes)) if channel_changes else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--inference-episodes", type=int, default=100)
    args = parser.parse_args(argv)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    _setup_baseline_import_paths()
    os.chdir(save_dir)
    if str(BASELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(BASELINE_ROOT))
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    train_info = _run_training(args.episodes, args.seed)
    inference = _run_inference(
        args.inference_episodes,
        int(args.seed) + 900_000,
    )

    history = {
        "method": "TF centralized (no FL)",
        "config": {
            "episodes": args.episodes,
            "inference_episodes": args.inference_episodes,
            "seed": args.seed,
            "rho": 0.7,
        },
        "round_rewards": list(train_info.average_accumulated_reward_vec),
        "round_channel_changes": list(train_info.average_changed_channels_vec),
        "inference": inference,
    }

    with open(save_dir / "train_info.pk", "wb") as f:
        pickle.dump(train_info, f)
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(save_dir / "inference.json", "w", encoding="utf-8") as f:
        json.dump(inference, f, indent=2)
    with open(save_dir / "job_complete.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "seed": args.seed,
                "last_20_train_reward": float(np.mean(history["round_rewards"][-20:])),
                "inference_last_20": inference.get("inference_reward_last_20"),
            },
            f,
            indent=2,
        )
    print(json.dumps({"save_dir": str(save_dir), "status": "completed"}, indent=2))


if __name__ == "__main__":
    main()
