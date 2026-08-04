"""Centralized CARLTON training (paper CTDE) — pre-FL original loop.

Episode loop:
  1. Run one scenario via worker (global replay on, no local FL steps)
  2. train_model() on merged global replay buffer (40 gradient steps)
  3. Next episode loads latest weights from Train_weights_*
"""

import argparse
import json
import os
import pickle
import shutil
from datetime import datetime

import numpy as np

from BuildingBlocks.TrainBlock import train_model
from BuildingBlocks.Worker import worker
from SimulationEnvironments.Pythonic_Environment import python_env
from Utils.RandomLocationOfNetworks import set_random_location_of_networks
from Utils.ScenarioExamination import get_game_performamce
from Utils.dotdict import dotdict
from Utils.get_adress_scen_and_adress_algo import get_adress_scen_and_adress_algo
from Utils.save_to_df_csv import wrrape_game_history_do_df

try:
    from Utils.createImages import create_images
except ImportError:
    create_images = None


def _mean_window(values, start, end):
    chunk = values[start:end]
    return float(np.mean(chunk)) if chunk else float("nan")


def assess_learning(rewards, channel_changes):
    n = len(rewards)
    if n < 10:
        return {"status": "insufficient_data", "episodes": n}

    r_early = _mean_window(rewards, 0, min(10, n))
    r_late = _mean_window(rewards, max(0, n - 10), n)
    cc_early = _mean_window(channel_changes, 0, min(10, n))
    cc_late = _mean_window(channel_changes, max(0, n - 10), n)

    reward_improved = r_late > r_early * 1.03
    mobility_improved = cc_late < cc_early * 0.90
    passed = reward_improved or mobility_improved

    return {
        "status": "pass" if passed else "fail",
        "episodes": n,
        "reward_early": r_early,
        "reward_late": r_late,
        "reward_improved": reward_improved,
        "channel_changes_early": cc_early,
        "channel_changes_late": cc_late,
        "mobility_improved": mobility_improved,
        "pct_of_paper_ceiling_88": 100.0 * r_late / 88.0,
    }


def run_centralized_inference(
    n_episodes=100,
    history_length=1,
    number_of_channels=10,
    i_d_folder="",
    base_dir=None,
    rho=None,
    seed=None,
):
    """Run inference episodes with latest centralized weights (no train_model)."""
    if seed is not None:
        np.random.seed(int(seed))
        import random

        random.seed(int(seed))

    address_scen = ""
    if not base_dir:
        _, address_algo = get_adress_scen_and_adress_algo(script_path=os.getcwd())
        base_dir = address_algo or os.getcwd()
    number_of_possible_nets = 7

    rewards = []
    channel_changes = []
    for j in range(int(n_episodes)):
        # Paper: train with N in {2,...,7} inclusive (in-sample domain).
        number_of_nets = np.random.randint(2, number_of_possible_nets + 1)
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
            address_algo=base_dir,
            history_length=history_length,
            training=False,
            epsilon=0.0,
            save_to_global_rb=False,
            local_train_steps=0,
            batch_size=32,
            replay_memory_size=100000,
            i_d_folder=i_d_folder,
            rho=rho,
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


def single_training_run(
    n_episodes=1000,
    history_length=1,
    number_of_channels=10,
    i_d_folder="",
    fresh_start=False,
    early_check_at=20,
    log_every=10,
    checkpoint_every=50,
    make_plots=False,
    rho=None,
    inference_episodes=0,
    seed=None,
    output_dir=None,
):
    original_cwd = os.getcwd()
    address_scen = ""
    if output_dir:
        base_dir = os.path.abspath(output_dir)
        os.makedirs(base_dir, exist_ok=True)
    else:
        _, address_algo = get_adress_scen_and_adress_algo(script_path=original_cwd)
        base_dir = os.path.abspath(address_algo or original_cwd)

    os.chdir(base_dir)
    try:
        return _single_training_run_body(
            n_episodes=n_episodes,
            history_length=history_length,
            number_of_channels=number_of_channels,
            i_d_folder=i_d_folder,
            fresh_start=fresh_start,
            early_check_at=early_check_at,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            make_plots=make_plots,
            rho=rho,
            inference_episodes=inference_episodes,
            seed=seed,
            base_dir=base_dir,
            address_scen=address_scen,
        )
    finally:
        os.chdir(original_cwd)


def _single_training_run_body(
    n_episodes,
    history_length,
    number_of_channels,
    i_d_folder,
    fresh_start,
    early_check_at,
    log_every,
    checkpoint_every,
    make_plots,
    rho,
    inference_episodes,
    seed,
    base_dir,
    address_scen,
):
    if seed is not None:
        np.random.seed(int(seed))
        import random

        random.seed(int(seed))

    rb_dir = os.path.join(base_dir, f"Global_RB_Storage_{i_d_folder}")
    wt_dir = os.path.join(base_dir, f"Train_weights_{i_d_folder}")
    if fresh_start:
        for d in (rb_dir, wt_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)

    number_of_possible_nets = 7
    lr = 0.00025
    epsilon = 0.5
    mellowmax_constant = 0.02

    train_info = dotdict(
        {
            "average_accumulated_reward_vec": [],
            "average_changed_channels_vec": [],
            "cq_mean_vec_train": [],
            "cq_median": [],
            "cq_min": [],
            "ancc_vec_train": [],
            "ct_vec_train": [],
            "se_vec_train": [],
            "ancc_score_vec_train": [],
            "ct_score_vec_train": [],
            "ws_vec_train": [],
            "numebr_of_nets": [],
        }
    )

    print(
        f"=== Centralized CARLTON (no FL) | {n_episodes} episodes | "
        f"channels={number_of_channels} ===",
        flush=True,
    )

    for j in range(n_episodes):
        if j % log_every == 0:
            print(f"Episode {j}/{n_episodes}", flush=True)

        # Paper: train with N in {2,...,7} inclusive (in-sample domain).
        number_of_nets = np.random.randint(2, number_of_possible_nets + 1)
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
            address_algo=base_dir,
            history_length=history_length,
            training=True,
            epsilon=epsilon,
            save_to_global_rb=True,
            local_train_steps=0,
            batch_size=32,
            replay_memory_size=100000,
            i_d_folder=i_d_folder,
            rho=rho,
            verbose=False,
        )

        epsilon = max(0.01, epsilon - (0.5 - 0.01) / max(1, n_episodes // 2))

        # Table II: omega and lr switch at episode i > B/2
        half = n_episodes // 2
        if j > half:
            mellowmax_constant = 0.2
            lr = 0.0001
        else:
            mellowmax_constant = 0.02
            lr = 0.00025

        costs = train_model(
            trainig_iterations=40,
            action_space=number_of_channels,
            batch_size=32,
            learning_rate=lr,
            history=history_length,
            mellowmax_constant=mellowmax_constant,
            i_d_folder=i_d_folder,
            base_dir=base_dir,
            verbose=False,
        )
        mean_cost = float(np.mean(costs)) if costs else None
        loss_start = float(costs[0]) if costs else None
        loss_end = float(costs[-1]) if costs else None

        gh_df = wrrape_game_history_do_df(game_history, number_of_channels)
        perf = get_game_performamce(
            game_history=gh_df,
            number_of_channels=number_of_channels,
            save_file=False,
        )
        (
            ancc,
            ct,
            cq_mean,
            cq_median,
            _cq_max,
            cq_min,
            _above90,
            _below90,
            se,
            ancc_score,
            ct_score,
            ws,
            _used_ch,
            _reuse,
        ) = perf

        train_info.average_accumulated_reward_vec.append(float(avg_reward))
        train_info.average_changed_channels_vec.append(float(avg_cc))
        train_info.cq_mean_vec_train.append(float(cq_mean))
        train_info.cq_median.append(float(cq_median))
        train_info.cq_min.append(float(cq_min))
        train_info.ancc_vec_train.append(float(ancc))
        train_info.ct_vec_train.append(float(ct))
        train_info.se_vec_train.append(float(se))
        train_info.ancc_score_vec_train.append(float(ancc_score))
        train_info.ct_score_vec_train.append(float(ct_score))
        train_info.ws_vec_train.append(float(ws))
        train_info.numebr_of_nets.append(int(number_of_nets))

        if (j + 1) % log_every == 0 or j == 0:
            loss_msg = (
                f"loss={mean_cost:.3f} (start={loss_start:.3f} end={loss_end:.3f})"
                if mean_cost is not None and loss_start is not None and loss_end is not None
                else "loss=n/a (buffer warming up)"
            )
            print(
                f"  ep={j + 1} | nets={number_of_nets} | reward={avg_reward:.2f} | "
                f"cc={avg_cc:.2f} | cq_mean={cq_mean:.2f} | "
                f"{loss_msg} | "
                f"eps={epsilon:.3f}",
                flush=True,
            )

        if early_check_at and j + 1 == early_check_at:
            mid = assess_learning(
                train_info.average_accumulated_reward_vec,
                train_info.average_changed_channels_vec,
            )
            print(f"\n--- Early learning check @ ep {early_check_at}: {mid['status'].upper()} ---", flush=True)
            print(json.dumps(mid, indent=2), flush=True)
            if mid["status"] == "fail":
                print("STOP: no learning signal — debug before continuing.", flush=True)
                return train_info, mid, True, None

        if checkpoint_every and (j + 1) % checkpoint_every == 0:
            with open("train_info.pk", "wb") as f:
                pickle.dump(train_info, f)
            if make_plots and create_images is not None:
                create_images(train_info)

    final = assess_learning(
        train_info.average_accumulated_reward_vec,
        train_info.average_changed_channels_vec,
    )

    report = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "centralized_ctde",
        "episodes": n_episodes,
        "final_assessment": final,
        "last_20_mean_reward": float(np.mean(train_info.average_accumulated_reward_vec[-20:])),
        "last_20_mean_cc": float(np.mean(train_info.average_changed_channels_vec[-20:])),
        "rho": rho,
        "i_d_folder": i_d_folder,
        "output_dir": base_dir,
    }

    inference = None
    if inference_episodes and inference_episodes > 0:
        print(
            f"Running {inference_episodes} inference episodes (no weight updates)...",
            flush=True,
        )
        inference = run_centralized_inference(
            n_episodes=inference_episodes,
            history_length=history_length,
            number_of_channels=number_of_channels,
            i_d_folder=i_d_folder,
            base_dir=base_dir,
            rho=rho,
            seed=(int(seed) + 900_000) if seed is not None else None,
        )
        report["inference"] = inference
        with open(os.path.join(base_dir, "inference.json"), "w", encoding="utf-8") as f:
            json.dump(inference, f, indent=2)

    with open(os.path.join(base_dir, "train_info.pk"), "wb") as f:
        pickle.dump(train_info, f)
    with open(os.path.join(base_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Done | final: {final['status'].upper()} ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return train_info, final, False, inference


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized CARLTON (no FL)")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--fresh", action="store_true", help="Delete replay/weights before start")
    parser.add_argument("--quick", action="store_true", help="25-episode smoke test with early stop")
    args = parser.parse_args()

    if args.quick:
        single_training_run(n_episodes=25, fresh_start=True, early_check_at=20, log_every=5)
    else:
        single_training_run(n_episodes=args.episodes, fresh_start=args.fresh, early_check_at=0)
