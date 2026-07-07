"""Federated CARLTON training loop (PyTorch FRL refactor)."""

import json
import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
from BuildingBlocks.Worker import worker
from BuildingBlocks.TrainBlock import federated_averaging
from Utils.RandomLocationOfNetworks import set_random_location_of_networks
from SimulationEnvironments.Pythonic_Environment import python_env
from DeepMellow_Single_agent.ReplayMemory import ReplayMemory
from Utils.SetSpecificEnv import set_specific_env


def _build_fixed_topology(number_of_nets, map_seed):
    """Sample one fixed map layout (6 nets etc.) and reuse every round."""
    np.random.seed(int(map_seed))
    users_per_net, net_centers = set_random_location_of_networks(int(number_of_nets))
    users_locations_per_net = []
    for j in range(int(number_of_nets)):
        mean_x, mean_y, std_x, std_y = net_centers[j]
        n_users = users_per_net[j]
        location_x = np.random.normal(loc=mean_x, scale=std_x, size=(n_users, 1))
        location_y = np.random.normal(loc=mean_y, scale=std_y, size=(n_users, 1))
        users_locations_per_net.append(np.concatenate((location_x, location_y), axis=1))
    return users_per_net, users_locations_per_net


def _make_scenario(
    number_of_channels,
    fixed_topology,
    fixed_number_of_nets,
    scenario_map_seed,
    scenario_name,
    min_nets,
    max_nets,
    fixed_users_per_net,
    fixed_users_locations,
):
    if scenario_name:
        return set_specific_env(
            scenario_name=scenario_name,
            number_of_channels=number_of_channels,
            training=True,
        )

    if fixed_topology:
        return python_env(
            number_of_nets=len(fixed_users_per_net),
            number_of_users_in_each_net=fixed_users_per_net,
            net_center_location_and_std=None,
            users_locations_per_net=fixed_users_locations,
            possible_channels=number_of_channels,
            add_noise=False,
            training=True,
        )

    number_of_nets = np.random.randint(min_nets, max_nets + 1)
    users_per_net, net_centers = set_random_location_of_networks(number_of_nets)
    return python_env(
        number_of_nets=number_of_nets,
        number_of_users_in_each_net=users_per_net,
        net_center_location_and_std=net_centers,
        possible_channels=number_of_channels,
        add_noise=False,
        training=True,
    )


def _make_persistent_replay(number_of_channels, history_length, batch_size, capacity):
    num_of_bits = int(np.floor(np.log2(number_of_channels)) + 1)
    return ReplayMemory(
        capacity=int(capacity),
        number_of_channels=int(number_of_channels + num_of_bits),
        agent_history_length=int(history_length),
        batch_size=int(batch_size),
    )


def _save_checkpoint(global_weights, history, round_idx, out_dir="Train_weights_frl"):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"round_{round_idx + 1:04d}"
    with open(os.path.join(out_dir, f"global_weights_{tag}.pkl"), "wb") as f:
        pickle.dump(global_weights, f)
    with open(os.path.join(out_dir, f"history_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def run_inference_episodes(
    global_weights,
    n_episodes=100,
    history_length=1,
    number_of_channels=10,
    min_nets=2,
    max_nets=7,
    learning_rate=0.00025,
    batch_size=32,
    fixed_topology=False,
    fixed_number_of_nets=6,
    scenario_map_seed=42,
    scenario_name=None,
    rho=1.0,
    seed=None,
):
    """Evaluate frozen global weights (no replay merge, no local SGD)."""
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))
        torch.manual_seed(int(seed))

    fixed_users_per_net = None
    fixed_users_locations = None
    if fixed_topology and not scenario_name:
        fixed_users_per_net, fixed_users_locations = _build_fixed_topology(
            number_of_nets=fixed_number_of_nets,
            map_seed=scenario_map_seed,
        )

    rewards = []
    channel_changes = []
    for _ in range(int(n_episodes)):
        scenario = _make_scenario(
            number_of_channels=number_of_channels,
            fixed_topology=fixed_topology,
            fixed_number_of_nets=fixed_number_of_nets,
            scenario_map_seed=scenario_map_seed,
            scenario_name=scenario_name,
            min_nets=min_nets,
            max_nets=max_nets,
            fixed_users_per_net=fixed_users_per_net,
            fixed_users_locations=fixed_users_locations,
        )
        avg_reward, avg_cc, _, _ = worker(
            address_scen="",
            scenario=scenario,
            address_algo="",
            training=False,
            history_length=history_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epsilon=0.0,
            global_weights=global_weights,
            local_train_steps=0,
            persistent_replay_buffer=None,
            save_to_global_rb=False,
            rho=rho,
            verbose=False,
        )
        rewards.append(float(avg_reward))
        channel_changes.append(float(avg_cc))

    return {
        "inference_rewards": rewards,
        "inference_channel_changes": channel_changes,
        "inference_reward_mean": float(np.mean(rewards)) if rewards else None,
        "inference_reward_last_20": float(np.mean(rewards[-20:])) if len(rewards) >= 1 else None,
        "inference_cc_mean": float(np.mean(channel_changes)) if channel_changes else None,
    }


def run_federated_training(
    communication_rounds=1000,
    local_train_steps=10,
    history_length=1,
    number_of_channels=10,
    min_nets=2,
    max_nets=7,
    epsilon_start=0.5,
    epsilon_end=0.01,
    learning_rate=0.00025,
    batch_size=32,
    checkpoint_every=50,
    save_dir="Train_weights_frl",
    fedprox_mu=0.0,
    run_name="fedavg",
    seed=None,
    early_check_at=None,
    early_check_callback=None,
    persistent_replay=True,
    replay_memory_size=10000,
    fixed_topology=False,
    fixed_number_of_nets=6,
    scenario_map_seed=42,
    scenario_name=None,
    rho=None,
    inference_episodes=0,
):
    """Run FRL rounds: broadcast -> local train/FedProx -> upload -> FedAvg."""
    os.makedirs(save_dir, exist_ok=True)
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))
        torch.manual_seed(int(seed))

    global_weights = None
    epsilon = epsilon_start
    epsilon_decay_per_round = (epsilon_start - epsilon_end) / max(1, communication_rounds // 2)

    history = {
        "run_name": run_name,
        "config": {
            "communication_rounds": communication_rounds,
            "local_train_steps": local_train_steps,
            "history_length": history_length,
            "number_of_channels": number_of_channels,
            "min_nets": min_nets,
            "max_nets": max_nets,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "fedprox_mu": fedprox_mu,
            "seed": seed,
            "persistent_replay": persistent_replay,
            "replay_memory_size": replay_memory_size,
            "fixed_topology": fixed_topology,
            "fixed_number_of_nets": fixed_number_of_nets,
            "scenario_map_seed": scenario_map_seed,
            "scenario_name": scenario_name,
            "rho": rho,
            "inference_episodes": inference_episodes,
        },
        "round_rewards": [],
        "round_channel_changes": [],
        "round_num_agents": [],
        "epsilon": [],
        "replay_buffer_counts": [],
        "stopped_early": False,
        "early_assessment": None,
    }
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(history["config"], f, indent=2)

    persistent_replay_buffer = None
    if persistent_replay:
        persistent_replay_buffer = _make_persistent_replay(
            number_of_channels=number_of_channels,
            history_length=history_length,
            batch_size=batch_size,
            capacity=replay_memory_size,
        )

    fixed_users_per_net = None
    fixed_users_locations = None
    if fixed_topology and not scenario_name:
        fixed_users_per_net, fixed_users_locations = _build_fixed_topology(
            number_of_nets=fixed_number_of_nets,
            map_seed=scenario_map_seed,
        )
        print(
            f"Fixed topology enabled: {fixed_number_of_nets} networks, map_seed={scenario_map_seed}",
            flush=True,
        )

    for round_idx in range(communication_rounds):
        scenario = _make_scenario(
            number_of_channels=number_of_channels,
            fixed_topology=fixed_topology,
            fixed_number_of_nets=fixed_number_of_nets,
            scenario_map_seed=scenario_map_seed,
            scenario_name=scenario_name,
            min_nets=min_nets,
            max_nets=max_nets,
            fixed_users_per_net=fixed_users_per_net,
            fixed_users_locations=fixed_users_locations,
        )

        avg_reward, avg_channel_changes, agents, _ = worker(
            address_scen="",
            scenario=scenario,
            address_algo="",
            training=True,
            history_length=history_length,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epsilon=epsilon,
            global_weights=global_weights,
            local_train_steps=local_train_steps,
            fedprox_mu=fedprox_mu,
            persistent_replay_buffer=persistent_replay_buffer,
            rho=rho,
            verbose=False,
        )

        buffer_count = int(persistent_replay_buffer.count) if persistent_replay_buffer else 0
        history["replay_buffer_counts"].append(buffer_count)

        local_weights = [agent.state_dict() for agent in agents.values()]
        global_weights = federated_averaging(local_weights)

        history["round_rewards"].append(float(avg_reward))
        history["round_channel_changes"].append(float(avg_channel_changes))
        history["round_num_agents"].append(len(agents))
        history["epsilon"].append(float(epsilon))

        if round_idx < communication_rounds // 2:
            epsilon = max(epsilon_end, epsilon - epsilon_decay_per_round)

        if (round_idx + 1) % 10 == 0 or round_idx == 0:
            buffer_msg = (
                f" | replay_count={buffer_count} (batch={batch_size})"
                if persistent_replay_buffer is not None
                else ""
            )
            print(
                f"Round {round_idx + 1}/{communication_rounds} | "
                f"agents={len(agents)} | reward={avg_reward:.3f} | "
                f"avg_channel_changes={avg_channel_changes:.3f} | eps={epsilon:.3f}"
                f"{buffer_msg}",
                flush=True,
            )

        if checkpoint_every and (round_idx + 1) % checkpoint_every == 0:
            _save_checkpoint(global_weights, history, round_idx, out_dir=save_dir)

        if early_check_at and early_check_callback and (round_idx + 1) == early_check_at:
            assessment = early_check_callback(history)
            history["early_assessment"] = assessment
            with open(os.path.join(save_dir, "early_assessment.json"), "w", encoding="utf-8") as f:
                json.dump(assessment, f, indent=2)
            if assessment.get("status") == "fail":
                history["stopped_early"] = True
                print(
                    f"Early stop at round {round_idx + 1}: {assessment}",
                    flush=True,
                )
                break

    _save_checkpoint(global_weights, history, round_idx, out_dir=save_dir)
    with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if inference_episodes and inference_episodes > 0 and global_weights is not None:
        print(
            f"Running {inference_episodes} inference episodes (rho={rho}, no weight updates)...",
            flush=True,
        )
        inference = run_inference_episodes(
            global_weights=global_weights,
            n_episodes=inference_episodes,
            history_length=history_length,
            number_of_channels=number_of_channels,
            min_nets=min_nets,
            max_nets=max_nets,
            learning_rate=learning_rate,
            batch_size=batch_size,
            fixed_topology=fixed_topology,
            fixed_number_of_nets=fixed_number_of_nets,
            scenario_map_seed=scenario_map_seed,
            scenario_name=scenario_name,
            rho=rho,
            seed=(int(seed) + 900_000) if seed is not None else None,
        )
        history["inference"] = inference
        with open(os.path.join(save_dir, "inference.json"), "w", encoding="utf-8") as f:
            json.dump(inference, f, indent=2)
        with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    with open(os.path.join(save_dir, "training_complete.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "run_name": run_name,
                "requested_communication_rounds": communication_rounds,
                "completed_rounds": len(history["round_rewards"]),
                "local_train_steps": local_train_steps,
                "fedprox_mu": fedprox_mu,
                "stopped_early": history["stopped_early"],
                "final_reward_mean_20": float(np.mean(history["round_rewards"][-20:])),
            },
            f,
            indent=2,
        )
    return history, global_weights


if __name__ == "__main__":
    run_federated_training(communication_rounds=1000)
