"""Federated aggregation + centralized CARLTON training (pre-FL / paper CTDE)."""

import os
import pickle
from collections import OrderedDict
from copy import deepcopy

import torch

from Utils.utils import creat_player


def federated_averaging(local_state_dicts):
    """Compute FedAvg over local model weights."""
    if not local_state_dicts:
        raise ValueError("local_state_dicts is empty. Cannot run FedAvg.")

    global_state = OrderedDict()
    keys = local_state_dicts[0].keys()
    for key in keys:
        stacked = torch.stack([sd[key].detach().float().cpu() for sd in local_state_dicts], dim=0)
        global_state[key] = torch.mean(stacked, dim=0)
    return global_state


def broadcast_global_weights(agents, global_state_dict):
    """Load one global weight set into all agents."""
    for agent in agents.values():
        agent.load_state_dict(deepcopy(global_state_dict))


def collect_local_weights(agents):
    """Collect local `state_dict()` snapshots from all agents."""
    return [agent.state_dict() for agent in agents.values()]


def train_model(
    trainig_iterations=40,
    batch_size=32,
    action_space=10,
    learning_rate=0.00025,
    history=1,
    max_experience=100000,
    sensing_window=5,
    number_of_layers=3,
    number_of_nodes=128,
    mellowmax_constant=0.02,
    gamma=0.9,
    dropout=False,
    l2_regularization=False,
    i_d_folder="pre_fl",
    base_dir=None,
    verbose=False,
):
    """Centralized CTDE update on global replay buffer (paper Algorithm 4)."""
    base_dir = base_dir or os.getcwd()
    folder_name = os.path.join(base_dir, "Global_RB_Storage_" + str(i_d_folder))
    if not os.path.isdir(folder_name):
        if verbose:
            print("No global replay folder:", folder_name)
        return None

    folder_path = folder_name + os.sep
    files = sorted(os.listdir(folder_path), key=lambda t: os.stat(folder_path + t).st_mtime)
    if not files:
        return None

    with open(os.path.join(folder_path, files[-1]), "rb") as f:
        global_rb = pickle.load(f)

    agent = creat_player(
        number_of_actions=action_space,
        history_length=history,
        learning_rate=learning_rate,
        max_experience=max_experience,
        batch_size=batch_size,
        sensing_window=sensing_window,
        number_of_layers=number_of_layers,
        number_of_nodes=number_of_nodes,
        mellowmax_constant=mellowmax_constant,
        gamma=gamma,
        dropout=dropout,
        l2_regularization=l2_regularization,
        i_d=9999,
        i_d_folder=i_d_folder,
        verbose=verbose,
    )

    global_rb.define_new_batch_size(batch_size)
    agent.experience_replay_buffer = global_rb

    if global_rb.count < batch_size:
        if verbose:
            print(f"Skip train: buffer count {global_rb.count} < batch {batch_size}")
        return None

    costs = []
    for _ in range(trainig_iterations):
        costs.append(float(agent.learn()))
        if verbose:
            print("  learn cost:", costs[-1])

    weights_folder = os.path.join(base_dir, "Train_weights_" + str(i_d_folder))
    os.makedirs(weights_folder, exist_ok=True)
    tag = global_rb.current
    path = os.path.join(weights_folder, f"trained_weights_step_{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump(agent.get_model_weights(), f)

    return costs
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        