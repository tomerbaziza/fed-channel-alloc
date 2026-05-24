"""Federated aggregation utilities for CARLTON FRL."""

from collections import OrderedDict
from copy import deepcopy
import torch


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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        