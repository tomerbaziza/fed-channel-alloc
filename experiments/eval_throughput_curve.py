"""Shannon throughput of the federated policies as a function of training round.

For every saved global checkpoint we freeze the weights, play a fixed set of
frozen-policy scenarios, and read the SINR each network actually achieves on
the channel it settled on. The per-network rate is the Shannon-Hartley capacity
of a single CARLTON channel,

    R^n = B * log2(1 + SINR^n),      B = 2 MHz,

averaged over the users of the network. We report both the network-average rate
and the rate of the worst-served network of the scenario, the latter being the
throughput analogue of min_CQ.

Because the campaign checkpoints start at round 100 -- long after the policy
has left its random initialization -- the early part of the curve comes from a
dense re-run of the same seeds (`jobs/dense_ckpt_job.py`), passed via
`--dense-dir`.

Usage:
  py -3 experiments/eval_throughput_curve.py --study-dir <study> --method fedavg \\
      --seed-index 0 --dense-dir <dense> --episodes 12
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BuildingBlocks.Worker import worker
from SimulationEnvironments.Env_Utiles import db_to_watts, dbm_to_watts
from SimulationEnvironments.Pythonic_Environment import python_env
from Utils.RandomLocationOfNetworks import set_random_location_of_networks

K = 10
BANDWIDTH_HZ = 2e6           # Table II: 2 MHz per channel
THERMAL_NOISE_DB = -134.9    # matches create_sensed_vector
MIN_NETS, MAX_NETS = 2, 7


def network_rates_mbps(env) -> np.ndarray:
    """Per-network Shannon rate on the finally occupied channels [Mbps]."""
    rates = []
    for net in env.nets:
        noise_dbm = net.create_noise_matrix(env.nets)
        noise_w = dbm_to_watts(noise_dbm) + db_to_watts(THERMAL_NOISE_DB)
        sinr = db_to_watts(net.pr_min) / noise_w          # (users, channels), linear
        sinr_on_channel = sinr[:, int(net.channel)]
        rate_bps = BANDWIDTH_HZ * np.log2(1.0 + sinr_on_channel)
        rates.append(float(np.mean(rate_bps)) / 1e6)
    return np.asarray(rates)


def scenario_rate_mbps(env) -> float:
    return float(np.mean(network_rates_mbps(env)))


def play_episode(weights, n_nets: int, seed: int):
    np.random.seed(int(seed))
    users, centers = set_random_location_of_networks(n_nets)
    env = python_env(
        number_of_nets=n_nets,
        number_of_users_in_each_net=users,
        net_center_location_and_std=centers,
        possible_channels=K,
        add_noise=False,
        training=False,
    )
    worker(
        address_scen="",
        scenario=env,
        address_algo="",
        training=False,
        epsilon=0.0,
        global_weights=weights,
        local_train_steps=0,
        save_to_global_rb=False,
        verbose=False,
    )
    rates = network_rates_mbps(env)
    return float(np.mean(rates)), float(np.min(rates))


def round_of(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def collect_checkpoints(seed_dir: Path, dense_seed_dir: Path | None):
    """Dense early checkpoints (if available) plus the campaign's coarse ones."""
    by_round = {}
    if dense_seed_dir is not None and dense_seed_dir.exists():
        for p in dense_seed_dir.glob("global_weights_round_*.pkl"):
            by_round[round_of(p)] = p
    for p in seed_dir.glob("global_weights_round_*.pkl"):
        by_round.setdefault(round_of(p), p)
    return [(r, by_round[r]) for r in sorted(by_round)]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed-index", type=int, required=True)
    parser.add_argument("--dense-dir", default=None)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=777001)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    study_dir = Path(args.study_dir).resolve()
    seed_dir = sorted((study_dir / args.method).glob("seed_*"))[args.seed_index]
    dense_seed_dir = None
    if args.dense_dir:
        dense_seed_dir = Path(args.dense_dir).resolve() / args.method / seed_dir.name

    # Every checkpoint is scored on an identical scenario set, so the only thing
    # that varies along the curve is the policy, not the topology draw.
    games = []
    for game in range(args.episodes):
        seed = args.base_seed + game
        rng = np.random.RandomState(seed)
        games.append((seed, int(rng.randint(MIN_NETS, MAX_NETS + 1))))

    rows = {}
    for rnd, ckpt in collect_checkpoints(seed_dir, dense_seed_dir):
        with ckpt.open("rb") as f:
            weights = pickle.load(f)
        means, mins = [], []
        for seed, n_nets in games:
            m, lo = play_episode(weights, n_nets, seed)
            means.append(m)
            mins.append(lo)
        rows[rnd] = {"mean": means, "min": mins}
        print(
            f"  {args.method} {seed_dir.name} round {rnd}: "
            f"mean={np.mean(means):.2f} worst={np.mean(mins):.2f} Mbps",
            flush=True,
        )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.method}_{seed_dir.name}.json"
    out.write_text(
        json.dumps(
            {
                "method": args.method,
                "seed_dir": str(seed_dir),
                "bandwidth_hz": BANDWIDTH_HZ,
                "episodes": args.episodes,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "rounds": {str(r): v for r, v in sorted(rows.items())},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
