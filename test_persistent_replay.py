"""Verify persistent local replay grows across FL communication rounds."""

from main_v3 import run_federated_training


def main():
    history, _ = run_federated_training(
        communication_rounds=15,
        local_train_steps=1,
        fedprox_mu=0.01,
        batch_size=32,
        replay_memory_size=10000,
        persistent_replay=True,
        save_dir="experiments/results/_tests/persistent_replay_check",
        run_name="persistent_replay_check",
        seed=42,
        checkpoint_every=0,
    )

    counts = history["replay_buffer_counts"]
    per_client = history["replay_buffer_counts_per_client"]
    print("\n=== Replay buffer growth ===")
    for i, count in enumerate(counts, start=1):
        print(f"  round {i:2d}: replay_count={count}")

    assert counts[-1] > counts[0], "Replay buffer should grow across rounds."
    assert counts[-1] >= 32, f"Expected replay_count >= batch_size (32), got {counts[-1]}"

    # Each client owns a private buffer: it may only grow on rounds in which the
    # client actually participated, and the per-client counts must partition the
    # total.
    final = per_client[-1]
    assert sum(final.values()) == counts[-1], "Per-client counts must sum to the total."
    for round_idx, snapshot in enumerate(per_client):
        n_agents = history["round_num_agents"][round_idx]
        previous = per_client[round_idx - 1] if round_idx else {k: 0 for k in snapshot}
        for client_id, count in snapshot.items():
            if client_id >= n_agents:
                assert count == previous[client_id], (
                    f"client {client_id} gained data in round {round_idx} "
                    f"without participating"
                )

    print(f"\nPASS: final replay_count={counts[-1]} (batch_size=32)")
    print(f"PASS: private per-client buffers {final}")


if __name__ == "__main__":
    main()
