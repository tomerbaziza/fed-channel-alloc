"""Generate the scenario-illustration figure used in the paper (Section II).

Samples a topology with the same generator used by the simulator, marks each
network manager (the user minimizing the total intra-network distance), and
draws the neighbor links induced by the threshold Gamma.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.RandomLocationOfNetworks import set_random_location_of_networks

GAMMA = 500.0
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "figs",
    "scenario_illustration.pdf",
)


def sample_scenario(number_of_nets, seed):
    np.random.seed(seed)
    users_per_net, centers = set_random_location_of_networks(number_of_nets)
    nets = []
    for n in range(number_of_nets):
        mean_x, mean_y, std_x, std_y = centers[n]
        m = users_per_net[n]
        xs = np.random.normal(mean_x, std_x, m)
        ys = np.random.normal(mean_y, std_y, m)
        pts = np.stack([xs, ys], axis=1)
        dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1).sum(axis=1)
        nets.append(
            {
                "points": pts,
                "manager": int(np.argmin(dist)),
                "center": np.array([mean_x, mean_y]),
            }
        )
    return nets


def shift_to_positive(nets, margin=100.0):
    """Translate the whole deployment so all plotted coordinates are positive."""
    all_pts = np.concatenate([net["points"] for net in nets] +
                             [net["center"][None, :] for net in nets])
    offset = margin - all_pts.min(axis=0)
    for net in nets:
        net["points"] = net["points"] + offset
        net["center"] = net["center"] + offset
    return nets


def main():
    nets = sample_scenario(number_of_nets=6, seed=32)
    nets = shift_to_positive(nets)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    for i in range(len(nets)):
        for j in range(i + 1, len(nets)):
            d = np.linalg.norm(nets[i]["center"] - nets[j]["center"])
            if d <= GAMMA:
                pair = np.stack([nets[i]["center"], nets[j]["center"]])
                ax.plot(pair[:, 0], pair[:, 1], color="0.55", lw=1.0,
                        ls="--", zorder=1)

    for n, net in enumerate(nets):
        pts = net["points"]
        c = colors[n % 10]
        ax.scatter(pts[:, 0], pts[:, 1], s=26, color=c, edgecolor="white",
                   linewidth=0.5, zorder=3, label=rf"$\mathcal{{N}}^{{{n + 1}}}$")
        mx, my = pts[net["manager"]]
        ax.scatter([mx], [my], s=150, marker="*", color=c, edgecolor="black",
                   linewidth=0.7, zorder=4)
        ax.annotate(rf"Ma$^{{{n + 1}}}$", (mx, my), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, fontweight="bold",
                    color="black", zorder=5)

    ax.plot([], [], marker="*", ls="none", color="0.35", markersize=11,
            markeredgecolor="black", label="network manager")
    ax.plot([], [], color="0.55", lw=1.0, ls="--",
            label=rf"neighbors ($\leq \Gamma = {int(GAMMA)}$ m)")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
