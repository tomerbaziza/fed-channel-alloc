"""Merge the re-evaluated PyTorch CTDE rows into the Fig. 15 summary.

The original sweep scored PyTorch CTDE with an early checkpoint, because
`_latest_pytorch_weights` ordered unpadded step tags lexicographically. After
fixing the ordering the CTDE games were re-run in four shards (`fig15_fix_*`).
This script swaps those rows into the existing evaluation, keeping the
heuristic and federated rows untouched, and regenerates the figures.

Usage:
  py -3 experiments/merge_fig15_fix.py --study-dir <study>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.eval_paper_fig15 import aggregate, plot_fig13_curves, plot_fig15_bars


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--shard-glob", default="fig15_fix_*")
    parser.add_argument("--out-subdir", default="paper_fig15")
    args = parser.parse_args(argv)

    study_dir = Path(args.study_dir).resolve()
    out_dir = study_dir / args.out_subdir

    rows = json.loads((out_dir / "eval_rows.json").read_text())
    kept = [r for r in rows if r["method"] != "pytorch"]

    fixed = []
    shards = sorted(study_dir.glob(f"{args.shard_glob}/eval_rows.json"))
    if not shards:
        raise SystemExit(f"no shards matched {args.shard_glob}")
    for shard in shards:
        shard_rows = json.loads(shard.read_text())
        fixed.extend(r for r in shard_rows if r["method"] == "pytorch")
        print(f"{shard.parent.name}: {len(shard_rows)} rows")

    ns = sorted({r["networks"] for r in fixed})
    print(f"replacing {len(rows) - len(kept)} stale pytorch rows with {len(fixed)}")
    print(f"pytorch N coverage: {ns}")

    merged = kept + fixed
    n_range = sorted({r["networks"] for r in merged})
    summary = aggregate(merged, n_range=n_range)
    summary["meta"] = {
        "study_dir": str(study_dir),
        "n_range": n_range,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "PyTorch CTDE re-evaluated with final weights (checkpoint ordering fix)",
    }

    (out_dir / "eval_rows.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_fig15_bars(summary, out_dir / "fig15_composite_bars.png")
    plot_fig13_curves(summary, out_dir / "fig13_composite_vs_n.png")

    print(f"\n{'Method':20s} {'In':>8s} {'Out':>8s} {'All':>8s} {'WS':>8s} {'CTS':>8s}")
    for _m, s in summary["methods"].items():
        print(
            f"{s['label']:20s} {s['composite_in']:8.3f} {s['composite_out']:8.3f} "
            f"{s['composite_all']:8.3f} {s['ws_all']:8.3f} {s['cts_all']:8.3f}"
        )


if __name__ == "__main__":
    main()
