"""
compute_anchors.py
------------------
Derive intention-point anchors from the training data by k-means, and save them
for MTRDecoder to use.

Why bother
~~~~~~~~~~
The decoder's fallback anchors are seven hand-picked points. Two problems:

  * They do not reflect how agents in this dataset actually move. The
    winner-takes-all loss assigns every ground-truth future to its nearest
    anchor, so badly-placed anchors mean badly-assigned targets.
  * K = 7 is far too small for pruning to pay off. Only ~24% of inference
    depends on K at that size, so even discarding every mode but one cannot cut
    total cost by more than a quarter -- while the pruner itself costs more than
    that. Raising K to ~64 takes the K-dependent share past 75%, which is the
    regime where pruning is actually a meaningful idea.

Endpoints are collected in each agent's OWN frame (origin at its current
position, +x along its current heading), which is the frame the decoder
predicts in -- see utils/geometry.

Usage
~~~~~
    python hcp_project/tools/compute_anchors.py --n_modes 64
    python -m hcp_project.mtr_core.train --anchors hcp_project/outputs/anchors_64.npy ...
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hcp_project.data.dataset_router import DatasetRouter
from hcp_project.utils.geometry import build_agent_frames, scene_to_local


def collect_endpoints(dataset, max_slices=None, stride=1):
    """Gather every agent's future endpoint, expressed in that agent's own frame."""
    endpoints = []
    n = len(dataset)
    limit = min(n, max_slices) if max_slices else n
    for i in range(0, limit, stride):
        try:
            batch = dataset[i]
        except Exception:
            continue
        hist = torch.as_tensor(np.asarray(batch.history_traj), dtype=torch.float32)
        fut = torch.as_tensor(np.asarray(batch.future_traj), dtype=torch.float32)
        if hist.ndim != 3 or fut.ndim != 3 or hist.shape[0] == 0:
            continue
        origin, theta = build_agent_frames(hist.unsqueeze(0))
        local = scene_to_local(fut.unsqueeze(0), origin, theta)
        endpoints.append(local[0, :, -1, :2].numpy())
        if (i // max(stride, 1)) % 2000 == 0:
            print(f"  ...{i}/{limit} slices, {sum(len(e) for e in endpoints)} endpoints")
    if not endpoints:
        raise RuntimeError("No endpoints collected -- is the dataset extracted?")
    return np.concatenate(endpoints, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Derive decoder anchors by k-means")
    ap.add_argument("--n_modes", type=int, default=64)
    ap.add_argument("--nuscenes_dir", type=str, default="hcp_project/data/nuscenes")
    ap.add_argument("--waymo_dir", type=str, default="hcp_project/data/waymo")
    ap.add_argument("--max_slices", type=int, default=40000,
                    help="Cap on trajectory slices sampled. The endpoint distribution "
                         "converges long before the full dataset is needed.")
    ap.add_argument("--stride", type=int, default=5,
                    help="Sample every Nth slice. Consecutive slices of one track "
                         "overlap heavily, so a stride avoids over-weighting them.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print("Building DatasetRouter ...")
    ds = DatasetRouter(args.nuscenes_dir, args.waymo_dir, mode="nuscenes")

    print(f"Collecting future endpoints (stride {args.stride}, cap {args.max_slices}) ...")
    pts = collect_endpoints(ds, max_slices=args.max_slices, stride=args.stride)
    print(f"Collected {len(pts):,} endpoints.")

    from sklearn.cluster import KMeans
    print(f"Running k-means for K={args.n_modes} ...")
    km = KMeans(n_clusters=args.n_modes, n_init=10, random_state=0).fit(pts)
    anchors = km.cluster_centers_.astype(np.float32)

    # Sort by reach then bearing, so the bank has a stable, readable order.
    order = np.lexsort((np.arctan2(anchors[:, 1], anchors[:, 0]),
                        np.linalg.norm(anchors, axis=1)))
    anchors = anchors[order]

    out = args.out or f"hcp_project/outputs/anchors_{args.n_modes}.npy"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, anchors)

    d = np.linalg.norm(pts[:, None, :] - anchors[None, :, :], axis=-1).min(axis=1)
    print(f"\nSaved {anchors.shape[0]} anchors to {out}")
    print(f"  reach     : {np.linalg.norm(anchors,axis=1).min():.1f} .. "
          f"{np.linalg.norm(anchors,axis=1).max():.1f} m")
    print(f"  left/right: {(anchors[:,1]>0.5).sum()} / {(anchors[:,1]<-0.5).sum()} "
          f"(a roughly even split means the bank is not biased to one side)")
    print(f"  mean distance from a real endpoint to its nearest anchor: {d.mean():.2f} m")
    print(f"  (this is the floor on minADE imposed by the anchor bank alone)")
    print(f"\nTrain with:\n  python -m hcp_project.mtr_core.train --anchors {out} ...")


if __name__ == "__main__":
    main()
