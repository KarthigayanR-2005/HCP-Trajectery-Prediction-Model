"""
Real evaluation for the HCP + MTR trajectory prediction model.

Unlike the previous evaluate.py (which never touched the model or real data —
it just wrote out a hardcoded, fabricated JSON), this script:
  1. Loads an actual trained checkpoint.
  2. Runs real inference on real nuScenes data.
  3. Computes genuine minADE / minFDE / Miss Rate from the model's actual
     predictions vs. ground truth.
  4. Can run with HCP pruning ON or OFF, so you can produce the real
     "Ours (HCP+MTR)" vs. "No-HCP (MTR baseline)" comparison this project's
     research question is actually about, instead of made-up numbers.

HONEST LIMITATION: there is currently no held-out validation split — the
model was trained on the full dataset with nothing set aside. So numbers
from this script are an optimistic proxy (the model may be partly recalling
data it trained on), not a true measure of generalization to unseen data.
That requires a real train/val split, which is a separate, future step.
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hcp_project.data.dataset_router import DatasetRouter
from hcp_project.data.dataset_streamer import build_streaming_dataloader
from hcp_project.mtr_core.train import MTRMotionTransformer, load_state_dict_forgiving
from hcp_project.hcp.pruner import (
    HierarchicalCombinatorialPruner, generate_kinematic_candidates,
    generate_anchor_candidates,
)


def compute_ade_fde_mr(predictions, ground_truth, miss_threshold=2.0, confidences=None):
    """
    predictions : (N, K, T, 2)
    ground_truth: (N, T, 2)
    confidences : (N, K) or None. A mode with confidence exactly 0 was pruned by
                  HCP and never predicted, so it must not be scored.

    Returns per-agent arrays (ades, fdes, is_miss, n_modes_used) — NOT yet
    averaged, so the caller can accumulate across batches.

    Why confidences matter here
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This function used to take the minimum over ALL K modes and ignore
    confidence entirely. That made minADE/minFDE/miss-rate mathematically
    incapable of responding to pruning: the HCP mask only ever touched
    confidence, so both configurations produced byte-identical numbers (see
    eval_real_20260910_163553.json, where "Ours" and the baseline agree to every
    decimal place). That is not a null result about pruning -- it is the metric
    being blind to it by construction.

    Scoring only the modes a configuration actually predicted is what makes the
    accuracy half of the project's research question measurable at all.
    """
    N, K, T, _ = predictions.shape
    dists = np.linalg.norm(predictions - ground_truth[:, None, :, :], axis=-1)  # (N, K, T)
    mode_ade = dists.mean(axis=-1)   # (N, K)
    mode_fde = dists[:, :, -1]       # (N, K)

    if confidences is not None:
        alive = confidences > 0.0                     # (N, K)
        # An agent with nothing left alive keeps its best mode, matching the
        # pruner's own "at least one survivor" fallback.
        none_alive = ~alive.any(axis=1)
        if none_alive.any():
            alive = alive.copy()
            alive[none_alive, 0] = True
        mode_ade = np.where(alive, mode_ade, np.inf)
        mode_fde = np.where(alive, mode_fde, np.inf)
        n_modes_used = alive.sum(axis=1).astype(np.float32)
    else:
        n_modes_used = np.full((N,), float(K), dtype=np.float32)

    min_ade = mode_ade.min(axis=1)   # (N,)
    min_fde = mode_fde.min(axis=1)   # (N,)
    is_miss = (min_fde > miss_threshold).astype(np.float32)
    return min_ade, min_fde, is_miss, n_modes_used


def build_batch_tensors(collated, device):
    """Reuses the exact padding/reconstruction logic from train.py's training
    loop, so evaluation sees data shaped identically to training."""
    packed_hist  = collated["history_traj"]
    packed_fut   = collated["future_traj"]
    batch_splits = collated["batch_splits"]
    B     = len(batch_splits)
    N_max = max(batch_splits)
    T_hist = packed_hist.shape[1]
    T_fut  = packed_fut.shape[1]

    hist_padded = torch.zeros((B, N_max, T_hist, 6), dtype=torch.float32)
    fut_padded  = torch.zeros((B, N_max, T_fut,  5), dtype=torch.float32)
    cursor = 0
    for b, n in enumerate(batch_splits):
        hist_padded[b, :n] = packed_hist[cursor:cursor + n]
        fut_padded[b,  :n] = packed_fut[cursor:cursor + n]
        cursor += n
    hist_padded = hist_padded.to(device)
    fut_padded  = fut_padded.to(device)

    camera_images  = collated.get("camera_images")
    has_image_mask = collated.get("has_image")
    if camera_images is not None:
        camera_images = camera_images.to(device)

    map_tensors = collated["map_tensors"]
    map_splits  = collated["map_splits"]
    map_polylines_batch, mc = [], 0
    for nm in map_splits:
        map_polylines_batch.append([t.numpy() for t in map_tensors[mc:mc + nm]])
        mc += nm

    return hist_padded, fut_padded, camera_images, has_image_mask, map_polylines_batch, batch_splits


def build_hcp_mask(fut_padded, hist_padded, map_polylines_batch, batch_splits, pruner,
                   device, anchors):
    """Candidates are generated from the decoder's own anchors so that candidate k
    and decoder mode k are the same manoeuvre -- see generate_anchor_candidates."""
    B, N_max, T_fut, _ = fut_padded.shape
    K = anchors.shape[0]
    dense_candidates = torch.zeros((B, N_max, K, T_fut, 5), device=device)
    for b in range(B):
        for n in range(batch_splits[b]):
            dense_candidates[b, n] = generate_anchor_candidates(
                hist_padded[b, n], anchors, T_fut=T_fut, dt=0.5)
    hcp_masks = []
    for b in range(B):
        _, mask, _ = pruner(
            dense_candidates[b, :batch_splits[b]],
            hist_padded[b, :batch_splits[b]],
            map_polylines_batch[b],
        )
        pad_rows = N_max - batch_splits[b]
        if pad_rows > 0:
            mask = torch.cat([mask, torch.zeros((pad_rows, K), dtype=torch.bool, device=device)], dim=0)
        hcp_masks.append(mask)
    return torch.stack(hcp_masks)


def run_evaluation(checkpoint_path, nuscenes_dir, waymo_dir, num_samples=2000,
                    batch_size=2, use_hcp=True, device=None, num_workers=0,
                    scene_filter=None, scene_filter_label=None,
                    n_modes=6, anchors_path=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    print("Building DatasetRouter (loads/processes nuScenes metadata)...")
    if scene_filter is not None:
        print(f"Restricting evaluation to scene split: {scene_filter_label or 'custom'} "
              f"({len(scene_filter)} scenes) — NOTE: if these scenes were included in "
              f"training data, this is not a true held-out generalization test, just "
              f"an evaluation restricted to that scene subset.")
    base_dataset = DatasetRouter(nuscenes_dir, waymo_dir, mode="nuscenes", scene_filter=scene_filter)
    dataloader = build_streaming_dataloader(
        list(range(len(base_dataset))), base_dataset,
        batch_size=batch_size, num_workers=num_workers, shuffle=False,
    )

    print(f"Loading model from checkpoint: {checkpoint_path}")
    anchors_np = np.load(anchors_path) if anchors_path else None
    model = MTRMotionTransformer(d_model=256, n_modes=n_modes, anchors=anchors_np).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Cross-check what the checkpoint was trained on against what we are about
    # to evaluate on. Evaluating on scenes the model trained on is not a
    # generalisation measure, and it is easy to do by accident.
    trained_on = ckpt.get("train_split") if isinstance(ckpt, dict) else None
    if trained_on is None:
        print("\nNOTE: this checkpoint records no training split. It predates split "
              "support, which means it was almost certainly trained on all 850 "
              "scenes -- so nothing below is a generalisation measure.\n")
    elif trained_on == "all":
        print("\nWARNING: this checkpoint was trained on ALL scenes. Whatever split "
              "is evaluated here was seen during training; these numbers cannot "
              "support a generalisation claim.\n")
    elif scene_filter_label and trained_on in str(scene_filter_label):
        print(f"\nWARNING: evaluating on the '{trained_on}' split, which is the split "
              f"this checkpoint was TRAINED on. These are training-set numbers.\n")
    else:
        print(f"\nCheckpoint was trained on the '{trained_on}' split; "
              f"evaluating on '{scene_filter_label or 'all scenes'}'.\n")
    missing, unexpected, skipped = load_state_dict_forgiving(
        model, state_dict, context=f"evaluating {os.path.basename(checkpoint_path)}")
    if missing or unexpected:
        print(f"Note: {len(missing)} param(s) not found in checkpoint (fresh-initialised), "
              f"{len(unexpected)} unused param(s) in checkpoint ignored.")
    if skipped:
        print("Evaluating a checkpoint with freshly-initialised layers will produce "
              "meaningless metrics. Retrain before trusting anything below.")
    model.eval()

    pruner = HierarchicalCombinatorialPruner().to(device) if use_hcp else None

    all_ades, all_fdes, all_misses, all_modes_used = [], [], [], []
    n_evaluated = 0
    latencies_ms = []

    with torch.no_grad():
        for step_idx, collated in enumerate(dataloader):
            if n_evaluated >= num_samples:
                break
            hist_padded, fut_padded, camera_images, has_image_mask, map_polylines_batch, batch_splits = \
                build_batch_tensors(collated, device)

            hcp_mask = None
            if use_hcp:
                hcp_mask = build_hcp_mask(fut_padded, hist_padded, map_polylines_batch,
                                          batch_splits, pruner, device,
                                          model.decoder.intention_anchors)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            pred_trajs, confidences = model(
                hist_padded, map_polylines_batch, hcp_mask,
                camera_images=camera_images, has_image_mask=has_image_mask,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.time() - t0) * 1000.0

            B, N_max = hist_padded.shape[0], hist_padded.shape[1]
            n_real_this_batch = sum(batch_splits)
            latencies_ms.append(elapsed_ms / max(n_real_this_batch, 1))  # per-agent latency

            pred_np = pred_trajs[..., :2].cpu().numpy()   # (B, N_max, K, T, 2)
            gt_np   = fut_padded[..., :2].cpu().numpy()   # (B, N_max, T, 2)
            conf_np = confidences.cpu().numpy()            # (B, N_max, K)

            for b in range(B):
                n = batch_splits[b]
                if n == 0:
                    continue
                ades, fdes, misses, modes_used = compute_ade_fde_mr(
                    pred_np[b, :n], gt_np[b, :n], confidences=conf_np[b, :n])
                all_ades.extend(ades.tolist())
                all_fdes.extend(fdes.tolist())
                all_misses.extend(misses.tolist())
                all_modes_used.extend(modes_used.tolist())
                n_evaluated += n

            if step_idx % 50 == 0:
                print(f"  ...{n_evaluated}/{num_samples} agents evaluated")

    if scene_filter is not None:
        note = (f"Evaluated ONLY on the {scene_filter_label} ({len(scene_filter)} scenes). "
                f"HONEST CAVEAT: if this checkpoint was trained on the full dataset, these "
                f"scenes were already seen during training — this is a soft/approximate "
                f"check, not a true held-out generalization test.")
    else:
        note = ("Evaluated on the same data distribution used for training — "
                "no scene filtering applied, so these numbers are an optimistic proxy, "
                "not a generalization measure at all.")

    results = {
        "minADE": float(np.mean(all_ades)) if all_ades else None,
        "minFDE": float(np.mean(all_fdes)) if all_fdes else None,
        "miss_rate_2m": float(np.mean(all_misses)) if all_misses else None,
        "num_agents_evaluated": n_evaluated,
        "avg_latency_ms_per_agent": float(np.mean(latencies_ms)) if latencies_ms else None,
        "avg_modes_scored": float(np.mean(all_modes_used)) if all_modes_used else None,
        "hcp_pruning_used": use_hcp,
        "checkpoint": checkpoint_path,
        "scene_filter_used": scene_filter_label,
        "NOTE": note,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real evaluation of the HCP+MTR checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nuscenes_dir", type=str, default="hcp_project/data/nuscenes")
    parser.add_argument("--waymo_dir", type=str, default="hcp_project/data/waymo")
    parser.add_argument("--num_samples", type=int, default=2000,
                         help="Number of individual agent trajectories to evaluate on.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_modes", type=int, default=6,
                         help="Must match the trained checkpoint.")
    parser.add_argument("--anchors", type=str, default=None,
                         help="Path to the same .npy anchor bank the checkpoint was trained with.")
    parser.add_argument("--compare_hcp", action="store_true",
                         help="Run twice — once with HCP pruning, once without — "
                              "and print both, for the real Ours-vs-baseline comparison.")
    parser.add_argument("--split", type=str, default=None,
                         choices=["train", "val", "all"],
                         help="Which official nuScenes scene split to evaluate on. "
                              "Use 'val' for a genuine held-out measurement of a "
                              "checkpoint trained with --split train.")
    parser.add_argument("--val_split_only", action="store_true",
                         help="Restrict evaluation to the official nuScenes val split "
                              "(150 scenes) via nuscenes-devkit's create_splits_scenes(). "
                              "HONEST CAVEAT: if this checkpoint was trained on the full "
                              "dataset (all 850 scenes, train+val together), this is NOT "
                              "a true held-out generalization test — the model has already "
                              "seen this data. It's a soft/approximate check, not a rigorous "
                              "one; a genuinely valid test would require training a separate "
                              "model using only the official train split.")
    args = parser.parse_args()

    scene_filter, scene_filter_label = None, None
    if args.split and args.split != "all":
        from nuscenes.utils.splits import create_splits_scenes
        scene_filter = set(create_splits_scenes()[args.split])
        scene_filter_label = f"official nuScenes {args.split} split"
        print(f"Evaluating on the official '{args.split}' split "
              f"({len(scene_filter)} scenes).")
    elif args.val_split_only:
        from nuscenes.utils.splits import create_splits_scenes
        scene_filter = set(create_splits_scenes()["val"])
        scene_filter_label = "official nuScenes val split"
        print(f"--val_split_only set: restricting to {len(scene_filter)} official val scenes. "
              f"Remember — if this checkpoint trained on the full dataset, this is a soft "
              f"check, not a true held-out test.")

    output_dir = "hcp_project/outputs"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = {}
    configs = [("Ours (HCP+MTR)", True), ("No-HCP (MTR baseline)", False)] if args.compare_hcp \
        else [("Ours (HCP+MTR)", True)]

    for name, use_hcp in configs:
        print(f"\n=== Evaluating: {name} ===")
        result = run_evaluation(
            args.checkpoint, args.nuscenes_dir, args.waymo_dir,
            num_samples=args.num_samples, batch_size=args.batch_size, use_hcp=use_hcp,
            scene_filter=scene_filter, scene_filter_label=scene_filter_label,
            n_modes=args.n_modes, anchors_path=args.anchors,
        )
        all_results[name] = result
        print(f"minADE: {result['minADE']:.4f} | minFDE: {result['minFDE']:.4f} | "
              f"Miss Rate (2m): {result['miss_rate_2m']:.4f} | "
              f"Latency: {result['avg_latency_ms_per_agent']:.2f} ms/agent")

    json_path = os.path.join(output_dir, f"eval_real_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nReal evaluation results saved to {json_path}")