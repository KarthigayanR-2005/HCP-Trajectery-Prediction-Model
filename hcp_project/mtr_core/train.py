"""
train.py — HCP + MTR training loop (optimised)
------------------------------------------------
Integrations vs. the original:
  1. ``LargeScaleDrivingStreamer`` + ``dynamic_collate_fn`` replace the
     plain ``DataLoader(DatasetRouter, …)`` call → no max-agent padding,
     multi-worker streaming, fault-tolerant loading.
  2. ``ScaleTrainingManager`` wraps the optimizer → AMP (float16/bfloat16)
     + gradient accumulation.  Manual ``loss.backward(); optimizer.step()``
     blocks are removed.
  3. Map tokenisation distance computation is fully vectorised using
     ``torch.cdist`` — the old O(M × N) Python loop is gone.
  4. New CLI args: ``--grad_accum`` (default 4), ``--num_workers`` (default 2).
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from hcp_project.data.dataset_router import DatasetRouter
from hcp_project.data.dataset_streamer import (
    LargeScaleDrivingStreamer,
    dynamic_collate_fn,
    build_streaming_dataloader,
)
from hcp_project.mtr_core.tokenizer import MapTokenizer, AgentTokenizer
from hcp_project.mtr_core.encoder import MTREncoder
from hcp_project.fusion.fusion import CrossAttentionFusionLayer
from hcp_project.mtr_core.decoder import MTRDecoder
from hcp_project.hcp.pruner import HierarchicalCombinatorialPruner
from hcp_project.utils.mixed_precision import ScaleTrainingManager


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MTRMotionTransformer(nn.Module):
    def __init__(self, d_model: int = 256, n_modes: int = 6, T_fut: int = 12):
        super().__init__()
        self.map_tokenizer   = MapTokenizer(in_channels=3,  d_model=d_model)
        self.agent_tokenizer = AgentTokenizer(in_channels=6, d_model=d_model)
        self.encoder         = MTREncoder(d_model=d_model, nhead=8, num_layers=4)
        self.fusion          = CrossAttentionFusionLayer(d_model=d_model, nhead=8)
        self.decoder         = MTRDecoder(d_model=d_model, n_modes=n_modes, T_fut=T_fut)

    def forward(self, history_traj, map_polylines, hcp_mask=None):
        """
        Args:
            history_traj  : (B, N, T_hist, 6)
            map_polylines : list[list[np.ndarray]] — outer = batch, inner = polylines
            hcp_mask      : (B, N, n_modes) bool, optional
        """
        B, N, T_hist, _ = history_traj.shape
        device = history_traj.device

        # 1. Tokenise agent histories -------------------------------------------
        flat_hist   = history_traj.view(B * N, T_hist, -1)
        agent_tokens = self.agent_tokenizer(flat_hist)   # (B*N, d_model)
        agent_tokens = agent_tokens.view(B, N, -1)        # (B, N, d_model)

        # 2. Tokenise map polylines ----------------------------------------------
        max_polys = 20
        padded_map = torch.zeros((B, max_polys, 10, 3), device=device)           # (B, M, P, C)
        distances  = torch.zeros((B, max_polys, N),      device=device)           # (B, M, N)

        for b in range(B):
            polys = map_polylines[b]
            for m in range(min(len(polys), max_polys)):
                poly = polys[m]
                pts  = poly[:, :3] if hasattr(poly, '__len__') else np.asarray(poly)[:, :3]
                if len(pts) > 10:
                    pts = pts[:10]
                elif len(pts) < 10:
                    pts = np.pad(pts, ((0, 10 - len(pts)), (0, 0)), mode='edge')
                padded_map[b, m] = torch.tensor(pts, dtype=torch.float32, device=device)

            # --- Vectorised distance computation (replaces per-(m,n) Python loop) ---
            # poly_centers : (max_polys, 2) — mean of (x, y) over the 10 padded points
            poly_centers = padded_map[b, :, :, :2].mean(dim=1)                    # (M, 2)
            # agent_pos    : (N, 2) — last history step
            agent_pos    = history_traj[b, :, -1, :2]                             # (N, 2)
            # torch.cdist computes pairwise L2; result is (M, N)
            distances[b] = torch.cdist(poly_centers.float(), agent_pos.float())   # (M, N)

        # Tokenise map
        flat_map   = padded_map.view(B * max_polys, 10, -1)
        map_tokens = self.map_tokenizer(flat_map)    # (B*M, d_model)
        map_tokens = map_tokens.view(B, max_polys, -1)  # (B, M, d_model)

        # 3. Encode (6-layer RoPE transformer) -----------------------------------
        agent_context = self.encoder(agent_tokens)    # (B, N, d_model)

        # 4. Cross-attention fusion ----------------------------------------------
        fused_map  = self.fusion(map_tokens, agent_context, distances)  # (B, M, d_model)

        # 5. Decode --------------------------------------------------------------
        traj_out, confidences = self.decoder(agent_context, fused_map, hcp_mask)

        return traj_out, confidences


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_gmm_loss(pred_trajs, confidences, gt_trajs):
    """
    GMM Negative Log-Likelihood loss.

    pred_trajs  : (B, N, K, T, 5)   x, y at indices 0, 1
    confidences : (B, N, K)
    gt_trajs    : (B, N, T, 5)
    """
    B, N, K, T, _ = pred_trajs.shape

    pred_xy = pred_trajs[..., :2]
    gt_xy   = gt_trajs[..., :2].unsqueeze(2)         # (B, N, 1, T, 2)

    dists = torch.norm(pred_xy - gt_xy, dim=-1)       # (B, N, K, T)
    ade   = dists.mean(dim=-1)                         # (B, N, K)

    best_mode_idx = ade.argmin(dim=-1)                 # (B, N)

    # Regression on best mode (x, y only)
    best_idx_exp  = (best_mode_idx
                     .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                     .expand(-1, -1, -1, T, 2))
    best_pred_xy  = torch.gather(pred_xy, 2, best_idx_exp).squeeze(2)  # (B, N, T, 2)
    reg_loss      = F.smooth_l1_loss(best_pred_xy, gt_trajs[..., :2])

    # Classification (winner-takes-all CE)
    flat_conf   = confidences.view(B * N, K)
    flat_target = best_mode_idx.view(B * N)
    cls_loss    = F.nll_loss(torch.log(flat_conf + 1e-8), flat_target)

    return reg_loss + 2.0 * cls_loss, reg_loss.item(), cls_loss.item()


# ---------------------------------------------------------------------------
# Loss wrapper used by ScaleTrainingManager.execute_step
# ---------------------------------------------------------------------------

def _compute_loss_for_manager(model, batch_data, device):
    """
    Bridges the ScaleTrainingManager API to the model's forward + GMM loss.
    batch_data must contain 'hist_traj', 'gt_traj', 'map_polylines', 'hcp_mask'.
    """
    hist_traj    = batch_data["hist_traj"].to(device)
    gt_traj      = batch_data["gt_traj"].to(device)
    map_polylines = batch_data["map_polylines"]
    hcp_mask     = batch_data["hcp_mask"].to(device) if batch_data.get("hcp_mask") is not None else None

    pred_trajs, confidences = model(hist_traj, map_polylines, hcp_mask)
    return compute_gmm_loss(pred_trajs, confidences, gt_traj)


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

def train_model(
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 1e-4,
    grad_accum: int = 4,
    num_workers: int = 2,
):
    print("Initialising training pipeline …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # ------------------------------------------------------------------ data
    data_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nuscenes_dir = os.path.join(data_dir, "data", "nuscenes")
    waymo_dir    = os.path.join(data_dir, "data", "waymo")

    # Build base dataset router (index-addressable)
    base_dataset = DatasetRouter(nuscenes_dir, waymo_dir, mode="waymo")

    # High-throughput streaming dataloader
    scenario_indices = list(range(len(base_dataset)))
    dataloader = build_streaming_dataloader(
        scenarios_list=scenario_indices,
        parser_instance=base_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    # -------------------------------------------------------------- model
    model  = MTRMotionTransformer(d_model=256, n_modes=6).to(device)
    pruner = HierarchicalCombinatorialPruner().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # AMP + gradient accumulation manager
    scale_mgr = ScaleTrainingManager(
        model=model,
        optimizer=optimizer,
        grad_accumulation_steps=grad_accum,
    )

    history_loss: list = []
    print(f"Starting training … (effective batch = {batch_size} × {grad_accum} = "
          f"{batch_size * grad_accum} scenes)")

    for epoch in range(epochs):
        model.train()
        epoch_loss = epoch_reg = epoch_cls = 0.0
        n_steps    = 0
        start_time = time.time()

        for step_idx, collated in enumerate(dataloader):
            # ---- Unpack collated batch ----------------------------------------
            # collated["history_traj"] : (sum_N, T_hist, 6) — all agents packed
            # collated["batch_splits"] : list[int]          — agents per scene
            # We need (B, N, T_hist, 6) for the model; pad to max N in this batch.
            packed_hist  = collated["history_traj"]   # (sum_N, T_hist, 6)
            packed_fut   = collated["future_traj"]    # (sum_N, T_fut,  5)
            batch_splits = collated["batch_splits"]   # list[int]
            B            = len(batch_splits)
            N_max        = max(batch_splits)
            T_hist       = packed_hist.shape[1]
            T_fut        = packed_fut.shape[1]

            # Pad to (B, N_max, T, C) — padding only within the micro-batch,
            # NOT to a global maximum across the dataset.
            hist_padded = torch.zeros((B, N_max, T_hist, 6),  dtype=torch.float32)
            fut_padded  = torch.zeros((B, N_max, T_fut,  5),  dtype=torch.float32)
            cursor = 0
            for b, n in enumerate(batch_splits):
                hist_padded[b, :n] = packed_hist[cursor:cursor + n]
                fut_padded[b,  :n] = packed_fut[cursor:cursor + n]
                cursor += n

            hist_padded = hist_padded.to(device)
            fut_padded  = fut_padded.to(device)

            # Reconstruct per-scene polyline lists for the model
            map_tensors = collated["map_tensors"]   # flat list of (P_i, 3) tensors
            map_splits  = collated["map_splits"]    # polylines per scene
            map_polylines_batch: list = []
            mc = 0
            for nm in map_splits:
                scene_polys = [t.numpy() for t in map_tensors[mc:mc + nm]]
                map_polylines_batch.append(scene_polys)
                mc += nm

            # ---- HCP pruning mask (dense candidates from GT) -----------------
            dense_candidates = torch.zeros((B, N_max, 6, T_fut, 5), device=device)
            for b in range(B):
                for n in range(batch_splits[b]):
                    gt = fut_padded[b, n]
                    for k in range(6):
                        noise = torch.randn_like(gt) * (k * 0.5)
                        dense_candidates[b, n, k] = gt + noise

            hcp_masks = []
            for b in range(B):
                _, mask, _ = pruner(
                    dense_candidates[b, :batch_splits[b]],
                    hist_padded[b, :batch_splits[b]],
                    map_polylines_batch[b],
                )
                # Pad mask to N_max
                pad_rows = N_max - batch_splits[b]
                if pad_rows > 0:
                    mask = torch.cat(
                        [mask, torch.zeros((pad_rows, 6), dtype=torch.bool, device=device)],
                        dim=0)
                hcp_masks.append(mask)
            hcp_mask = torch.stack(hcp_masks)   # (B, N_max, 6)

            # ---- AMP forward + backward via ScaleTrainingManager --------------
            batch_data = {
                "hist_traj":     hist_padded,
                "gt_traj":       fut_padded,
                "map_polylines": map_polylines_batch,
                "hcp_mask":      hcp_mask,
            }
            step_loss = scale_mgr.execute_step(
                batch_data, _compute_loss_for_manager, step_idx)

            epoch_loss += step_loss
            n_steps    += 1

        # ---------------------------------------------------------------- epoch summary
        avg_loss = epoch_loss / max(n_steps, 1)
        history_loss.append(avg_loss)
        elapsed  = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Steps: {n_steps} | Time: {elapsed:.2f}s")

    # ------------------------------------------------------------------ save outputs
    out_dir = os.path.join(data_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(range(1, epochs + 1), history_loss, marker='o', color='#1D9E75')
    plt.title("HCP + MTR Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    torch.save(model.state_dict(), os.path.join(out_dir, "mtr_checkpoint.pth"))
    print("Training finished.  Checkpoint saved.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HCP + MTR autonomous driving model")
    parser.add_argument("--epochs",      type=int,   default=3,    help="Training epochs")
    parser.add_argument("--batch_size",  type=int,   default=2,    help="Scenes per batch")
    parser.add_argument("--lr",          type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_accum",  type=int,   default=4,
                        help="Gradient accumulation steps (effective batch × this)")
    parser.add_argument("--num_workers", type=int,   default=2,
                        help="DataLoader worker processes (0 = main process)")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum=args.grad_accum,
        num_workers=args.num_workers,
    )
