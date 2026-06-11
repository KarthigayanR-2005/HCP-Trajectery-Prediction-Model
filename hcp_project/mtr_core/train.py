import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from hcp_project.data.dataset_router import DatasetRouter
from hcp_project.mtr_core.tokenizer import MapTokenizer, AgentTokenizer
from hcp_project.mtr_core.encoder import MTREncoder
from hcp_project.fusion.fusion import CrossAttentionFusionLayer
from hcp_project.mtr_core.decoder import MTRDecoder
from hcp_project.hcp.pruner import HierarchicalCombinatorialPruner

class MTRMotionTransformer(nn.Module):
    def __init__(self, d_model=256, n_modes=6, T_fut=12):
        super().__init__()
        self.map_tokenizer = MapTokenizer(in_channels=3, d_model=d_model)
        self.agent_tokenizer = AgentTokenizer(in_channels=6, d_model=d_model)
        self.encoder = MTREncoder(d_model=d_model, nhead=8, num_layers=4)
        self.fusion = CrossAttentionFusionLayer(d_model=d_model, nhead=8)
        self.decoder = MTRDecoder(d_model=d_model, n_modes=n_modes, T_fut=T_fut)
        
    def forward(self, history_traj, map_polylines, hcp_mask=None):
        """
        Args:
            history_traj (Tensor): Shape (B, N, T_hist, 6)
            map_polylines (list of list of np.ndarray): outer is Batch, inner is polylines list
            hcp_mask (Tensor): Shape (B, N, n_modes)
        """
        B, N, T_hist, _ = history_traj.shape
        device = history_traj.device
        
        # 1. Tokenize agent history
        # Flatten B and N
        flat_hist = history_traj.view(B * N, T_hist, -1)
        agent_tokens = self.agent_tokenizer(flat_hist) # (B * N, d_model)
        agent_tokens = agent_tokens.view(B, N, -1) # (B, N, d_model)
        
        # 2. Tokenize map polylines
        # Since map_polylines is a list of arrays per batch, we pad or process sequentially
        # For simplicity, we stack map polylines to a fixed size of 20 polylines per scenario
        max_polys = 20
        padded_map = torch.zeros((B, max_polys, 10, 3), device=device) # (B, M, P, C_in)
        distances = torch.zeros((B, max_polys, N), device=device) # (B, M, N)
        
        for b in range(B):
            polys = map_polylines[b]
            for m in range(min(len(polys), max_polys)):
                poly = polys[m]
                # Pad/slice poly points to 10 points
                pts = poly[:, :3]
                if len(pts) > 10:
                    pts = pts[:10]
                elif len(pts) < 10:
                    pts = np.pad(pts, ((0, 10 - len(pts)), (0, 0)), mode='edge')
                padded_map[b, m] = torch.tensor(pts, dtype=torch.float32, device=device)
                
                # Compute distance from map polyline center to agents at t=0
                poly_center = pts[:, :2].mean(axis=0)
                for n in range(N):
                    agent_pos = history_traj[b, n, -1, :2].detach().cpu().numpy()
                    distances[b, m, n] = float(np.linalg.norm(poly_center - agent_pos))
                    
        # Tokenize map
        flat_map = padded_map.view(B * max_polys, 10, -1)
        map_tokens = self.map_tokenizer(flat_map) # (B * M, d_model)
        map_tokens = map_tokens.view(B, max_polys, -1) # (B, M, d_model)
        
        # 3. MTR Encoder (RoPE positional encoding)
        agent_context = self.encoder(agent_tokens) # (B, N, d_model)
        
        # 4. Cross-Attention Fusion (novel component)
        fused_map_tokens = self.fusion(map_tokens, agent_context, distances) # (B, M, d_model)
        
        # 5. Decoder (with Intention point query and GMM)
        traj_out, confidences = self.decoder(agent_context, fused_map_tokens, hcp_mask)
        
        return traj_out, confidences

def compute_gmm_loss(pred_trajs, confidences, gt_trajs):
    """
    Computes GMM Negative Log-Likelihood loss.
    pred_trajs shape: (B, N, K, T, 5) -> x, y is index 0, 1
    confidences shape: (B, N, K)
    gt_trajs shape: (B, N, T, 5) -> x, y is index 0, 1
    """
    B, N, K, T, _ = pred_trajs.shape
    device = pred_trajs.device
    
    # 1. Find the best mode (minimum ADE) per agent
    # pred_xy: (B, N, K, T, 2)
    # gt_xy: (B, N, 1, T, 2)
    pred_xy = pred_trajs[..., :2]
    gt_xy = gt_trajs[..., :2].unsqueeze(2)
    
    # Compute ADE for each mode
    # dist shape: (B, N, K, T)
    dists = torch.norm(pred_xy - gt_xy, dim=-1)
    ade = torch.mean(dists, dim=-1) # (B, N, K)
    
    # Get index of best mode
    best_mode_idx = torch.argmin(ade, dim=-1) # (B, N)
    
    # 2. Regression Loss: Smooth L1 on the best mode
    # Gather best mode trajectories
    best_mode_idx_exp = best_mode_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, T, 2)
    best_pred_xy = torch.gather(pred_xy, 2, best_mode_idx_exp).squeeze(2) # (B, N, T, 2)
    
    reg_loss = F.smooth_l1_loss(best_pred_xy, gt_trajs[..., :2], reduction='mean')
    
    # 3. Classification Loss: Cross entropy between confidences and best mode
    # Flatten B and N
    flat_conf = confidences.view(B * N, K)
    flat_target = best_mode_idx.view(B * N)
    # To prevent log(0) issues in cross entropy, we add epsilon or use standard CE on logits
    # Confidences are already softmax outputs, so we compute NLL Loss on log(confidences)
    cls_loss = F.nll_loss(torch.log(flat_conf + 1e-8), flat_target, reduction='mean')
    
    total_loss = reg_loss + 2.0 * cls_loss
    return total_loss, reg_loss.item(), cls_loss.item()

def train_model(epochs=5, batch_size=2, lr=1e-4):
    print("Initializing training pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # Setup data paths
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nuscenes_dir = os.path.join(data_dir, "data", "nuscenes")
    waymo_dir = os.path.join(data_dir, "data", "waymo")
    
    # Initialize unified dataset router
    dataset = DatasetRouter(nuscenes_dir, waymo_dir, mode="waymo")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    
    # Initialize modules
    model = MTRMotionTransformer(d_model=256, n_modes=6).to(device)
    pruner = HierarchicalCombinatorialPruner().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Logs
    history_loss = []
    
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reg = 0.0
        epoch_cls = 0.0
        start_time = time.time()
        
        for i, batch_list in enumerate(dataloader):
            # Format inputs
            # B = len(batch_list)
            # Stack histories and futures
            hist_traj = torch.tensor(np.stack([b.history_traj for b in batch_list]), dtype=torch.float32, device=device) # (B, N, T_hist, 6)
            gt_traj = torch.tensor(np.stack([b.future_traj for b in batch_list]), dtype=torch.float32, device=device) # (B, N, T_fut, 5)
            
            # Map polylines list
            map_polylines = [b.map_polylines for b in batch_list]
            
            # 1. Run HCP to get pruning mask
            # For training, we generate dense candidates using a dummy predictor or perturbed GT trajectories
            # Dense candidates shape: (B, N, K=6, T_fut=12, 5)
            B, N, T_hist, _ = hist_traj.shape
            dense_candidates = torch.zeros((B, N, 6, 12, 5), device=device)
            # Populate candidates: perturb ground truth or extrapolate history
            for b in range(B):
                for n in range(N):
                    gt = gt_traj[b, n] # (T_fut, 5)
                    # Let's generate 6 candidate modes around ground truth with noise
                    for k in range(6):
                        noise = torch.randn_like(gt) * (k * 0.5) # noise depends on mode
                        dense_candidates[b, n, k] = gt + noise
                        
            # Run pruner on dense candidates to compute mask
            hcp_masks = []
            for b in range(B):
                # We slice per scenario
                _, mask, _ = pruner(dense_candidates[b], hist_traj[b], map_polylines[b])
                hcp_masks.append(mask)
            hcp_mask = torch.stack(hcp_masks) # (B, N, 6)
            
            # 2. Run MTR model
            optimizer.zero_grad()
            pred_trajs, confidences = model(hist_traj, map_polylines, hcp_mask)
            
            # 3. Compute loss
            loss, r_loss, c_loss = compute_gmm_loss(pred_trajs, confidences, gt_traj)
            
            # 4. Step optimizer
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_reg += r_loss
            epoch_cls += c_loss
            
        avg_loss = epoch_loss / len(dataloader)
        avg_reg = epoch_reg / len(dataloader)
        avg_cls = epoch_cls / len(dataloader)
        
        history_loss.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (Reg: {avg_reg:.4f}, Cls: {avg_cls:.4f}) | Time: {time.time() - start_time:.2f}s")
        
    # Save training logs plot
    os.makedirs(os.path.join(data_dir, "outputs"), exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs + 1), history_loss, marker='o', color='#1D9E75')
    plt.title("HCP + MTR Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(data_dir, "outputs", "loss_curve.png"))
    plt.close()
    
    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(data_dir, "outputs", "mtr_checkpoint.pth"))
    print("Training finished. Checkpoint saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size)
