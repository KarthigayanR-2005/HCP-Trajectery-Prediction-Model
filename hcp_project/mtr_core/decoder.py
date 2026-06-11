import torch
import torch.nn as nn
import torch.nn.functional as F

class MTRDecoder(nn.Module):
    """
    MTR Decoder.
    Uses intention point query anchors and a 3-layer MLP trajectory refinement head.
    Optionally masks out queries using HCP-pruned candidate masks.
    
    Complexity: O(N * K * d_model + N * K * T_fut * 5) where:
      - N is number of agents
      - K is number of modes (intention points)
      - T_fut is future steps (12)
    """
    def __init__(self, d_model=256, n_modes=6, T_fut=12):
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.T_fut = T_fut
        
        # Learned intention point anchors (x, y coordinates at t=T_fut)
        # We initialize them as 6 anchors: straight-fast, straight-slow, left, right, hard-left, stop
        anchors = torch.tensor([
            [40.0, 0.0],   # Straight fast
            [15.0, 0.0],   # Straight slow
            [25.0, 15.0],  # Left turn
            [25.0, -15.0], # Right turn
            [5.0, 15.0],   # Hard left
            [1.0, 0.0]     # Stop/slow
        ])
        self.register_buffer("intention_anchors", anchors)
        
        # Intention point projection to query embeddings
        self.query_proj = nn.Linear(2, d_model)
        
        # Decoder layer: Cross-attends queries to map/agent contexts
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        
        # Trajectory Refinement Head: 3-layer MLP
        # Output size: T_fut * 5 (x, y, vx, vy, heading) + 1 (probability/confidence)
        self.refinement_head = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, T_fut * 5 + 1)
        )
        
    def forward(self, agent_embeds, map_embeds, hcp_mask=None):
        """
        Args:
            agent_embeds (Tensor): Shape (B, N, d_model) agent history embeddings
            map_embeds (Tensor): Shape (B, M, d_model) fused map/agent embeddings
            hcp_mask (Tensor, optional): Shape (B, N, n_modes) pruning mask from HCP.
                                         True means keep, False means prune.
        Returns:
            trajectories (Tensor): Shape (B, N, n_modes, T_fut, 5) predicted future trajectories
            confidences (Tensor): Shape (B, N, n_modes) class probabilities per mode
        """
        B, N, _ = agent_embeds.shape
        M = map_embeds.shape[1]
        
        # 1. Project intention point anchors to queries
        # anchors: (n_modes, 2) -> (n_modes, d_model)
        query_embeds = self.query_proj(self.intention_anchors) # (n_modes, d_model)
        
        # Expand queries for all agents in the batch
        # queries: (B * N, n_modes, d_model)
        queries = query_embeds.unsqueeze(0).expand(B * N, -1, -1)
        
        # Reshape contexts to (B * N, Seq, d_model) or use standard cross-attention
        # Let's repeat agent context per mode, or cross-attend queries to map/agent tokens
        # Context tokens: we concatenate map embeds and agent embeds
        context = torch.cat([map_embeds, agent_embeds], dim=1) # (B, M + N, d_model)
        context_expanded = context.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, M + N, -1)
        
        # Query attends to context
        # query_attn: (B * N, n_modes, d_model)
        query_attn, _ = self.cross_attn(queries, context_expanded, context_expanded)
        
        # Combine query attention with agent history context
        # agent_embeds shape: (B, N, d_model) -> expand to (B * N, n_modes, d_model)
        agent_embeds_expanded = agent_embeds.unsqueeze(2).expand(-1, -1, self.n_modes, -1).reshape(B * N, self.n_modes, -1)
        
        combined_features = torch.cat([query_attn, agent_embeds_expanded], dim=-1) # (B * N, n_modes, 2 * d_model)
        
        # 5. Output predictions
        outputs = self.refinement_head(combined_features) # (B * N, n_modes, T_fut * 5 + 1)
        outputs = outputs.view(B, N, self.n_modes, -1)
        
        # Extract coordinates and confidences
        traj_out = outputs[..., :self.T_fut * 5].view(B, N, self.n_modes, self.T_fut, 5)
        conf_logits = outputs[..., -1] # (B, N, self.n_modes)
        
        # Add intention anchor offsets to trajectories (to bias predictions correctly based on anchors)
        # anchors: (n_modes, 2) -> broadcasted to trajectories
        anchor_offsets = self.intention_anchors.view(1, 1, self.n_modes, 1, 2)
        # Interpolate anchors across timesteps for trajectory translation bias
        # For simplicity, we add the anchor offset multiplied by (t / T_fut) to the predicted (x, y) coordinates
        t_factor = torch.linspace(0.0, 1.0, self.T_fut, device=agent_embeds.device).view(1, 1, 1, self.T_fut, 1)
        traj_out[..., :2] = traj_out[..., :2] + anchor_offsets * t_factor
        
        # Apply HCP Pruning Mask by setting pruned modes to very low confidence logits
        if hcp_mask is not None:
            # hcp_mask shape: (B, N, n_modes) where True = valid, False = pruned
            # Use logit masking: set False locations to -1e9
            conf_logits = conf_logits.masked_fill(~hcp_mask, -1e9)
            
        confidences = F.softmax(conf_logits, dim=-1) # (B, N, n_modes)
        
        return traj_out, confidences

if __name__ == "__main__":
    decoder = MTRDecoder(d_model=256, n_modes=6)
    
    agent_emb = torch.randn(2, 5, 256) # 2 scenes, 5 agents
    map_emb = torch.randn(2, 30, 256)   # 30 map features
    
    # 2 scenes, 5 agents, 6 modes pruning mask
    # Let's prune modes 4 and 5 for agent 0 in scene 0
    hcp_m = torch.ones((2, 5, 6), dtype=torch.bool)
    hcp_m[0, 0, 4:] = False
    
    trajs, confs = decoder(agent_emb, map_emb, hcp_mask=hcp_m)
    print("Decoder output:")
    print("Trajectories shape:", trajs.shape) # Should be (2, 5, 6, 12, 5)
    print("Confidences shape:", confs.shape)   # Should be (2, 5, 6)
    print("Confidences for scene 0, agent 0 (pruned modes):", confs[0, 0].detach().numpy())
