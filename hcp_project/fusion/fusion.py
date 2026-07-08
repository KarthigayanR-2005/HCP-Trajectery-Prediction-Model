import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusionLayer(nn.Module):
    """
    Cross-Attention Fusion Layer.
    Allows map tokens to attend to agent tokens with a geometry-conditioned attention bias.
    
    Attention Formula:
      Score_ij = (Q_i * K_j^T) / sqrt(d_head) + MLP(RBF(dist_ij))
      
    Complexity: O(B * H * M * N * d_head) where:
      - B is Batch size
      - H is Number of heads
      - M is Number of map tokens
      - N is Number of agent tokens
      - d_head is Dimension per head
    """
    def __init__(self, d_model=256, nhead=8, rbf_kernels=16):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Radial Basis Function (RBF) kernels for geometry distances
        # We will distribute kernel centers between 0m and 100m
        self.register_buffer("rbf_centers", torch.linspace(0.0, 100.0, rbf_kernels))
        self.register_buffer("rbf_sigma", torch.tensor([5.0]))
        
        # MLP to map RBF features to attention bias per head
        self.bias_mlp = nn.Sequential(
            nn.Linear(rbf_kernels, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, nhead)
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, map_tokens, agent_tokens, distances):
        """
        Args:
            map_tokens (Tensor): Shape (B, M, d_model)
            agent_tokens (Tensor): Shape (B, N, d_model)
            distances (Tensor): Pairwise distances of shape (B, M, N)
        Returns:
            fused_map_tokens (Tensor): Fused tokens of shape (B, M, d_model)
        """
        B, M, _ = map_tokens.shape
        _, N, _ = agent_tokens.shape
        
        # 1. Compute geometry-conditioned attention bias
        # Distance shape: (B, M, N). Broadcast RBF kernels.
        # dists_expanded: (B, M, N, 1), centers: (K)
        dists_expanded = distances.unsqueeze(-1)
        rbf_features = torch.exp(-((dists_expanded - self.rbf_centers) ** 2) / (2 * (self.rbf_sigma ** 2))) # (B, M, N, K)
        
        # Map to attention bias of shape (B, M, N, H) -> transpose to (B, H, M, N)
        attn_bias = self.bias_mlp(rbf_features) # (B, M, N, H)
        attn_bias = attn_bias.permute(0, 3, 1, 2) # (B, H, M, N)
        
        # 2. Project queries, keys, values
        q = self.q_proj(map_tokens).view(B, M, self.nhead, self.head_dim).transpose(1, 2) # (B, H, M, d_head)
        k = self.k_proj(agent_tokens).view(B, N, self.nhead, self.head_dim).transpose(1, 2) # (B, H, N, d_head)
        v = self.v_proj(agent_tokens).view(B, N, self.nhead, self.head_dim).transpose(1, 2) # (B, H, N, d_head)
        
        # 3. Scaled dot-product attention + geometry bias
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, H, M, N)
        scores = scores + attn_bias # Add geometry attention bias
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v) # (B, H, M, d_head)
        
        # 4. Concatenate heads and project back
        out = out.transpose(1, 2).contiguous().view(B, M, self.d_model)
        out = self.out_proj(out)
        
        # Residual connection and normalization
        return self.norm(map_tokens + out)

if __name__ == "__main__":
    fusion = CrossAttentionFusionLayer(d_model=256, nhead=8)
    
    # 2 scenes, 30 map polylines, 5 agents
    map_toks = torch.randn(2, 30, 256)
    agent_toks = torch.randn(2, 5, 256)
    dists = torch.rand(2, 30, 5) * 80.0 # distances up to 80m
    
    out = fusion(map_toks, agent_toks, dists)
    print("Cross-Attention Fusion Output shape:", out.shape) # Should be (2, 30, 256)
