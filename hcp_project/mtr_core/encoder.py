import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Applies a rotation to query and key tensors in the attention mechanism.
    
    Complexity: O(B * H * L * D)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _rotate_half(self, x):
        # Splitting and rotating pairs
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        # x1, x2 shape: (B, H, L, D/2)
        # Stack and merge: yields [-x_1, x_0, -x_3, x_2, ...]
        out = torch.stack((-x2, x1), dim=-1) # (B, H, L, D/2, 2)
        return out.flatten(-2)

    def forward(self, x, seq_len):
        # x shape: (B, H, L, D)
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(t, self.inv_freq) # (L, D/2)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1) # (L, D)
        
        # Reshape for broadcasting
        cos = freqs.cos().unsqueeze(0).unsqueeze(1) # (1, 1, L, D)
        sin = freqs.sin().unsqueeze(0).unsqueeze(1) # (1, 1, L, D)
        
        # Apply Euler's formula rotation
        x_rot = (x * cos) + (self._rotate_half(x) * sin)
        return x_rot

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rope = RotaryPositionEmbedding(self.head_dim)
        
    def forward(self, x, mask=None):
        B, L, _ = x.shape
        
        # Project and split heads
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2) # (B, H, L, D_head)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2) # (B, H, L, D_head)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2) # (B, H, L, D_head)
        
        # Apply RoPE positional encoding to queries and keys
        q = self.rope(q, L)
        k = self.rope(k, L)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, H, L, L)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v) # (B, H, L, D_head)
        
        # Concatenate heads and project back
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)

class MTRTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # Pre-LN or Post-LN structure. Here we use standard Post-LN
        src2 = self.self_attn(src, mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MTREncoder(nn.Module):
    """
    MTR Encoder.
    6-layer transformer encoder utilizing Rotary Position Embeddings.
    
    Complexity: O(N_layers * (L^2 * d_model + L * d_model^2))
    """
    def __init__(self, d_model=256, nhead=8, num_layers=6, dim_feedforward=512):
        super().__init__()
        self.layers = nn.ModuleList([
            MTRTransformerEncoderLayer(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
    def forward(self, tokens, mask=None):
        """
        Args:
            tokens (Tensor): Shape (B, L, d_model) where L is sequence length.
            mask (Tensor, optional): Attention mask of shape (B, 1, L, L) or (B, L, L).
        Returns:
            context_embeddings (Tensor): Shape (B, L, d_model).
        """
        x = tokens
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return x

if __name__ == "__main__":
    encoder = MTREncoder(d_model=256, nhead=8, num_layers=6)
    dummy_input = torch.randn(2, 50, 256) # Batch size 2, Sequence length 50, d_model 256
    output = encoder(dummy_input)
    print("Encoder output shape:", output.shape) # Should be (2, 50, 256)
