import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageTokenizer(nn.Module):
    """
    Image Tokenizer (ViT-S/16 style).
    Converts visual frames of shape (B, C, H, W) into sequence of patch tokens.
    
    Complexity: O(B * H/16 * W/16 * PatchSize^2 * d_model)
    """
    def __init__(self, in_channels=3, patch_size=16, d_model=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.proj(x) # (B, d_model, H/16, W/16)
        x = x.flatten(2).transpose(1, 2) # (B, NumPatches, d_model)
        return x

class MapTokenizer(nn.Module):
    """
    Map Polyline Tokenizer (PointNet style).
    Converts map polylines of shape (M_polylines, P_points, C_in) into polyline tokens.
    
    Complexity: O(M * P * C_in * d_model)
    """
    def __init__(self, in_channels=3, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
    def forward(self, polylines):
        """
        Args:
            polylines (list of Tensors or Tensor): Shape (M, P, C_in) where:
                M is number of polylines, P is points per polyline, C_in is features.
        Returns:
            polyline_tokens (Tensor): Shape (M, d_model)
        """
        # Element-wise MLP
        features = self.mlp(polylines) # (M, P, d_model)
        # PointNet max pooling over point dimension P
        tokens = torch.max(features, dim=1)[0] # (M, d_model)
        return tokens

class AgentTokenizer(nn.Module):
    """
    Agent Track History Tokenizer.
    Converts agent track history of shape (N, T_hist, C_in) into agent state tokens.
    
    Complexity: O(N * T_hist * C_in * d_model)
    """
    def __init__(self, in_channels=6, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
    def forward(self, track_history):
        """
        Args:
            track_history (Tensor): Shape (N, T_hist, C_in)
        Returns:
            agent_tokens (Tensor): Shape (N, d_model)
        """
        features = self.mlp(track_history) # (N, T_hist, d_model)
        # Max pool over history timesteps T_hist
        tokens = torch.max(features, dim=1)[0] # (N, d_model)
        return tokens

if __name__ == "__main__":
    # Test tokenizer modules
    img_tok = ImageTokenizer(patch_size=16, d_model=256)
    map_tok = MapTokenizer(in_channels=3, d_model=256)
    agent_tok = AgentTokenizer(in_channels=6, d_model=256)
    
    img = torch.randn(2, 3, 224, 224)
    polys = torch.randn(10, 20, 3)
    history = torch.randn(5, 5, 6)
    
    print("Tokens dimensions:")
    print("Image tokenized:", img_tok(img).shape) # Should be (2, 196, 256)
    print("Map tokenized:", map_tok(polys).shape) # Should be (10, 256)
    print("Agent tokenized:", agent_tok(history).shape) # Should be (5, 256)
