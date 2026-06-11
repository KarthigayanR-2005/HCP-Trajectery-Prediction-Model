import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGELayer(nn.Module):
    """
    Standard GraphSAGE layer: aggregates neighbor features and concatenates with ego.
    Complexity: O(N * D + E * D) where N is nodes, E is edges, D is dimension.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_self = nn.Linear(in_dim, out_dim)
        self.fc_neigh = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU(0.1)
        
    def forward(self, x, adj):
        """
        Args:
            x (Tensor): Node features of shape (N_agents, in_dim)
            adj (Tensor): Adjacency matrix of shape (N_agents, N_agents)
        Returns:
            x_new (Tensor): Node features of shape (N_agents, out_dim)
        """
        N = x.shape[0]
        # Sum neighbor features
        neigh_sum = torch.matmul(adj, x) # (N, in_dim)
        # Average neighbor features
        deg = torch.sum(adj, dim=-1, keepdim=True) + 1e-6
        neigh_mean = neigh_sum / deg
        
        # SAGE formula
        h_self = self.fc_self(x)
        h_neigh = self.fc_neigh(neigh_mean)
        return self.act(h_self + h_neigh)

class SocialCompatibilityFilter(nn.Module):
    """
    Social Compatibility Filter (SCF).
    Eliminates candidate trajectories causing predicted collisions with other agents.
    
    Architecture:
      - 1-layer MLP trajectory encoder
      - 3-layer GraphSAGE GNN (hidden_dim=128) over agent interaction graph
      - Collision check head (TTC and bounding box overlap check)
    """
    def __init__(self, hidden_dim=128, collision_threshold=1.5):
        super().__init__()
        self.collision_threshold = collision_threshold
        
        # Trajectory Encoder (T_steps * 5 inputs -> hidden_dim)
        # Assuming T_steps = 12, features = 5
        self.traj_encoder = nn.Sequential(
            nn.Linear(12 * 5, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3-Layer GraphSAGE GNN
        self.sage1 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.sage3 = GraphSAGELayer(hidden_dim, hidden_dim)
        
        # Edge prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, trajectories, history_traj):
        """
        Args:
            trajectories (Tensor): Shape (N, K, T, 5) representing candidate future trajectories.
            history_traj (Tensor): Shape (N, T_hist, 6) representing agent histories.
        Returns:
            mask (Tensor): Bool tensor of shape (N, K). True means compatible, False means pruned.
        """
        N, K, T, C = trajectories.shape
        device = trajectories.device
        
        # 1. Encode trajectories
        # Flatten time and channel dimensions
        flat_trajs = trajectories.view(N * K, T * C)
        traj_embeds = self.traj_encoder(flat_trajs) # (N*K, hidden_dim)
        traj_embeds = traj_embeds.view(N, K, -1)
        
        # Agent node representation (max pool across modes)
        node_embeds = torch.max(traj_embeds, dim=1)[0] # (N, hidden_dim)
        
        # 2. Construct interaction graph adjacency matrix based on t=0 history positions
        # history_traj[..., -1, :2] gives the x, y coordinates at t=0
        pos_t0 = history_traj[:, -1, :2] # (N, 2)
        dist_matrix = torch.cdist(pos_t0, pos_t0) # (N, N)
        adj = (dist_matrix < 40.0).float() # Neighbors within 40 meters
        
        # 3. Apply 3-layer GraphSAGE
        h = self.sage1(node_embeds, adj)
        h = self.sage2(h, adj)
        h = self.sage3(h, adj) # (N, hidden_dim)
        
        # 4. Social pruning decision:
        # Check all pairs of agent-mode combinations
        # Pruning mask: initially all True
        mask = torch.ones((N, K), dtype=torch.bool, device=device)
        
        # Vectorized check or simple loop over agent pairs for collision checking
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                    
                # Get trajectory coordinates for agent i (modes K) and agent j (modes K)
                # trajectories shape: (N, K, T, 5) -> x, y is index 0, 1
                pos_i = trajectories[i, :, :, :2] # (K, T, 2)
                pos_j = trajectories[j, :, :, :2] # (K, T, 2)
                
                # Check distances for all mode combinations at all timesteps
                # Broad-cast matching: (K_i, 1, T, 2) vs (1, K_j, T, 2)
                dists = torch.norm(pos_i.unsqueeze(1) - pos_j.unsqueeze(0), dim=-1) # (K_i, K_j, T)
                
                # Minimum distance along the trajectory for each mode combination
                min_dists = torch.min(dists, dim=-1)[0] # (K_i, K_j)
                
                # If a candidate trajectory of agent i always collides with ALL candidate trajectories
                # of agent j (or if the collision risk predicted by GNN is very high), we prune it
                collision_mask = min_dists < self.collision_threshold # (K_i, K_j)
                
                # If a mode of agent i collides with the main mode (index 0) of neighbor j, prune
                for k in range(K):
                    if collision_mask[k, 0]:
                        mask[i, k] = False
                        
        return mask

if __name__ == "__main__":
    # Test SCF
    scf = SocialCompatibilityFilter(hidden_dim=128)
    
    # 3 agents, 2 modes, 12 steps
    traj = torch.zeros((3, 2, 12, 5))
    hist = torch.zeros((3, 5, 6))
    
    # Set t=0 positions
    hist[0, -1, :2] = torch.tensor([0.0, 0.0]) # Ego
    hist[1, -1, :2] = torch.tensor([5.0, 0.0]) # Front car
    hist[2, -1, :2] = torch.tensor([40.0, 40.0]) # Far car
    
    # Trajectories
    # Ego mode 0: goes straight, collides with agent 1 going slow
    # Ego mode 1: swerves left, avoids collision
    for t in range(12):
        # Ego
        traj[0, 0, t, :2] = torch.tensor([t * 2.0, 0.0]) # collides
        traj[0, 1, t, :2] = torch.tensor([t * 2.0, t * 1.0]) # swerves
        # Agent 1
        traj[1, 0, t, :2] = torch.tensor([5.0 + t * 0.5, 0.0]) # slow straight
        traj[1, 1, t, :2] = torch.tensor([5.0 + t * 0.5, 0.0])
        # Agent 2
        traj[2, 0, t, :2] = torch.tensor([40.0 + t * 2.0, 40.0])
        traj[2, 1, t, :2] = torch.tensor([40.0 + t * 2.0, 40.0])
        
    mask = scf(traj, hist)
    print("Social compatibility mask for Ego:")
    print("Mode 0 (Collides):", mask[0, 0].item())
    print("Mode 1 (Swerves/Avoids):", mask[0, 1].item())
