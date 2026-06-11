import torch
import torch.nn as nn

class KinematicFeasibilityFilter(nn.Module):
    """
    Kinematic Feasibility Filter (KFF).
    Eliminates candidate trajectories violating physical limits of vehicle dynamics.
    
    Constraints:
      - Max Curvature kappa_max = 0.2 rad/m
      - Max Jerk j_max = 5.0 m/s^3
      - Max Lateral Acc a_lat_max = 4.0 m/s^2
      
    Complexity: O(N * K * T) where:
      - N is number of agents
      - K is number of candidate modes
      - T is number of timesteps
    """
    def __init__(self, kappa_max=0.2, j_max=5.0, a_lat_max=4.0, dt=0.5):
        super().__init__()
        self.kappa_max = kappa_max
        self.j_max = j_max
        self.a_lat_max = a_lat_max
        self.dt = dt
        self.eps = 1e-6
        
    def forward(self, trajectories):
        """
        Args:
            trajectories (Tensor): Shape (N, K, T, 5) representing [x, y, vx, vy, heading]
        Returns:
            mask (Tensor): Bool tensor of shape (N, K). True means feasible, False means pruned.
        """
        N, K, T, C = trajectories.shape
        if T < 4:
            # Not enough timesteps to calculate jerk; return all True
            return torch.ones((N, K), dtype=torch.bool, device=trajectories.device)
            
        x = trajectories[..., 0]
        y = trajectories[..., 1]
        
        # 1st order differences: velocity
        vx = (x[..., 1:] - x[..., :-1]) / self.dt
        vy = (y[..., 1:] - y[..., :-1]) / self.dt
        v = torch.sqrt(vx**2 + vy**2 + self.eps)
        
        # 2nd order differences: acceleration
        ax = (vx[..., 1:] - vx[..., :-1]) / self.dt
        ay = (vy[..., 1:] - vy[..., :-1]) / self.dt
        a = torch.sqrt(ax**2 + ay**2 + self.eps)
        
        # 3rd order differences: jerk
        jx = (ax[..., 1:] - ax[..., :-1]) / self.dt
        jy = (ay[..., 1:] - ay[..., :-1]) / self.dt
        j = torch.sqrt(jx**2 + jy**2 + self.eps)
        
        # Curvature: kappa = |vx * ay - vy * ax| / (v^3)
        # Note: shapes of vx/vy are (N, K, T-1), shapes of ax/ay are (N, K, T-2).
        # We slice velocity to match acceleration shape (N, K, T-2).
        vx_sliced = vx[..., :-1]
        vy_sliced = vy[..., :-1]
        v_sliced = v[..., :-1]
        
        kappa = torch.abs(vx_sliced * ay - vy_sliced * ax) / (v_sliced**3 + self.eps)
        
        # Lateral Acceleration: a_lat = v^2 * kappa = |vx * ay - vy * ax| / v
        a_lat = torch.abs(vx_sliced * ay - vy_sliced * ax) / (v_sliced + self.eps)
        
        # Check constraints
        # Max Curvature
        kappa_violated = torch.any(kappa > self.kappa_max, dim=-1) # (N, K)
        
        # Max Jerk
        jerk_violated = torch.any(j > self.j_max, dim=-1) # (N, K)
        
        # Max Lateral Acc
        a_lat_violated = torch.any(a_lat > self.a_lat_max, dim=-1) # (N, K)
        
        # Trajectory is feasible if no constraints are violated
        feasible = ~(kappa_violated | jerk_violated | a_lat_violated)
        
        return feasible

if __name__ == "__main__":
    # Unit testing code
    kff = KinematicFeasibilityFilter()
    # Create 2 agents, 3 candidates, 12 steps
    # Candidate 0: Straight line (feasible)
    # Candidate 1: Sharp turn (infeasible curvature)
    # Candidate 2: High acceleration/jerk (infeasible)
    traj = torch.zeros((2, 3, 12, 5))
    
    # Fill candidate 0: straight line at 10m/s
    for t in range(12):
        traj[:, 0, t, 0] = t * 5.0 # x
        traj[:, 0, t, 2] = 10.0 # vx
        
    # Fill candidate 1: sharp sinus curve (violates curvature)
    for t in range(12):
        traj[:, 1, t, 0] = t * 5.0 # x
        traj[:, 1, t, 1] = 10.0 * torch.sin(torch.tensor(t * 1.5)) # y
        
    # Fill candidate 2: exponential acceleration (violates jerk)
    for t in range(12):
        traj[:, 2, t, 0] = float(t)**3.5 # x
        
    mask = kff(traj)
    print("Feasibility mask per candidate:")
    print("Candidate 0 (Straight):", mask[0, 0].item())
    print("Candidate 1 (Sinus):", mask[0, 1].item())
    print("Candidate 2 (High Jerk):", mask[0, 2].item())
