import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point, Polygon

class SpatialReachabilityFilter(nn.Module):
    """
    Spatial Reachability Filter (SRF).
    Eliminates candidate trajectories leaving the drivable area or colliding with static obstacles.
    
    Uses:
      - KD-Tree spatial index for O(log M) obstacle collision queries.
      - Boundary containment check.
      
    Complexity: O(N * K * T * log M) where:
      - N is number of agents
      - K is number of candidate modes
      - T is number of timesteps
      - M is number of static map/obstacle points
    """
    def __init__(self, collision_threshold=1.0):
        super().__init__()
        self.collision_threshold = collision_threshold
        
    def forward(self, trajectories, map_polylines, drivable_area=None):
        """
        Args:
            trajectories (Tensor): Shape (N, K, T, 5) representing [x, y, vx, vy, heading]
            map_polylines (list of np.ndarray): Polylines representing roads/lanes/curbs.
            drivable_area (shapely.geometry.Polygon): Drivable zone boundary.
        Returns:
            mask (Tensor): Bool tensor of shape (N, K). True means feasible, False means pruned.
        """
        N, K, T, _ = trajectories.shape
        device = trajectories.device
        
        # 1. Build KD-tree of static map/obstacle points (lane boundaries, crosswalk edges)
        obstacle_pts = []
        for polyline in map_polylines:
            # Type 1 = lane center, Type 2 = crosswalk
            # Let's treat lane boundaries/curbs or crosswalk boundaries as obstacles if trajectory gets too close to non-drivable bounds
            # For this filter, we extract coordinates
            for pt in polyline:
                # If polyline is not lane centerline (i.e. is crosswalk boundary or road curb), add to obstacles
                if pt[2] != 1.0: 
                    obstacle_pts.append(pt[:2])
                    
        # Fallback if no obstacle points
        if len(obstacle_pts) == 0:
            # Put a dummy far-away point
            obstacle_pts.append([9999.0, 9999.0])
            
        kdtree = KDTree(np.array(obstacle_pts))
        
        # Move trajectories to CPU for KD-Tree query
        traj_cpu = trajectories.detach().cpu().numpy() # (N, K, T, 5)
        
        mask = np.ones((N, K), dtype=bool)
        
        for agent_idx in range(N):
            for mode_idx in range(K):
                mode_traj = traj_cpu[agent_idx, mode_idx] # (T, 5)
                
                # Check collision with static obstacles
                # Query nearest distances for all T points along trajectory
                pts = mode_traj[:, :2] # (T, 2)
                dists, _ = kdtree.query(pts) # O(T * log M)
                
                # If any point is closer than threshold to static obstacles, prune
                if np.any(dists < self.collision_threshold):
                    mask[agent_idx, mode_idx] = False
                    continue
                    
                # Check if points stay within drivable area
                if drivable_area is not None:
                    # Check first, middle, and last points to optimize speed
                    for step in [0, T//2, T-1]:
                        pt_shapely = Point(pts[step, 0], pts[step, 1])
                        if not drivable_area.contains(pt_shapely):
                            mask[agent_idx, mode_idx] = False
                            break
                            
        return torch.tensor(mask, dtype=torch.bool, device=device)

if __name__ == "__main__":
    srf = SpatialReachabilityFilter(collision_threshold=1.5)
    
    # 2 agents, 2 modes, 10 timesteps
    traj = torch.zeros((2, 2, 10, 5))
    # Agent 0, Mode 0: stays near center (feasible)
    # Agent 0, Mode 1: moves to (20.0, 20.0) which is near crosswalk obstacle
    traj[0, 0, :, 0] = torch.linspace(0.0, 10.0, 10)
    traj[0, 1, :, 0] = torch.linspace(0.0, 20.0, 10)
    traj[0, 1, :, 1] = torch.linspace(0.0, 20.0, 10)
    
    # Map polylines
    # An obstacle at (20.0, 21.0) with type 2.0 (crosswalk boundary)
    map_polylines = [
        np.array([[20.0, 21.0, 2.0], [21.0, 21.0, 2.0]])
    ]
    
    # Drivable area
    drivable_area = Polygon([(-30, -30), (30, -30), (30, 30), (-30, 30)])
    
    mask = srf(traj, map_polylines, drivable_area)
    print("Feasibility mask:")
    print("Mode 0 (center):", mask[0, 0].item())
    print("Mode 1 (hits obstacle):", mask[0, 1].item())
