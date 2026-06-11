import unittest
import torch
import numpy as np
from shapely.geometry import Polygon
from hcp_project.hcp.kff import KinematicFeasibilityFilter
from hcp_project.hcp.srf import SpatialReachabilityFilter
from hcp_project.hcp.scf import SocialCompatibilityFilter
from hcp_project.hcp.pruner import HierarchicalCombinatorialPruner

class TestHCPModule(unittest.TestCase):
    def test_kff(self):
        kff = KinematicFeasibilityFilter(kappa_max=0.2, j_max=5.0, a_lat_max=4.0)
        
        # 1 agent, 2 modes, 12 steps
        traj = torch.zeros((1, 2, 12, 5))
        # Mode 0: Straight line (feasible)
        for t in range(12):
            traj[0, 0, t, 0] = t * 2.0
            traj[0, 0, t, 2] = 4.0 # vx
            
        # Mode 1: High frequency wave (violates jerk and lateral acceleration)
        for t in range(12):
            traj[0, 1, t, 0] = t * 2.0
            traj[0, 1, t, 1] = 5.0 * np.sin(t * 2.0)
            
        mask = kff(traj)
        self.assertTrue(mask[0, 0].item())
        self.assertFalse(mask[0, 1].item())
        
    def test_srf(self):
        srf = SpatialReachabilityFilter(collision_threshold=1.0)
        
        # 1 agent, 2 modes, 10 steps
        traj = torch.zeros((1, 2, 10, 5))
        # Mode 0: stays near center (feasible)
        traj[0, 0, :, 0] = torch.linspace(0.0, 5.0, 10)
        # Mode 1: moves to obstacle (pruned)
        traj[0, 1, :, 0] = torch.linspace(0.0, 10.0, 10)
        traj[0, 1, :, 1] = torch.linspace(0.0, 10.0, 10)
        
        # Map curb obstacle at (10, 10)
        map_polylines = [np.array([[10.0, 10.0, 2.0]])]
        
        mask = srf(traj, map_polylines)
        self.assertTrue(mask[0, 0].item())
        self.assertFalse(mask[0, 1].item())
        
    def test_scf(self):
        scf = SocialCompatibilityFilter(collision_threshold=1.0)
        
        # 2 agents, 2 modes, 12 steps
        traj = torch.zeros((2, 2, 12, 5))
        hist = torch.zeros((2, 5, 6))
        
        hist[0, -1, :2] = torch.tensor([0.0, 0.0])
        hist[1, -1, :2] = torch.tensor([4.0, 0.0]) # close to ego
        
        # Ego Mode 0: goes straight and hits agent 1
        # Ego Mode 1: moves sideways and avoids
        for t in range(12):
            traj[0, 0, t, :2] = torch.tensor([t * 1.0, 0.0]) # collision
            traj[0, 1, t, :2] = torch.tensor([0.0, t * 1.0]) # safe
            
            traj[1, 0, t, :2] = torch.tensor([4.0, 0.0]) # stationary obstacle car
            traj[1, 1, t, :2] = torch.tensor([4.0, 0.0])
            
        mask = scf(traj, hist)
        self.assertFalse(mask[0, 0].item())
        self.assertTrue(mask[0, 1].item())
        
    def test_pruner(self):
        pruner = HierarchicalCombinatorialPruner()
        traj = torch.zeros((2, 4, 12, 5))
        hist = torch.zeros((2, 5, 6))
        map_polylines = [np.array([[-10.0, 0.0, 1.0], [10.0, 0.0, 1.0]])]
        
        # SDC is at (0,0), Other is at (2,0)
        hist[0, -1, :2] = torch.tensor([0.0, 0.0])
        hist[1, -1, :2] = torch.tensor([2.0, 0.0])
        
        for k in range(4):
            for t in range(12):
                traj[0, k, t, :2] = torch.tensor([t * 1.0, 0.0])
                traj[1, k, t, :2] = torch.tensor([2.0, t * 0.5])
                
        sparse, mask, stats = pruner(traj, hist, map_polylines)
        self.assertEqual(len(sparse), 2)
        self.assertIn("pruning_ratio", stats)
        self.assertIn("latency_reduction_pct", stats)

if __name__ == "__main__":
    unittest.main()
