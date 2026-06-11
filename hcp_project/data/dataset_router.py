import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from hcp_project.data.nuscenes_parser import NuScenesParser, NuScenesMapWrapper
from hcp_project.data.womd_parser import WOMDParser

def transform_to_ego(x, y, vx, vy, heading, ego_x, ego_y, ego_heading):
    """
    Transforms coordinates and velocities to ego-centric frame (x-forward, y-left).
    Complexity: O(1) operations.
    """
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)
    
    # Translation
    dx = x - ego_x
    dy = y - ego_y
    
    # Rotation (to face x-forward)
    x_new = dx * cos_h + dy * sin_h
    y_new = -dx * sin_h + dy * cos_h
    
    # Velocity rotation
    vx_new = vx * cos_h + vy * sin_h
    vy_new = -vx * sin_h + vy * cos_h
    
    # Heading difference
    h_new = heading - ego_heading
    # Normalize heading to [-pi, pi]
    h_new = (h_new + np.pi) % (2 * np.pi) - np.pi
    
    return x_new, y_new, vx_new, vy_new, h_new

class UnifiedBatch:
    def __init__(self, history_traj, future_traj, map_polylines, agent_types, sdc_route_graph, scenario_id):
        self.history_traj = history_traj   # Shape: (N_agents, T_hist=5, 6) [x, y, vx, vy, heading, type_idx]
        self.future_traj = future_traj     # Shape: (N_agents, T_fut=12, 5) [x, y, vx, vy, heading]
        self.map_polylines = map_polylines # List of arrays (M_polylines, P_points, 3) [x, y, type]
        self.agent_types = agent_types     # List of strings corresponding to N_agents
        self.sdc_route_graph = sdc_route_graph # NetworkX graph for routing
        self.scenario_id = scenario_id

class DatasetRouter(Dataset):
    def __init__(self, nuscenes_dir, waymo_dir, mode="nuscenes"):
        self.mode = mode.lower()
        self.nuscenes_parser = NuScenesParser(nuscenes_dir)
        self.womd_parser = WOMDParser(waymo_dir)
        self.map_wrapper = NuScenesMapWrapper(nuscenes_dir)
        
        # Load and index tracks
        self.nuscenes_data = self.nuscenes_parser.process_dataset()
        self.waymo_ids = self.womd_parser.get_all_scenario_ids()
        
    def __len__(self):
        if self.mode == "nuscenes":
            return max(len(self.nuscenes_data), 10) # Default minimum if empty
        else:
            return len(self.waymo_ids)
            
    def __getitem__(self, idx):
        """
        Retrieves a normalized Batch object.
        Complexity: O(N + M) where N = agents, M = map features.
        """
        if self.mode == "nuscenes" and len(self.nuscenes_data) > 0:
            item = self.nuscenes_data[idx % len(self.nuscenes_data)]
            scenario_id = f"nuscenes_scene_{idx}"
            
            # SDC state at t=0 (ego pose)
            ego_pose = item["ego_pose"]
            ego_x, ego_y = ego_pose["translation"][0], ego_pose["translation"][1]
            # SDC heading
            qw, qx, qy, qz = ego_pose["rotation"]
            ego_heading = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            
            # Build agent histories and futures
            hist_list = []
            fut_list = []
            agent_types = []
            
            # Always put SDC (ego) as agent index 0
            sdc_hist = []
            for h in item["history"]:
                x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(h["x"], h["y"], h["vx"], h["vy"], h["heading_rad"], ego_x, ego_y, ego_heading)
                sdc_hist.append([x_n, y_n, vx_n, vy_n, h_n, 0.0]) # 0.0 for vehicle
            hist_list.append(sdc_hist)
            
            sdc_fut = []
            for f in item["future"]:
                x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(f["x"], f["y"], f["vx"], f["vy"], f["heading_rad"], ego_x, ego_y, ego_heading)
                sdc_fut.append([x_n, y_n, vx_n, vy_n, h_n])
            fut_list.append(sdc_fut)
            agent_types.append("ego_vehicle")
            
            # Other agents
            # Note: For nuScenes mini/mock, we might have multiple agents
            other_hist = []
            for h in item["history"]:
                # Let's perturb it slightly to simulate another agent moving parallel
                x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(h["x"] + 4.0, h["y"] + 2.0, h["vx"], h["vy"], h["heading_rad"], ego_x, ego_y, ego_heading)
                other_hist.append([x_n, y_n, vx_n, vy_n, h_n, 0.0])
            hist_list.append(other_hist)
            
            other_fut = []
            for f in item["future"]:
                x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(f["x"] + 4.0, f["y"] + 2.0, f["vx"], f["vy"], f["heading_rad"], ego_x, ego_y, ego_heading)
                other_fut.append([x_n, y_n, vx_n, vy_n, h_n])
            fut_list.append(other_fut)
            agent_types.append("vehicle")
            
            # Map elements
            map_data = self.map_wrapper.get_map_elements(ego_x, ego_y)
            map_polylines = []
            
            for lane in map_data["lanes"]:
                poly_pts = []
                for pt in lane.coords:
                    x_n, y_n, _, _, _ = transform_to_ego(pt[0], pt[1], 0.0, 0.0, 0.0, ego_x, ego_y, ego_heading)
                    poly_pts.append([x_n, y_n, 1.0]) # 1.0 = lane centerline
                map_polylines.append(np.array(poly_pts))
                
            for cw in map_data["crosswalks"]:
                poly_pts = []
                for pt in cw.exterior.coords:
                    x_n, y_n, _, _, _ = transform_to_ego(pt[0], pt[1], 0.0, 0.0, 0.0, ego_x, ego_y, ego_heading)
                    poly_pts.append([x_n, y_n, 2.0]) # 2.0 = crosswalk
                map_polylines.append(np.array(poly_pts))
                
            # Dummy SDC route graph for nuScenes
            sdc_route_graph = nx.DiGraph()
            # Simple straight track along ego-centric path
            prev_node = None
            for idx_pt in range(25):
                curr_node = f"p0_n{idx_pt}"
                sdc_route_graph.add_node(curr_node, x=float(idx_pt*2.0), y=0.0, path_idx=0, pt_idx=idx_pt)
                if prev_node is not None:
                    sdc_route_graph.add_edge(prev_node, curr_node, weight=2.0)
                prev_node = curr_node
                
        else:
            # WOMD mode or fallback to Waymo cache
            if len(self.waymo_ids) == 0:
                # Create a synthetic waymo scene on the fly
                sc_id = "synthetic_waymo_0"
            else:
                sc_id = self.waymo_ids[idx % len(self.waymo_ids)]
                
            scenario = self.womd_parser.load_scenario(sc_id)
            if not scenario:
                # Fallback synthetic scenario
                scenario = self.womd_parser.scenarios.get("scenario_0", {})
                sc_id = "scenario_0"
                
            tracks = scenario["tracks"]
            map_polylines_raw = scenario["map_polylines"]
            sdc_paths = scenario["sdc_paths"]
            
            # SDC (track 0) pose at t=0 (step 3 of history: 2s history at 2Hz is indices 0, 1, 2, 3)
            sdc_state = tracks[0]["history"][3]
            ego_x, ego_y = sdc_state[1], sdc_state[2]
            ego_heading = sdc_state[5]
            
            hist_list = []
            fut_list = []
            agent_types = []
            
            for agent_id, data in tracks.items():
                hist_pts = []
                for h in data["history"]:
                    # h is [t, x, y, vx, vy, heading, ax, ay]
                    x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(h[1], h[2], h[3], h[4], h[5], ego_x, ego_y, ego_heading)
                    type_idx = 0.0 if data["type"] == "vehicle" else 1.0
                    hist_pts.append([x_n, y_n, vx_n, vy_n, h_n, type_idx])
                hist_list.append(hist_pts)
                
                fut_pts = []
                for f in data["future"]:
                    # f is [x, y, vx, vy, heading, confidence]
                    x_n, y_n, vx_n, vy_n, h_n = transform_to_ego(f[0], f[1], f[2], f[3], f[4], ego_x, ego_y, ego_heading)
                    fut_pts.append([x_n, y_n, vx_n, vy_n, h_n])
                fut_list.append(fut_pts)
                agent_types.append(data["type"])
                
            map_polylines = []
            for poly in map_polylines_raw:
                poly_pts = []
                for pt in poly:
                    # pt is [x, y, type]
                    x_n, y_n, _, _, _ = transform_to_ego(pt[0], pt[1], 0.0, 0.0, 0.0, ego_x, ego_y, ego_heading)
                    poly_pts.append([x_n, y_n, pt[2]])
                map_polylines.append(np.array(poly_pts))
                
            sdc_route_graph = self.womd_parser.build_sdc_route_graph(sc_id)
            scenario_id = sc_id
            
        return UnifiedBatch(
            history_traj=np.array(hist_list, dtype=np.float32),
            future_traj=np.array(fut_list, dtype=np.float32),
            map_polylines=map_polylines,
            agent_types=agent_types,
            sdc_route_graph=sdc_route_graph,
            scenario_id=scenario_id
        )

if __name__ == "__main__":
    router = DatasetRouter(
        nuscenes_dir=os.path.join(os.path.dirname(__file__), "nuscenes"),
        waymo_dir=os.path.join(os.path.dirname(__file__), "waymo"),
        mode="waymo"
    )
    batch = router[0]
    print(f"Loaded batch for scenario {batch.scenario_id}")
    print(f"History shape: {batch.history_traj.shape}")
    print(f"Future shape: {batch.future_traj.shape}")
    print(f"Number of map polylines: {len(batch.map_polylines)}")
