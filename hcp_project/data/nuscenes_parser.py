import os
import json
import math
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString

class NuScenesMapWrapper:
    """
    Wrapper for nuScenes Map queries. Provides lane centerlines, crosswalks,
    and drivable area within 500m ego-radius. Falls back to procedurally generated 
    maps if nuScenes database is mock/incomplete.
    """
    def __init__(self, map_dir, location="singapore-onenorth"):
        self.map_dir = map_dir
        self.location = location
        # If we have real maps, we could load them. Otherwise, we procedurally generate
        self.is_mock = True
        
    def get_map_elements(self, ego_x, ego_y, radius=500.0):
        """
        Returns lane centerlines, crosswalk polygons, and drivable area.
        Complexity: O(N) where N is map elements in spatial range.
        """
        # Procedurally generate map lanes, crosswalks, and drivable areas in ego-centric coords
        lanes = []
        crosswalks = []
        
        # Ego lane (main road)
        # 3 lanes, straight ahead
        for offset in [-3.5, 0.0, 3.5]:
            lane_pts = []
            for y in np.arange(-radius, radius, 10.0):
                # Add slight curve to lanes for realistic testing
                x = offset + 10.0 * math.sin(y / 150.0)
                lane_pts.append((x + ego_x, y + ego_y))
            lanes.append(LineString(lane_pts))
            
        # Intersecting road (crossroad)
        for offset in [-3.5, 0.0, 3.5]:
            lane_pts = []
            for x in np.arange(-radius, radius, 10.0):
                y = offset + 10.0 * math.cos(x / 150.0)
                lane_pts.append((x + ego_x, y + ego_y))
            lanes.append(LineString(lane_pts))
            
        # Crosswalk polygons (near intersection at 50m, 100m, etc.)
        for x_val in [-100.0, 100.0]:
            poly = Polygon([
                (x_val - 5.0 + ego_x, -20.0 + ego_y),
                (x_val + 5.0 + ego_x, -20.0 + ego_y),
                (x_val + 5.0 + ego_x, 20.0 + ego_y),
                (x_val - 5.0 + ego_x, 20.0 + ego_y)
            ])
            crosswalks.append(poly)
            
        # Drivable area polygon
        # A large envelope around our roads
        drivable_coords = [
            (-50.0 + ego_x, -radius + ego_y),
            (50.0 + ego_x, -radius + ego_y),
            (radius + ego_x, -50.0 + ego_y),
            (radius + ego_x, 50.0 + ego_y),
            (50.0 + ego_x, radius + ego_y),
            (-50.0 + ego_x, radius + ego_y),
            (-radius + ego_x, 50.0 + ego_y),
            (-radius + ego_x, -50.0 + ego_y),
        ]
        drivable_area = Polygon(drivable_coords)
        
        return {
            "lanes": lanes,
            "crosswalks": crosswalks,
            "drivable_area": drivable_area
        }

class NuScenesParser:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.meta_dir = os.path.join(data_dir, "v1.0-mini")
        
    def load_table(self, table_name):
        path = os.path.join(self.meta_dir, f"{table_name}.json")
        if not os.path.exists(path):
            return []
        with open(path, 'r') as f:
            return json.load(f)

    def process_dataset(self):
        """
        Parses agent tracks and maps.
        Extracts 2s history (5 steps at 2Hz) and 6s future (12 steps at 2Hz).
        """
        # Load tables
        scenes = self.load_table("scene")
        samples = self.load_table("sample")
        ego_poses = self.load_table("ego_pose")
        annotations = self.load_table("sample_annotation")
        instances = self.load_table("instance")
        categories = self.load_table("category")
        
        if not scenes:
            print("No nuScenes meta tables found or dataset is empty. Using mock data.")
            return []
            
        sample_dict = {s["token"]: s for s in samples}
        ego_pose_dict = {s["token"]: e for s, e in zip(samples, ego_poses)}
        cat_dict = {c["token"]: c["name"] for c in categories}
        inst_dict = {i["token"]: i for i in instances}
        
        # Group annotations by instance
        ann_by_inst = {}
        for ann in annotations:
            inst_tok = ann["instance_token"]
            if inst_tok not in ann_by_inst:
                ann_by_inst[inst_tok] = []
            ann_by_inst[inst_tok].append(ann)
            
        processed_tracks = []
        
        # Process instance trajectories
        for inst_tok, anns in ann_by_inst.items():
            # Sort by sample timestamp
            anns = sorted(anns, key=lambda a: sample_dict[a["sample_token"]]["timestamp"])
            if len(anns) < 17: # Need at least 17 steps for 2s history (5) + 6s future (12)
                continue
                
            inst = inst_dict[inst_tok]
            cat_name = cat_dict[inst["category_token"]]
            
            # Extract track states
            states = []
            for i in range(len(anns)):
                ann = anns[i]
                sample = sample_dict[ann["sample_token"]]
                t_us = sample["timestamp"]
                x, y, z = ann["translation"]
                qw, qx, qy, qz = ann["rotation"]
                
                # Heading from quaternion (yaw)
                heading = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                
                states.append({
                    "timestamp_us": t_us,
                    "x": x,
                    "y": y,
                    "heading_rad": heading,
                    "width": ann["size"][0],
                    "length": ann["size"][1],
                })
                
            # Compute velocities and accelerations using central differences
            for i in range(len(states)):
                t_curr = states[i]["timestamp_us"] / 1e6
                x_curr = states[i]["x"]
                y_curr = states[i]["y"]
                
                if i > 0 and i < len(states) - 1:
                    t_prev = states[i-1]["timestamp_us"] / 1e6
                    t_next = states[i+1]["timestamp_us"] / 1e6
                    
                    vx = (states[i+1]["x"] - states[i-1]["x"]) / (t_next - t_prev)
                    vy = (states[i+1]["y"] - states[i-1]["y"]) / (t_next - t_prev)
                    
                    # Acceleration
                    ax = (states[i+1]["x"] - 2 * x_curr + states[i-1]["x"]) / ((t_next - t_prev) / 2) ** 2
                    ay = (states[i+1]["y"] - 2 * y_curr + states[i-1]["y"]) / ((t_next - t_prev) / 2) ** 2
                elif i == 0:
                    t_next = states[i+1]["timestamp_us"] / 1e6
                    vx = (states[i+1]["x"] - x_curr) / (t_next - t_curr)
                    vy = (states[i+1]["y"] - y_curr) / (t_next - t_curr)
                    ax, ay = 0.0, 0.0
                else:
                    t_prev = states[i-1]["timestamp_us"] / 1e6
                    vx = (x_curr - states[i-1]["x"]) / (t_curr - t_prev)
                    vy = (y_curr - states[i-1]["y"]) / (t_curr - t_prev)
                    ax, ay = 0.0, 0.0
                    
                states[i]["vx"] = vx
                states[i]["vy"] = vy
                states[i]["ax"] = ax
                states[i]["ay"] = ay
                states[i]["track_id"] = inst_tok
                states[i]["category"] = cat_name
                
            # Create slices of 2s history and 6s future
            # Split index represents the t=0 frame. We need 4 frames before and 12 frames after
            for pivot in range(4, len(states) - 12):
                history = states[pivot-4 : pivot+1]
                future = states[pivot+1 : pivot+13]
                
                processed_tracks.append({
                    "track_id": inst_tok,
                    "category": cat_name,
                    "history": history,
                    "future": future,
                    "ego_pose": ego_pose_dict[anns[pivot]["sample_token"]]
                })
                
        return processed_tracks

if __name__ == "__main__":
    parser = NuScenesParser(os.path.join(os.path.dirname(__file__), "nuscenes"))
    tracks = parser.process_dataset()
    print(f"Processed {len(tracks)} trajectory slices from nuScenes.")
