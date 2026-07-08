import os
import tarfile
import hashlib
import sys
import argparse

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def extract_archive(archive_path, dest_dir):
    print(f"Extracting {archive_path} to {dest_dir}...")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    try:
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(path=dest_dir)
        print(f"Finished extracting {archive_path}")
        return True
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False

def generate_mock_nuscenes(dest_dir):
    print("Generating mock nuScenes dataset for testing...")
    # Create required directory structure for mini
    v1_mini_dir = os.path.join(dest_dir, "v1.0-mini")
    os.makedirs(v1_mini_dir, exist_ok=True)
    
    # We will create dummy json tables for nuscenes devkit validation
    tables = ["scene.json", "sample.json", "sample_data.json", "sample_annotation.json", 
              "instance.json", "category.json", "attribute.json", "visibility.json", 
              "ego_pose.json", "calibrated_sensor.json", "sensor.json", "log.json", "map.json"]
    
    import json
    for table in tables:
        file_path = os.path.join(v1_mini_dir, table)
        dummy_data = []
        if table == "scene.json":
            dummy_data = [{"token": "scene_0", "name": "scene-0001", "description": "Mock scene for testing", 
                           "first_sample_token": "sample_0", "last_sample_token": "sample_24", 
                           "nbr_samples": 25, "log_token": "log_0"}]
        elif table == "sample.json":
            dummy_data = [{"token": f"sample_{i}", "timestamp": 1500000000000000 + i * 500000, 
                           "scene_token": "scene_0", "prev": f"sample_{i-1}" if i > 0 else "", 
                           "next": f"sample_{i+1}" if i < 24 else ""} for i in range(25)]
        elif table == "ego_pose.json":
            dummy_data = [{"token": f"ego_pose_{i}", "timestamp": 1500000000000000 + i * 500000,
                           "translation": [i * 2.0, i * 0.5, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]} for i in range(25)]
        elif table == "sample_annotation.json":
            dummy_data = [{"token": f"ann_{i}", "sample_token": f"sample_{i}", "instance_token": "inst_0",
                           "visibility_token": "1", "translation": [i * 2.2 + 5.0, i * 0.4 + 1.0, 0.0],
                           "size": [2.0, 4.0, 1.5], "rotation": [1.0, 0.0, 0.0, 0.0], 
                           "attribute_tokens": [], "next": f"ann_{i+1}" if i < 24 else "", "prev": f"ann_{i-1}" if i > 0 else ""} for i in range(25)]
        elif table == "instance.json":
            dummy_data = [{"token": "inst_0", "category_token": "cat_0", "nbr_annotations": 25,
                           "first_annotation_token": "ann_0", "last_annotation_token": "ann_24"}]
        elif table == "category.json":
            dummy_data = [{"token": "cat_0", "name": "vehicle.car", "description": "Vehicle"}]
        elif table == "log.json":
            dummy_data = [{"token": "log_0", "map_token": "map_0", "date_captured": "2026-06-11", "location": "singapore-onenorth"}]
        elif table == "map.json":
            dummy_data = [{"token": "map_0", "log_tokens": ["log_0"], "category": "semantic_prior", "filename": "maps/mock_map.png"}]
        
        with open(file_path, 'w') as f:
            json.dump(dummy_data, f, indent=2)
                
    # Create maps directory
    os.makedirs(os.path.join(dest_dir, "maps"), exist_ok=True)
    print("Mock nuScenes dataset created.")

def generate_mock_waymo(dest_dir):
    print("Generating mock WOMD dataset...")
    os.makedirs(dest_dir, exist_ok=True)
    mock_file = os.path.join(dest_dir, "mock_scenario.pkl")
    import pickle
    import numpy as np
    
    # Generate mock scenarios
    mock_scenarios = {}
    for sc_idx in range(15):
        sc_id = f"scenario_{sc_idx}"
        # Agent tracks
        agent_ids = [0, 1, 2] # 0 is SDC
        tracks = {}
        for aid in agent_ids:
            # 2s history (4 steps at 2Hz) -> 6s future (12 steps)
            history = np.zeros((4, 8)) # t, x, y, vx, vy, heading, ax, ay
            future = np.zeros((12, 6)) # x, y, vx, vy, heading, confidence
            
            # Straight line movement with noise
            heading = 0.5 if aid == 0 else -0.2
            v = 10.0 if aid == 0 else 5.0
            
            # SDC starts at 0,0
            start_x = 0.0 if aid == 0 else (15.0 if aid == 1 else -10.0)
            start_y = 0.0 if aid == 0 else (5.0 if aid == 1 else -5.0)
            
            for t_idx in range(4):
                t = -1.5 + t_idx * 0.5
                dx = v * t * np.cos(heading)
                dy = v * t * np.sin(heading)
                history[t_idx] = [t, start_x + dx, start_y + dy, v * np.cos(heading), v * np.sin(heading), heading, 0.0, 0.0]
                
            for t_idx in range(12):
                t = 0.5 + t_idx * 0.5
                dx = v * t * np.cos(heading)
                dy = v * t * np.sin(heading)
                future[t_idx] = [start_x + dx, start_y + dy, v * np.cos(heading), v * np.sin(heading), heading, 1.0]
                
            tracks[aid] = {
                "history": history,
                "future": future,
                "type": "vehicle" if aid < 2 else "pedestrian",
                "length": 4.5,
                "width": 1.8
            }
            
        # Map features
        map_polylines = []
        # Lane 1 (ego lane)
        lane1 = np.zeros((20, 3))
        for i in range(20):
            x = -50.0 + i * 5.0
            y = 0.0
            lane1[i] = [x, y, 1] # x, y, lane_type
        map_polylines.append(lane1)
        
        # Lane 2 (left lane)
        lane2 = np.zeros((20, 3))
        for i in range(20):
            x = -50.0 + i * 5.0
            y = 3.5
            lane2[i] = [x, y, 1]
        map_polylines.append(lane2)
        
        # Crosswalk
        crosswalk = np.zeros((5, 3))
        for i in range(5):
            x = 20.0
            y = -10.0 + i * 5.0
            crosswalk[i] = [x, y, 2] # crosswalk type
        map_polylines.append(crosswalk)
        
        # SDC paths (TNT route graph candidate paths)
        # We will represent paths as list of coordinate lists
        sdc_paths = [
            np.array([[x, 0.0] for x in np.arange(0.0, 50.0, 2.0)]), # straight
            np.array([[x, x*0.1] for x in np.arange(0.0, 50.0, 2.0)]), # slight right
            np.array([[x, -x*0.1] for x in np.arange(0.0, 50.0, 2.0)]), # slight left
        ]
        
        mock_scenarios[sc_id] = {
            "tracks": tracks,
            "map_polylines": map_polylines,
            "sdc_paths": sdc_paths
        }
        
    with open(mock_file, 'wb') as f:
        pickle.dump(mock_scenarios, f)
    print("Mock Waymo dataset created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true", help="Extract actual archives")
    parser.add_argument("--test", action="store_true", help="Just generate mock datasets for testing")
    args = parser.parse_args()
    
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    nuscenes_dest = os.path.join(os.path.dirname(__file__), "nuscenes")
    waymo_dest = os.path.join(os.path.dirname(__file__), "waymo")
    
    if args.test:
        generate_mock_nuscenes(nuscenes_dest)
        generate_mock_waymo(waymo_dest)
        sys.exit(0)
        
    # Check for zip/tar files in workspace
    mini_archive = os.path.join(workspace_dir, "v1.0-mini.tgz")
    val_archive = os.path.join(workspace_dir, "val.tar")
    
    mini_extracted = False
    val_extracted = False
    
    if os.path.exists(mini_archive):
        print(f"Found v1.0-mini.tgz (size: {os.path.getsize(mini_archive)} bytes)")
        mini_extracted = extract_archive(mini_archive, nuscenes_dest)
    else:
        print("v1.0-mini.tgz not found. Generating mock instead.")
        generate_mock_nuscenes(nuscenes_dest)
        
    if os.path.exists(val_archive):
        print(f"Found val.tar (size: {os.path.getsize(val_archive)} bytes)")
        val_extracted = extract_archive(val_archive, nuscenes_dest)
    else:
        print("val.tar not found. Generating mock instead.")
        
    # Always generate waymo mock as the tfrecord requires specific TF processing which we'll implement in womd_parser.py
    generate_mock_waymo(waymo_dest)
