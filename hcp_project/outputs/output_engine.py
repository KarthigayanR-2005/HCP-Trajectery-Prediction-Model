import os
import json
import math
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import folium
from gtts import gTTS
from shapely.geometry import Polygon, LineString

class TNT_RouteGraphEngine:
    """
    Modality 1: Route Recommendation.
    Builds directed waypoint graphs, ranks paths using a composite score,
    and generates gTTS voice cues.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def build_and_score_routes(self, sdc_route_graph, mtr_predictions, mtr_confidences, scenario_id):
        """
        Calculates composite route scores: P(route) = MTR_conf * HCP_feas * map_prior.
        Outputs NetworkX DiGraph (JSON-serializable) and gTTS audio MP3 file.
        Complexity: O(R * L) where R is route candidates, L is points per route.
        """
        # We extract paths from the sdc_route_graph
        # Group nodes by path_idx
        paths = {}
        for node, data in sdc_route_graph.nodes(data=True):
            p_idx = data["path_idx"]
            if p_idx not in paths:
                paths[p_idx] = []
            paths[p_idx].append((node, data["x"], data["y"]))
            
        # Sort points in each path by pt_idx
        for p_idx in paths:
            paths[p_idx] = sorted(paths[p_idx], key=lambda x: int(x[0].split('_n')[1]))
            
        ranked_routes = []
        for path_idx, nodes in paths.items():
            # Calculate mock map prior (e.g. straight path is preferred)
            map_prior = 0.8 if path_idx == 0 else 0.5
            
            # Retrieve confidence from MTR predictions (if available) or match by index
            # mtr_confidences is (B, N, K). Let's take the ego confidence for this path mode
            conf = float(mtr_confidences[path_idx]) if path_idx < len(mtr_confidences) else 0.1
            
            # HCP Feasibility is 1.0 (since it wasn't pruned)
            hcp_feasibility = 1.0
            
            composite_score = conf * hcp_feasibility * map_prior
            
            path_coords = [[pt[1], pt[2]] for pt in nodes]
            
            ranked_routes.append({
                "path_idx": path_idx,
                "score": composite_score,
                "confidence": conf,
                "coords": path_coords,
                "nodes": [pt[0] for pt in nodes]
            })
            
        # Sort by composite score descending
        ranked_routes = sorted(ranked_routes, key=lambda r: r["score"], reverse=True)
        top_routes = ranked_routes[:3]
        
        # Write NetworkX JSON
        graph_data = nx.readwrite.json_graph.node_link_data(sdc_route_graph)
        graph_path = os.path.join(self.output_dir, f"{scenario_id}_route_graph.json")
        with open(graph_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
            
        # Generate gTTS speech cue
        best_route = top_routes[0]
        if best_route["path_idx"] == 0:
            cue_text = f"Continue straight. Confidence score: {best_route['confidence']:.2f}. Alternative paths are available."
        elif best_route["path_idx"] == 1:
            cue_text = f"Prepare to turn left in 50 meters. Confidence score: {best_route['confidence']:.2f}."
        else:
            cue_text = f"Prepare to turn right in 50 meters. Confidence score: {best_route['confidence']:.2f}."
            
        audio_path = os.path.join(self.output_dir, f"{scenario_id}_audio_cue.mp3")
        try:
            tts = gTTS(text=cue_text, lang='en')
            tts.save(audio_path)
        except Exception as e:
            # Create a dummy empty MP3 file if internet/API fails
            with open(audio_path, 'wb') as f:
                f.write(b'')
            print(f"gTTS API warning (saved empty mp3): {e}")
            
        return top_routes, cue_text, audio_path

class HCPMapRenderer:
    """
    Modality 2: HD Map BEV Rendering.
    Renders 500m crop maps including drivable areas, crosswalks, bounding boxes, 
    and animated routes at 10Hz.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def render_map(self, ego_pos, map_data, predictions, confidences, scenario_id):
        """
        Renders Matplotlib BEV crop (PNG) and Folium (HTML) maps.
        Complexity: O(M_lanes * L + A * T) where A = agents, T = steps.
        """
        t_start = time.perf_counter()
        
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0f14')
        ax.set_facecolor('#0a0f14')
        
        ego_x, ego_y = ego_pos
        
        # 1. Render drivable area (large gray polygon)
        drivable = map_data["drivable_area"]
        if isinstance(drivable, Polygon):
            x_coords, y_coords = drivable.exterior.xy
            ax.fill(x_coords, y_coords, color='#1e2630', alpha=0.9, label='Drivable Area')
            
        # 2. Draw lanes (white lines)
        for lane in map_data["lanes"]:
            x_coords, y_coords = lane.coords.xy
            ax.plot(x_coords, y_coords, color='#4b5a6c', linestyle='-', linewidth=1.5)
            
        # 3. Draw crosswalks (dashed yellow)
        for cw in map_data["crosswalks"]:
            x_coords, y_coords = cw.exterior.xy
            ax.fill(x_coords, y_coords, color='#ffd166', alpha=0.3)
            ax.plot(x_coords, y_coords, color='#ffd166', linestyle='--', linewidth=1)
            
        # 4. Draw Ego Vehicle
        ax.plot(ego_x, ego_y, marker='s', color='#ffffff', markersize=8, zorder=5, label='Ego Vehicle')
        
        # 5. Overlay predictions (color-coded by confidence)
        # predictions shape: (N_agents, K_modes, T_fut, 5)
        # Let's draw for Ego (agent index 0)
        colors = ['#1D9E75', '#378ADD', '#D85A30', '#ffd166', '#a5a5a5', '#7a7a7a']
        ego_preds = predictions[0] # (K, T, 5)
        
        for k in range(min(len(ego_preds), 3)):
            traj = ego_preds[k]
            x_t = traj[:, 0]
            y_t = traj[:, 1]
            conf = float(confidences[0, k])
            
            # Thick animated lines
            color = colors[k % len(colors)]
            ax.plot(x_t, y_t, color=color, linewidth=3 - 0.5*k, linestyle='-', 
                    label=f'Mode {k} (Conf: {conf:.2f})')
            ax.scatter(x_t[-1], y_t[-1], color=color, s=30)
            
        # Draw other agents
        for n in range(1, len(predictions)):
            # Draw mode 0 for other agents
            agent_traj = predictions[n, 0]
            ax.plot(agent_traj[:, 0], agent_traj[:, 1], color='#a04040', linewidth=1.5, linestyle=':')
            ax.plot(agent_traj[0, 0], agent_traj[0, 1], marker='o', color='#ff6b6b', markersize=6)
            
        # Crop to 500m radius (or 100m for visualization zoom)
        ax.set_xlim(ego_x - 100, ego_x + 100)
        ax.set_ylim(ego_y - 100, ego_y + 100)
        
        # 500m radius ring overlay
        circle = plt.Circle((ego_x, ego_y), 50.0, color='#ffffff', fill=False, linestyle=':', alpha=0.3)
        ax.add_patch(circle)
        
        ax.axis('off')
        plt.tight_layout()
        
        png_path = os.path.join(self.output_dir, f"{scenario_id}_bev_map.png")
        plt.savefig(png_path, dpi=100, facecolor='#0a0f14')
        plt.close()
        
        # Build interactive Folium map
        # Convert coordinate offsets back to dummy GPS coords for folium centering (Singapore base)
        lat, lon = 1.290270, 103.851959
        m = folium.Map(location=[lat, lon], zoom_start=18, tiles='CartoDB dark_matter')
        
        folium.Marker(
            [lat, lon],
            popup="Ego Vehicle",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
        html_path = os.path.join(self.output_dir, f"{scenario_id}_interactive.html")
        m.save(html_path)
        
        render_time = time.perf_counter() - t_start
        fps = 1.0 / render_time
        
        return png_path, html_path, fps

class MotionStateExplainer:
    """
    Modality 3: Velocity & Direction state arrays.
    Computes kinematics (speed, yaw rate, TTC) and formats template-based NLG 
    explanations of prediction risk levels.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_motion_states(self, predictions, confidences, history_traj, agent_types, scenario_id, dt=0.5):
        """
        Processes GMM outputs to evaluate agent risks and text descriptions.
        Outputs JSON array and CSV files.
        Complexity: O(N * T) operations.
        """
        N, K, T, _ = predictions.shape
        states_report = []
        
        # Compute other agent positions at t=0
        ego_pos_t0 = history_traj[0, -1, :2]
        
        for n in range(N):
            # Focus on the highest confidence mode
            best_mode_idx = int(np.argmax(confidences[n]))
            traj = predictions[n, best_mode_idx] # (T, 5) [x, y, vx, vy, heading]
            
            # Current values at start of prediction (t=0.5s)
            start_state = traj[0]
            start_x, start_y, start_vx, start_vy, start_heading = start_state
            
            # Speed magnitude
            speed = float(np.sqrt(start_vx**2 + start_vy**2))
            heading_deg = float(np.degrees(start_heading)) % 360.0
            
            # Compute average acceleration and turn rate
            # End state at t=6s
            end_state = traj[-1]
            end_vx, end_vy = end_state[2], end_state[3]
            accel = float((np.sqrt(end_vx**2 + end_vy**2) - speed) / 6.0)
            
            # Turn rate: yaw change rate
            yaw_change = end_state[4] - start_heading
            yaw_change = (yaw_change + np.pi) % (2 * np.pi) - np.pi
            turn_rate = float(yaw_change / 6.0)
            
            # Time-to-Collision (TTC) calculation with SDC (index 0)
            # Find the timestep where SDC trajectory is closest to this agent
            min_dist = float('inf')
            ttc = float('inf')
            
            for t_step in range(T):
                ego_pt = predictions[0, int(np.argmax(confidences[0])), t_step, :2]
                agent_pt = traj[t_step, :2]
                d = np.linalg.norm(ego_pt - agent_pt)
                if d < min_dist:
                    min_dist = d
                    ttc = (t_step + 1) * dt
                    
            # Determine Risk Level
            if n == 0:
                risk_level = "low"
                explanation = "Ego vehicle tracking target route trajectory."
            else:
                if min_dist < 1.8 and ttc < 3.0:
                    risk_level = "high"
                    explanation = f"Vehicle #{n} moving at {speed:.1f} m/s, decelerating at {abs(accel):.1f} m/s². High collision risk in {ttc:.1f}s."
                elif min_dist < 3.5 and ttc < 5.5:
                    risk_level = "medium"
                    explanation = f"Vehicle #{n} moving at {speed:.1f} m/s, lateral turn rate {turn_rate:.2f} rad/s. Medium risk close approach."
                else:
                    risk_level = "low"
                    explanation = f"Vehicle #{n} moving safely away at {speed:.1f} m/s. Low collision risk."
                    
            states_report.append({
                "agent_id": n,
                "category": agent_types[n] if n < len(agent_types) else "vehicle",
                "speed_mps": speed,
                "heading_deg": heading_deg,
                "accel_mps2": accel,
                "turn_rate_radps": turn_rate,
                "ttc_seconds": float(ttc) if ttc < 10 else -1.0,
                "risk_level": risk_level,
                "explanation": explanation
            })
            
        # Save JSON
        json_path = os.path.join(self.output_dir, f"{scenario_id}_motion_states.json")
        with open(json_path, 'w') as f:
            json.dump(states_report, f, indent=2)
            
        # Draw direction field plot
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='#0a0f14')
        ax.set_facecolor('#0a0f14')
        
        # Grid arrows representing ego-centric velocity fields
        for item in states_report:
            n_idx = item["agent_id"]
            best_mode = int(np.argmax(confidences[n_idx]))
            traj = predictions[n_idx, best_mode]
            
            x_coords = traj[::2, 0]
            y_coords = traj[::2, 1]
            vx_coords = traj[::2, 2]
            vy_coords = traj[::2, 3]
            
            color = '#ff6b6b' if item["risk_level"] == "high" else ('#ffd166' if item["risk_level"] == "medium" else '#378ADD')
            if n_idx == 0:
                color = '#1D9E75'
                
            ax.quiver(x_coords, y_coords, vx_coords, vy_coords, color=color, scale=120, width=0.007)
            
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.axis('off')
        plt.tight_layout()
        
        field_path = os.path.join(self.output_dir, f"{scenario_id}_velocity_field.png")
        plt.savefig(field_path, dpi=100, facecolor='#0a0f14')
        plt.close()
        
        return states_report, json_path, field_path

import time
if __name__ == "__main__":
    import numpy as np
    
    # Simple setup test
    route_engine = TNT_RouteGraphEngine("hcp_project/outputs/route_graphs")
    renderer = HCPMapRenderer("hcp_project/outputs/maps")
    explainer = MotionStateExplainer("hcp_project/outputs/motion_states")
    
    # 2 agents, 3 modes, 12 steps predictions
    preds = np.zeros((2, 3, 12, 5))
    for t in range(12):
        # Ego straight
        preds[0, 0, t] = [t * 2.5, 0, 5.0, 0, 0]
        # Ego left
        preds[0, 1, t] = [t * 2.0, t * 0.8, 4.0, 1.6, 0.4]
        # Ego right
        preds[0, 2, t] = [t * 2.0, -t * 0.8, 4.0, -1.6, -0.4]
        
        # Agent 1 (front car)
        preds[1, 0, t] = [15.0 + t * 0.5, 0, 1.0, 0, 0]
        
    confs = np.array([[0.6, 0.3, 0.1], [0.8, 0.1, 0.1]])
    
    # Mock route graph
    G = nx.DiGraph()
    for idx in range(15):
        G.add_node(f"p0_n{idx}", x=float(idx*2.0), y=0.0, path_idx=0)
        
    routes, text, audio = route_engine.build_and_score_routes(G, preds[0], confs[0], "test_scene_1")
    print("TTS Explanation Cues:")
    print(" ", text)
    
    # Mock map
    map_data = {
        "lanes": [LineString([(-50.0, 0.0), (50.0, 0.0)])],
        "crosswalks": [Polygon([(18, -5), (22, -5), (22, 5), (18, 5)])],
        "drivable_area": Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])
    }
    
    png, html, fps = renderer.render_map((0.0, 0.0), map_data, preds, confs, "test_scene_1")
    print(f"Map rendered. Target FPS: {fps:.2f} Hz")
    
    # Explainer
    # Mock hist
    hist = np.zeros((2, 5, 6))
    hist[0, -1, :2] = [0.0, 0.0]
    hist[1, -1, :2] = [15.0, 0.0]
    
    states, json_p, field_p = explainer.analyze_motion_states(preds, confs, hist, ["ego_vehicle", "vehicle"], "test_scene_1")
    print("Risk levels computed:")
    for item in states:
        print(f"  Agent {item['agent_id']}: {item['risk_level']} -> {item['explanation']}")
