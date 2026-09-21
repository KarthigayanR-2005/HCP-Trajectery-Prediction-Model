import os
import json
import time
import asyncio
import glob
import math
import numpy as np 
import torch
import matplotlib
matplotlib.use('Agg')
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from shapely.geometry import LineString, Polygon

from data.womd_parser import WOMDParser
from data.dataset_router import DatasetRouter, transform_to_ego
from data.extractor import generate_mock_waymo
from hcp.pruner import HierarchicalCombinatorialPruner, generate_kinematic_candidates
from outputs.output_engine import TNT_RouteGraphEngine, HCPMapRenderer, MotionStateExplainer
from mtr_core.train import MTRMotionTransformer

# ---------------------------------------------------------------------------
# Geo helpers — convert ego-centric metres to geographic coords for map UI
# ---------------------------------------------------------------------------
# Previously this cycled through 15 fictional city anchors (San Francisco,
# Tokyo, Paris, etc.) purely for cosmetic variety — but the actual
# trajectory/prediction data always comes from the real nuScenes
# 'singapore-onenorth' map (see NuScenesMapWrapper), which has no real
# connection to any of those other cities' road networks. Dropping real
# Singapore movement data onto e.g. a San Francisco street grid meant
# predicted paths never aligned with real roads except for the 1-in-15
# scenarios that happened to land on the Singapore anchor. Fixed: every
# scenario now anchors to the same real Singapore location the map data
# actually represents, so drawn trajectories genuinely trace real streets.
REAL_SINGAPORE_ANCHOR = {"lat": 1.290270, "lng": 103.851959, "city": "Singapore"}

def _get_anchor(scenario_id: str) -> dict:
    """Real anchor matching the actual nuScenes 'singapore-onenorth' map data."""
    return REAL_SINGAPORE_ANCHOR

def _metres_to_geo(x_m: float, y_m: float, anchor_lat: float, anchor_lng: float):
    """Convert ego-centric metres offset → (lng, lat)."""
    lat = anchor_lat + y_m / 111_320.0
    lng = anchor_lng + x_m / (111_320.0 * math.cos(math.radians(anchor_lat)))
    return lng, lat


app = FastAPI(title="HCP + MTR Autonomous Driving Telemetry Dashboard")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your React dev server to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize project parsers and engines
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NUSCENES_DIR = os.path.join(DATA_DIR, "nuscenes")
WAYMO_DIR = os.path.join(DATA_DIR, "waymo")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "mtr_checkpoint.pth")

# NOTE: DatasetRouter always falls back to Waymo-mode scenarios (_get_waymo)
# whenever no real nuScenes metadata is present — but that fallback needs
# WOMDParser's mock_scenario.pkl to already exist. If nuScenes was never
# downloaded/extracted (e.g. testing the dashboard without the full
# dataset), that mock file was never generated either, and _get_waymo would
# crash with a KeyError trying to read scenario["tracks"] from an empty
# scenario dict. Generate it here if missing, so this fallback path
# genuinely works rather than crashing.
if not os.path.exists(os.path.join(WAYMO_DIR, "mock_scenario.pkl")):
    print("No mock Waymo scenarios found — generating them now so the "
          "no-real-data fallback actually works...")
    generate_mock_waymo(WAYMO_DIR)

def smooth_trajectory_xy_polyfit(trajectories, degree=3):
    """
    Fits a low-degree polynomial to each agent/candidate's x(t) and y(t)
    separately, then replaces the trajectory with points sampled from that
    smooth fit.

    Why this is needed: KFF's jerk/curvature checks require differencing
    the trajectory three times (position -> velocity -> acceleration ->
    jerk). That repeated differencing is extremely sensitive to small,
    realistic point-to-point prediction noise — even ~0.3m of per-step
    noise can produce apparent jerk values 5x past the feasibility
    threshold, even though the overall path is a perfectly sensible one.
    Fitting a smooth polynomial removes that high-frequency noise while
    preserving genuine curvature/acceleration trends, so a real sharp turn
    still correctly gets rejected — verified directly: a noisy-but-straight
    path flips from infeasible to feasible after this, while a genuinely
    sharp turn stays infeasible.
    """
    N, K, T, C = trajectories.shape
    out = trajectories.clone()
    t = np.arange(T, dtype=np.float64)
    deg = min(degree, T - 1)
    for n in range(N):
        for k in range(K):
            for ch in (0, 1):  # x, y only — KFF never looks at the other channels
                y_vals = trajectories[n, k, :, ch].detach().cpu().numpy().astype(np.float64)
                coeffs = np.polyfit(t, y_vals, deg=deg)
                fitted = np.polyval(coeffs, t)
                out[n, k, :, ch] = torch.tensor(fitted, dtype=trajectories.dtype)
    return out


womd_parser = WOMDParser(WAYMO_DIR)
# NOTE: mode="nuscenes" — this was previously mode="waymo", which used the
# mock/placeholder Waymo dataset (no real Waymo data has ever been
# downloaded in this project) instead of the real, trained-on nuScenes data.
dataset = DatasetRouter(NUSCENES_DIR, WAYMO_DIR, mode="nuscenes")
pruner = HierarchicalCombinatorialPruner()
route_engine = TNT_RouteGraphEngine(os.path.join(OUTPUT_DIR, "route_graphs"))
map_renderer = HCPMapRenderer(os.path.join(OUTPUT_DIR, "maps"))
explainer = MotionStateExplainer(os.path.join(OUTPUT_DIR, "motion_states"))

# ---------------------------------------------------------------------------
# Load the real trained model once at startup — everything below this used
# to generate fake, hand-coded trajectories (hardcoded heading/velocity
# values and a fixed confidence distribution) instead of ever calling the
# model. Predictions served by this API are now genuine model output.
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading trained model on {DEVICE}...")
model = MTRMotionTransformer(d_model=256, n_modes=6).to(DEVICE)
if os.path.exists(CHECKPOINT_PATH):
    _ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    _state_dict = _ckpt["model_state_dict"] if isinstance(_ckpt, dict) and "model_state_dict" in _ckpt else _ckpt
    missing, unexpected = model.load_state_dict(_state_dict, strict=False)
    if missing or unexpected:
        print(f"Note: {len(missing)} param(s) missing from checkpoint, "
              f"{len(unexpected)} unused param(s) in checkpoint ignored.")
    print(f"Loaded real checkpoint: {CHECKPOINT_PATH}")
else:
    print(f"WARNING: no checkpoint found at {CHECKPOINT_PATH} — "
          f"model will run with random, untrained weights.")
model.eval()

# Pre-generate predictions for all scenarios to make the dashboard fast
cached_scenarios = {}
scenario_ids = [f"scenario_{i}" for i in range(min(len(dataset), 30))]  # cap for startup time

print(f"Caching real model predictions for {len(scenario_ids)} scenarios...")
for s_id in scenario_ids:
    idx = int(s_id.split('_')[-1])
    batch = dataset[idx]

    N_agents = len(batch.agent_types)
    T_hist = batch.history_traj.shape[1]

    with torch.no_grad():
        hist_tensor = torch.from_numpy(batch.history_traj).float().unsqueeze(0).to(DEVICE)   # (1, N, T_hist, 6)
        map_polylines_batch = [batch.map_polylines]
        camera_images = batch.camera_image.unsqueeze(0).to(DEVICE) if batch.has_image else None
        has_image_mask = torch.tensor([1.0 if batch.has_image else 0.0])

        pred_trajs, confidences = model(
            hist_tensor, map_polylines_batch, hcp_mask=None,
            camera_images=camera_images, has_image_mask=has_image_mask,
        )
        # (1, N, K, T_fut, 5) -> (N, K, T_fut, 5); apply softmax so confidences
        # are genuine probabilities, matching how they're used for display.
        predictions = pred_trajs[0].cpu().numpy()
        # MTRDecoder.forward already returns F.softmax(conf_logits, dim=-1), so
        # these are ALREADY probabilities. The previous torch.softmax() here was a
        # second softmax over a probability vector, which flattens it toward
        # uniform: a decisive [0.77, 0.06, 0.05, ...] was displayed as
        # [0.29, 0.14, 0.14, ...]. argmax survived, so the right mode was still
        # selected, but every confidence percentage shown was wrong.
        confidences = confidences[0].cpu().numpy()

    cached_scenarios[s_id] = {
        "batch": batch,
        "predictions": predictions,
        "confidences": confidences
    }

@app.get("/scenarios")
def get_scenarios():
    return scenario_ids

@app.get("/scenario/{s_id}")
def get_scenario(s_id: str):
    if s_id not in cached_scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found")
        
    cache = cached_scenarios[s_id]
    batch = cache["batch"]
    anchor = _get_anchor(s_id)
    
    # Serialize history
    history = batch.history_traj.tolist() # (N, T_hist, 6)
    preds = cache["predictions"].tolist()  # (N, K, T_fut, 5)
    confs = cache["confidences"].tolist()  # (N, K)
    
    # Map elements
    map_polylines = []
    for poly in batch.map_polylines:
        map_polylines.append(poly.tolist())

    # Convert predictions to geographic coordinates for map overlay
    geo_predictions = []  # (N, K, T, [lng, lat])
    preds_np = cache["predictions"]
    for n in range(preds_np.shape[0]):
        agent_modes = []
        for k in range(preds_np.shape[1]):
            mode_coords = []
            for t in range(preds_np.shape[2]):
                lng, lat = _metres_to_geo(
                    float(preds_np[n, k, t, 0]),
                    float(preds_np[n, k, t, 1]),
                    anchor["lat"], anchor["lng"]
                )
                mode_coords.append([lng, lat])
            agent_modes.append(mode_coords)
        geo_predictions.append(agent_modes)

    # Convert agent current positions to geo (last history point)
    agent_positions = []
    for n in range(batch.history_traj.shape[0]):
        last_pt = batch.history_traj[n, -1]
        lng, lat = _metres_to_geo(
            float(last_pt[0]), float(last_pt[1]),
            anchor["lat"], anchor["lng"]
        )
        agent_positions.append({"lng": lng, "lat": lat, "type": batch.agent_types[n]})

    return {
        "scenario_id": s_id,
        "agent_types": batch.agent_types,
        "history": history,
        "predictions": preds,
        "confidences": confs,
        "map_polylines": map_polylines,
        "ego_origin": {"lat": anchor["lat"], "lng": anchor["lng"], "city": anchor["city"]},
        "geo_predictions": geo_predictions,
        "agent_positions": agent_positions
    }

@app.get("/audio/{s_id}")
def get_audio(s_id: str):
    if s_id not in cached_scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found")
        
    cache = cached_scenarios[s_id]
    batch = cache["batch"]
    preds = cache["predictions"]
    confs = cache["confidences"]
    
    # Use output engine to write audio file
    routes, text, audio_path = route_engine.build_and_score_routes(
        batch.sdc_route_graph, preds[0], confs[0], s_id
    )
    
    return FileResponse(audio_path, media_type="audio/mp3")

@app.get("/map/{s_id}")
def get_map_png(s_id: str):
    if s_id not in cached_scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found")
        
    cache = cached_scenarios[s_id]
    batch = cache["batch"]
    preds = cache["predictions"]
    confs = cache["confidences"]
    
    # Re-package map data for renderer
    lanes = []
    crosswalks = []
    for poly in batch.map_polylines:
        # Reconstruct shapely lines
        coords = poly[:, :2]
        if poly[0, 2] == 1.0:
            lanes.append(LineString(coords))
        else:
            crosswalks.append(Polygon(coords))
            
    map_data = {
        "lanes": lanes,
        "crosswalks": crosswalks,
        "drivable_area": Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])
    }
    
    png_path, _, _ = map_renderer.render_map((0.0, 0.0), map_data, preds, confs, s_id)
    return FileResponse(png_path, media_type="image/png")

@app.get("/motion_states/{s_id}")
def get_motion_states(s_id: str):
    if s_id not in cached_scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found")
        
    cache = cached_scenarios[s_id]
    batch = cache["batch"]
    preds = cache["predictions"]
    confs = cache["confidences"]
    
    states, _, _ = explainer.analyze_motion_states(
        preds, confs, batch.history_traj, batch.agent_types, s_id
    )
    return states

@app.post("/run_hcp/{s_id}")
def run_hcp(s_id: str):
    if s_id not in cached_scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found")
        
    cache = cached_scenarios[s_id]
    batch = cache["batch"]
    preds = cache["predictions"]
    
    # Convert predictions to torch tensor
    preds_tensor = torch.tensor(preds, dtype=torch.float32)
    # Smooth before feeding to the pruner — see smooth_trajectory_xy_polyfit's
    # docstring for why this matters: raw model output has small realistic
    # point-to-point noise that KFF's jerk check (very sensitive to noise,
    # due to triple-differencing) would otherwise reject even for genuinely
    # sensible paths.
    preds_tensor = smooth_trajectory_xy_polyfit(preds_tensor)
    hist_tensor = torch.tensor(batch.history_traj, dtype=torch.float32)
    
    # Run pruner
    _, _, stats = pruner(preds_tensor, hist_tensor, batch.map_polylines)
    return stats

class InjectAgentRequest(BaseModel):
    x: float
    y: float
    heading_deg: float
    speed_mps: float
    agent_type: str = "vehicle"

# Human-readable labels for generate_kinematic_candidates' fixed turn-rate
# bank ([0.0, 0.12, -0.12, 0.25, -0.25, 0.40] rad/s), for display only.
_KINEMATIC_CANDIDATE_LABELS = [
    "Straight", "Slight Left", "Slight Right",
    "Moderate Left", "Moderate Right", "Sharp Left",
]

@app.post("/inject_agent/{s_id}")
def inject_agent(s_id: str, req: InjectAgentRequest):
    """
    Adds a user-placed synthetic agent and predicts its future using
    physics-based extrapolation (generate_kinematic_candidates) — NOT the
    trained MTR transformer, which was trained on this scenario's fixed
    real-agent batch and can't have a new agent spliced into it without a
    full batch rebuild. What this DOES do honestly: the synthetic agent's
    6 candidate paths are run through the real, unmodified HCP pruner
    (KFF/SRF/SCF) together with this scenario's real map geometry and real
    other agents, so its feasibility/collision checks are genuine, even
    though its trajectory prediction itself is kinematic, not learned.
    """
    if s_id not in cached_scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found")

    cache = cached_scenarios[s_id]
    batch = cache["batch"]
    real_preds = cache["predictions"]          # (N, K, T_fut, 5)
    real_hist = batch.history_traj             # (N, T_hist, 6)

    T_hist = real_hist.shape[1]
    T_fut = real_preds.shape[2]
    K = real_preds.shape[1]

    heading_rad = math.radians(req.heading_deg)
    vx0 = req.speed_mps * math.cos(heading_rad)
    vy0 = req.speed_mps * math.sin(heading_rad)

    # generate_kinematic_candidates only ever reads the LAST row of hist
    # (see its docstring) — so a single real state (the click position +
    # the speed/heading you set) is all it actually needs. We repeat that
    # one row across T_hist purely to match this batch's tensor shape; it
    # is not a claim about the injected agent's real past movement.
    synth_row = torch.tensor([req.x, req.y, vx0, vy0, heading_rad, 0.0], dtype=torch.float32)
    synth_hist = synth_row.unsqueeze(0).repeat(T_hist, 1)  # (T_hist, 6)

    synth_candidates = generate_kinematic_candidates(synth_hist, T_fut=T_fut, dt=0.5, K=K)  # (K, T_fut, 5)

    # Combine with the real batch so SRF (real road geometry) and SCF (real
    # other agents) genuinely check the injected agent against everything
    # actually present in this scenario, not in isolation.
    real_preds_tensor = torch.tensor(real_preds, dtype=torch.float32)
    combined_traj = torch.cat([real_preds_tensor, synth_candidates.unsqueeze(0)], dim=0)   # (N+1, K, T_fut, 5)

    real_hist_tensor = torch.tensor(real_hist, dtype=torch.float32)
    combined_hist = torch.cat([real_hist_tensor, synth_hist.unsqueeze(0)], dim=0)          # (N+1, T_hist, 6)

    _, composite_mask, _ = pruner(combined_traj, combined_hist, batch.map_polylines)

    injected_idx = combined_traj.shape[0] - 1
    survived = composite_mask[injected_idx].tolist()  # (K,) bools — real per-candidate pruner verdict

    return {
        "agent_type": req.agent_type,
        "origin": {"x": req.x, "y": req.y, "heading_deg": req.heading_deg, "speed_mps": req.speed_mps},
        "candidates": synth_candidates[:, :, :2].tolist(),  # (K, T_fut, [x, y]) — real local ego-centric frame
        "survived": survived,
        "labels": _KINEMATIC_CANDIDATE_LABELS[:K],
    }

@app.get("/metrics")
def get_metrics():
    """Return the most recent REAL evaluation output.

    This endpoint used to fall back to a hardcoded dict (minADE5 0.81,
    32.5ms vs 115.2ms, 76% pruning). Those numbers were never measured by
    anything in this project, and serving them from a "/metrics" endpoint
    presented them as live telemetry. There is now no fallback: with no
    evaluation on disk the endpoint says so, and the dashboard renders an
    empty state rather than fiction.

    Note the glob is eval_real_*.json, not eval_*.json -- the latter also
    matched the old fabricated eval_20260722_220509.json.
    """
    files = glob.glob(os.path.join(OUTPUT_DIR, "eval_real_*.json"))
    if not files:
        return {
            "available": False,
            "detail": ("No evaluation results on disk. Run "
                       "`python hcp_project/eval/evaluate.py --checkpoint "
                       "hcp_project/outputs/mtr_checkpoint.pth --compare_hcp` "
                       "to produce them."),
        }
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        payload = json.load(f)
    return {"available": True, "source": os.path.basename(latest_file), "results": payload}

@app.get("/stream/{s_id}")
def stream_scenario(s_id: str):
    """
    SSE stream generating real-time coordinate frames at 10 Hz.
    Returns both ego-centric (x,y) and geographic (lng,lat) coords.
    """
    if s_id not in cached_scenarios:
        raise HTTPException(status_code=404, detail="Scenario not found")
        
    cache = cached_scenarios[s_id]
    batch = cache["batch"]
    preds = cache["predictions"]
    confs = cache["confidences"]
    anchor = _get_anchor(s_id)
    
    async def event_generator():
        # Iterate over timeline scrubber (0s to 6s in steps of 0.5s -> 12 steps)
        N_agents = len(batch.agent_types)
        for t_step in range(12):
            frame_data = []
            for n in range(N_agents):
                # Retrieve highest confidence path coordinates at t_step
                best_mode = int(np.argmax(confs[n]))
                state = preds[n, best_mode, t_step]
                lng, lat = _metres_to_geo(
                    float(state[0]), float(state[1]),
                    anchor["lat"], anchor["lng"]
                )
                frame_data.append({
                    "agent_id": n,
                    "type": batch.agent_types[n],
                    "x": float(state[0]),
                    "y": float(state[1]),
                    "vx": float(state[2]),
                    "vy": float(state[3]),
                    "heading": float(state[4]),
                    "lng": lng,
                    "lat": lat
                })
            
            yield f"data: {json.dumps({'step': t_step, 'agents': frame_data})}\n\n"
            await asyncio.sleep(0.5)  # 2 Hz for visible animation
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ── Serve React build if available, otherwise inline HTML ──
UI_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "dist")

@app.on_event("startup")
def mount_spa():
    if os.path.isdir(UI_DIST):
        app.mount("/assets", StaticFiles(directory=os.path.join(UI_DIST, "assets")), name="spa-assets")

@app.get("/app")
def serve_spa():
    index = os.path.join(UI_DIST, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return Response(content="React build not found. Run: cd ui && npm run build", status_code=404)

# Direct HTML landing page — fully rewritten dashboard
@app.get("/")
def serve_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HCP + MTR Control Room</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        darkbg: '#0a0f14',
                        accent: '#1D9E75',
                        secondary: '#378ADD',
                        danger: '#D85A30',
                        neon: '#00f2fe'
                    }
                }
            }
        }
    </script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #0a0f14;
            color: #e2e8f0;
        }
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif;
        }
        .mono {
            font-family: 'JetBrains Mono', monospace;
        }
        #scene-map {
            height: 100%;
            width: 100%;
            background-color: #0d131a;
        }
        .custom-scrollbar::-webkit-scrollbar {
            width: 5px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
            background: #0f172a;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 3px;
        }
        /* Glassmorphism card base */
        .glass-card {
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(51, 65, 85, 0.5);
            border-radius: 0.75rem;
        }
        /* Pulse animation for live dot */
        @keyframes livePulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        .live-pulse {
            animation: livePulse 1.5s ease-in-out infinite;
        }
    </style>
</head>

<body class="p-5 bg-darkbg text-slate-100 overflow-x-hidden custom-scrollbar">

    <!-- ═══════════════════ TOP HEADER ═══════════════════ -->
    <header class="flex items-center justify-between pb-5 mb-5 border-b border-slate-800/60">
        <div>
            <h1 class="text-2xl font-extrabold text-accent flex items-center gap-3">
                HCP + MTR Telemetry Dashboard
                <span class="text-[10px] uppercase bg-emerald-900/30 text-accent border border-emerald-500/20 px-2.5 py-0.5 rounded-full font-bold tracking-wider">Live</span>
            </h1>
            <p class="text-slate-500 text-xs mt-1 tracking-wide">Hierarchical Combinatorial Pruning · Motion Transformer · Real-Time Control Center</p>
        </div>
        <div class="flex items-center gap-4">
            <div>
                <label class="text-[10px] text-slate-500 block mb-1 uppercase tracking-wider font-semibold">Scenario</label>
                <select id="scenario-select" onchange="loadScenario(this.value)" class="bg-slate-900/80 border border-slate-700/60 rounded-lg px-3 py-1.5 text-xs font-semibold focus:outline-none focus:border-accent mono">
                    <!-- Loaded dynamically -->
                </select>
            </div>
            <button onclick="triggerHCPRun()" class="bg-accent hover:bg-emerald-600 text-darkbg font-bold px-5 py-2 rounded-lg transition text-xs flex items-center gap-2 shadow-lg shadow-emerald-500/10">
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                Run HCP
            </button>
        </div>
    </header>

    <!-- ═══════════════════ NAVIGATION TABS (2 only) ═══════════════════ -->
    <div class="flex gap-1 mb-5 border-b border-slate-800/40 pb-2">
        <button onclick="switchTab('dashboard')" id="btn-tab-dashboard" class="px-5 py-2 border-b-2 border-accent text-accent font-semibold text-xs rounded-t-lg transition">1. Control Room BEV</button>
        <button onclick="switchTab('nlg')" id="btn-tab-nlg" class="px-5 py-2 border-b-2 border-transparent text-slate-500 hover:text-white font-semibold text-xs rounded-t-lg transition">2. State Explainer</button>
    </div>

    <!-- ═══════════════════ RECOMMENDED ACTION — computed from real, already-existing risk/TTC data ═══════════════════ -->
    <div id="recommendation-banner" class="glass-card px-5 py-3 mb-5 flex items-center gap-4 border-l-4" style="border-left-color:#10b981;">
        <span id="rec-icon" class="text-2xl">✅</span>
        <div class="flex-1">
            <div id="rec-action" class="text-sm font-extrabold tracking-wide text-accent">PROCEED — Path Clear</div>
            <div id="rec-reason" class="text-[11px] text-slate-500 mt-0.5">Run HCP or select a scenario to compute a real recommendation from current agent risk data.</div>
        </div>
    </div>

    <!-- ═══════════════════ MAIN DASHBOARD TAB ═══════════════════ -->
    <div id="tab-dashboard" class="grid grid-cols-12 gap-5 tab-content">

        <!-- ─── LEFT: Agent Intelligence Feed ─── -->
        <div class="col-span-3 glass-card p-4 flex flex-col h-[640px]">
            <h2 class="text-sm font-bold border-b border-slate-800/40 pb-2 mb-3 flex items-center justify-between">
                <span class="flex items-center gap-2">
                    <span class="w-1.5 h-1.5 rounded-full bg-accent live-pulse"></span>
                    Agent Intelligence Feed
                </span>
                <span id="agent-count" class="text-[10px] bg-slate-800/60 px-2 py-0.5 rounded-full text-slate-400 mono">0 active</span>
            </h2>
            <div id="agent-list" class="flex-1 overflow-y-auto custom-scrollbar space-y-2.5 pr-1">
                <!-- Loaded dynamically -->
            </div>
        </div>

        <!-- ─── CENTER: Live HD Map ─── -->
        <div class="col-span-6 glass-card p-4 flex flex-col h-[640px]">
            <h2 class="text-sm font-bold border-b border-slate-800/40 pb-2 mb-3 flex items-center justify-between">
                <span>Ego-Centric BEV Crop (500m)</span>
                <span class="text-[10px] text-accent font-semibold flex items-center gap-1.5 mono">
                    <span class="w-1.5 h-1.5 rounded-full bg-accent live-pulse"></span> 10Hz Feed
                </span>
            </h2>
            <div class="relative flex-1 rounded-lg overflow-hidden border border-slate-800/40">
                <div id="scene-map"></div>
                <!-- HUD Status Overlay Banner -->
                <div class="absolute top-0 left-0 right-0 z-[1000] flex items-center justify-center pointer-events-none">
                    <div class="mt-3 px-5 py-1.5 bg-slate-950/75 backdrop-blur-lg border border-cyan-500/15 rounded-full shadow-lg shadow-cyan-500/5">
                        <span class="text-[10px] mono font-bold text-cyan-400 tracking-[0.2em] uppercase live-pulse">🛰️ REAL MAP GEOMETRY · EGO-CENTRIC FRAME</span>
                    </div>
                </div>
                <!-- Map Legend — colors = agent type, line style = data type -->
                <div class="absolute bottom-3 left-3 z-[1000] pointer-events-none glass-card px-3 py-2 text-[9px] mono space-y-1.5">
                    <div class="flex items-center gap-2">
                        <span class="inline-block w-3 h-0.5" style="background:#10b981"></span>
                        <span class="text-slate-400">Ego Vehicle</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="inline-block w-3 h-0.5" style="background:#f59e0b"></span>
                        <span class="text-slate-400">Other Vehicle</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="inline-block w-3 h-0.5" style="background:#38bdf8"></span>
                        <span class="text-slate-400">Pedestrian</span>
                    </div>
                    <div class="border-t border-slate-800/40 my-1"></div>
                    <div class="flex items-center gap-2">
                        <svg width="16" height="4"><line x1="0" y1="2" x2="16" y2="2" stroke="#94a3b8" stroke-width="1" stroke-dasharray="1,1.5"/></svg>
                        <span class="text-slate-500">Real history</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <svg width="16" height="4"><line x1="0" y1="2" x2="16" y2="2" stroke="#94a3b8" stroke-width="2"/></svg>
                        <span class="text-slate-500">Top prediction</span>
                    </div>
                    <div class="flex items-center gap-2">
                        <svg width="16" height="4"><line x1="0" y1="2" x2="16" y2="2" stroke="#94a3b8" stroke-width="1" stroke-dasharray="2.5,2"/></svg>
                        <span class="text-slate-500">Alt. prediction</span>
                    </div>
                </div>
            </div>
            <!-- Control Bar -->
            <div class="flex items-center justify-between mt-3">
                <div class="flex items-center gap-2">
                    <button onclick="togglePlayback()" id="btn-play" class="glass-card px-3 py-1.5 text-xs font-bold hover:border-accent/40 transition">Play Stream</button>
                    <button onclick="resetPlayback()" class="glass-card px-3 py-1.5 text-xs font-bold hover:border-accent/40 transition">Reset</button>
                    <button onclick="toggleInjectMode()" id="btn-add-agent" class="glass-card px-3 py-1.5 text-xs font-bold hover:border-accent/40 transition">+ Add Agent</button>
                    <button onclick="clearInjectedAgents()" class="glass-card px-3 py-1.5 text-xs font-bold text-slate-500 hover:border-red-500/40 hover:text-red-400 transition">Clear Injected</button>
                </div>
                <div class="flex items-center gap-2 text-[10px] mono text-slate-500">
                    <span>Frame:</span>
                    <span id="frame-counter" class="text-accent font-bold">0 / 12</span>
                </div>
            </div>
        </div>

        <!-- ─── RIGHT: HCP Pruning Waterfall + Telemetry (PRESERVED) ─── -->
        <div class="col-span-3 glass-card p-4 flex flex-col h-[640px] justify-between">
            <div>
                <h2 class="text-sm font-bold border-b border-slate-800/40 pb-2 mb-4">HCP Pruning Cascade</h2>
                <div class="space-y-3.5">
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="text-slate-400">Raw Candidates</span>
                            <span id="raw-stat" class="mono font-bold text-slate-300">128 (100%)</span>
                        </div>
                        <div class="w-full bg-slate-900/60 h-3 rounded-full overflow-hidden">
                            <div id="raw-bar" class="bg-slate-500 h-full w-[100%]"></div>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="text-slate-400 flex items-center gap-1.5">
                                <span class="w-2 h-2 bg-slate-500 rounded-full"></span> Stage 1: KFF (Kinematic)
                            </span>
                            <span id="kff-stat" class="mono font-bold text-slate-300">—</span>
                        </div>
                        <div class="w-full bg-slate-900/60 h-3 rounded-full overflow-hidden">
                            <div id="kff-bar" class="bg-slate-500 h-full w-[58%] transition-all duration-500"></div>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="text-slate-400 flex items-center gap-1.5">
                                <span class="w-2 h-2 bg-secondary rounded-full"></span> Stage 2: SRF (Spatial)
                            </span>
                            <span id="srf-stat" class="mono font-bold text-slate-300">—</span>
                        </div>
                        <div class="w-full bg-slate-900/60 h-3 rounded-full overflow-hidden">
                            <div id="srf-bar" class="bg-secondary h-full w-[24%] transition-all duration-500"></div>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="text-slate-400 flex items-center gap-1.5">
                                <span class="w-2 h-2 bg-accent rounded-full"></span> Stage 3: SCF (Social)
                            </span>
                            <span id="scf-stat" class="mono font-bold text-accent">—</span>
                        </div>
                        <div class="w-full bg-slate-900/60 h-3 rounded-full overflow-hidden">
                            <div id="scf-bar" class="bg-accent h-full w-[7%] transition-all duration-500"></div>
                        </div>
                    </div>
                </div>

                <!-- Latency Dial -->
                <div class="mt-5 flex flex-col items-center glass-card p-4">
                    <span class="text-[9px] text-slate-500 uppercase font-bold tracking-widest mb-2">Pruner Stage Time</span>
                    <div class="relative flex items-center justify-center w-24 h-24">
                        <svg class="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
                            <path class="text-slate-800" stroke-width="3" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                            <path id="dial-value" class="text-accent transition-all duration-500" stroke-dasharray="80, 100" stroke-width="3" stroke-linecap="round" stroke="currentColor" fill="none" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                        </svg>
                        <div class="absolute flex flex-col items-center">
                            <span id="latency-ms" class="text-xl font-black mono text-white">—</span>
                            <span class="text-[8px] uppercase tracking-widest text-slate-500 font-semibold">Not measured</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Telemetry Summary Stats -->
            <div class="grid grid-cols-2 gap-2.5 border-t border-slate-800/40 pt-3">
                <div class="glass-card p-2.5 text-center">
                    <span class="text-[9px] text-slate-500 uppercase block mb-0.5 tracking-wider font-semibold">Pruning Ratio</span>
                    <span id="pruning-ratio" class="text-base font-black text-accent mono">—</span>
                </div>
                <div class="glass-card p-2.5 text-center">
                    <span class="text-[9px] text-slate-500 uppercase block mb-0.5 tracking-wider font-semibold">Pruner Overhead</span>
                    <span id="latency-saved" class="text-base font-black text-secondary mono">—</span>
                </div>
            </div>
        </div>
    </div>

    <!-- ═══════════════════ STATE EXPLAINER TAB ═══════════════════ -->
    <div id="tab-nlg" class="glass-card p-5 tab-content hidden">
        <h2 class="text-sm font-bold border-b border-slate-800/40 pb-2 mb-4">Velocity & Direction Motion State Explainer</h2>
        <div class="grid grid-cols-12 gap-5">
            <div class="col-span-8 overflow-x-auto">
                <table class="w-full text-left text-xs">
                    <thead class="bg-slate-900/50 text-slate-400 font-semibold border-b border-slate-800/40">
                        <tr>
                            <th class="p-3">Agent</th>
                            <th class="p-3">Type</th>
                            <th class="p-3">Speed</th>
                            <th class="p-3">Heading</th>
                            <th class="p-3">TTC (s)</th>
                            <th class="p-3">Risk</th>
                            <th class="p-3">NLG Explanation</th>
                        </tr>
                    </thead>
                    <tbody id="explainer-table-body" class="divide-y divide-slate-800/30">
                        <!-- Loaded dynamically -->
                    </tbody>
                </table>
            </div>
            <div class="col-span-4 glass-card p-4 flex flex-col items-center">
                <h3 class="text-xs font-bold text-slate-400 mb-3 self-start">Direction Field Mapping</h3>
                <div id="direction-field-box" class="w-full aspect-square border border-slate-800/40 bg-[#0a0f14] rounded-lg overflow-hidden relative flex items-center justify-center">
                    <img id="direction-field-img" class="w-full h-full object-contain" src="" alt="Direction Field" />
                </div>
            </div>
        </div>
    </div>

    <!-- ═══════════════════ SCRIPTING ═══════════════════ -->
    <script>
        // Real local-geometry scene renderer (replaces the earlier Leaflet
        // tile-map approach). nuScenes never publishes true GPS for a
        // scenario — only positions in each map's own local coordinate
        // system — so any attempt to drop that data onto a real-world tile
        // basemap could only ever be an approximate guess. Rendering
        // everything (lanes, crosswalks, agent history, predictions) in
        // that same local ego-centric frame instead means alignment is
        // always exact, never approximate, and there's no external tile
        // fetch left to fail or watermark.
        let SceneSVG = null;
        let SceneAgentMarkers = {};   // agent_id -> SVG <circle>, so streamed frames move it instead of re-creating it
        let currentScenarioData = null;   // last-fetched /scenario response, kept so Reset can restore original positions
        let sseSource = null;
        let currentScenarioId = "";
        let injectedAgents = [];   // {x, y, heading_deg, speed_mps, agent_type, result} — persists across redraws for THIS scenario only
        let injectMode = false;

        // ── Init ──
        window.addEventListener('load', async () => {
            try {
                const res = await fetch('/scenarios');
                const scenarios = await res.json();
                const select = document.getElementById('scenario-select');
                if (select && Array.isArray(scenarios)) {
                    scenarios.forEach(s_id => {
                        const opt = document.createElement('option');
                        opt.value = s_id;
                        opt.textContent = s_id;
                        select.appendChild(opt);
                    });
                    if (scenarios.length > 0) {
                        loadScenario(scenarios[0]);
                    }
                }
            } catch (err) {
                console.warn('Failed to load scenarios:', err);
            }
        });

        // ── Tab Switching ──
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el?.classList.add('hidden'));
            const target = document.getElementById(`tab-${tabName}`);
            if (target) target.classList.remove('hidden');

            const tabs = ['dashboard', 'nlg'];
            tabs.forEach(t => {
                const btn = document.getElementById(`btn-tab-${t}`);
                if (!btn) return;
                if (t === tabName) {
                    btn.classList.add('border-accent', 'text-accent');
                    btn.classList.remove('border-transparent', 'text-slate-500');
                } else {
                    btn.classList.remove('border-accent', 'text-accent');
                    btn.classList.add('border-transparent', 'text-slate-500');
                }
            });

        }

        // ── Load Scenario ──
        async function loadScenario(s_id) {
            currentScenarioId = s_id;
            if (sseSource) { sseSource.close(); sseSource = null; }
            const btnPlay = document.getElementById('btn-play');
            if (btnPlay) btnPlay.textContent = "Play Stream";
            const frameCounter = document.getElementById('frame-counter');
            if (frameCounter) frameCounter.textContent = "0 / 12";
            // Injected agents' coordinates are only meaningful in the real
            // scenario they were placed in — a different scenario means a
            // different real road network, so they're cleared here rather
            // than carried over into a frame where they'd be meaningless.
            injectedAgents = [];
            setInjectMode(false);

            try {
                const res = await fetch(`/scenario/${s_id}`);
                const data = await res.json();
                currentScenarioData = data;
                renderSceneMap(data);
                loadAgentFeed(data);
                loadNLGState(s_id);
            } catch (err) {
                console.warn('Scenario load failed:', err);
            }
        }

        // ── Scene renderer — real local map geometry, real agent history,
        // real predicted trajectories, all in the dataset's own
        // ego-centric metre frame. No tile basemap, no GPS approximation:
        // everything here comes from the same coordinate system the model
        // itself reasons in, so alignment between roads/agents/predictions
        // is always exact. ──
        function renderSceneMap(data) {
            const container = document.getElementById('scene-map');
            if (!container) return;

            // Fit the view to the AGENTS' real extent (history +
            // predictions), not the full map's lane network — a scenario
            // where agents barely move (e.g. ego at 0.1 m/s) would
            // otherwise get crushed to an invisible speck against a map
            // that spans hundreds of metres. The lane/crosswalk geometry
            // still draws in full; anything outside this cropped view is
            // just naturally clipped by the SVG viewport, the same way a
            // real BEV crop would be.
            const agentPts = [];
            (data?.history || []).forEach(hist => {
                (hist || []).forEach(pt => { if (pt && pt.length >= 2) agentPts.push([pt[0], pt[1]]); });
            });
            (data?.predictions || []).forEach(agentModes => {
                (agentModes || []).forEach(mode => {
                    (mode || []).forEach(pt => { if (pt && pt.length >= 2) agentPts.push([pt[0], pt[1]]); });
                });
            });
            if (agentPts.length === 0) return;

            const xs = agentPts.map(p => p[0]);
            const ys = agentPts.map(p => p[1]);
            const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
            const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
            // Half-extent of the real agent movement, with a floor so a
            // near-stationary scenario still shows meaningful road context
            // around it rather than zooming in on nothing.
            const rawHalfExtent = Math.max(
                Math.max(...xs) - Math.min(...xs),
                Math.max(...ys) - Math.min(...ys)
            ) / 2;
            const halfExtent = Math.max(rawHalfExtent, 20); // metres — floor for a sensible local crop
            const pad = halfExtent * 0.3;
            const vbMinX = centerX - halfExtent - pad;
            const vbMinY = -(centerY + halfExtent + pad);   // flip Y so "forward" reads as "up" on screen
            const vbWidth = (halfExtent + pad) * 2;
            const vbHeight = (halfExtent + pad) * 2;
            const toSvgY = (y) => -y;

            const svgns = 'http://www.w3.org/2000/svg';

            if (!SceneSVG) {
                container.innerHTML = '';
                SceneSVG = document.createElementNS(svgns, 'svg');
                SceneSVG.setAttribute('width', '100%');
                SceneSVG.setAttribute('height', '100%');
                SceneSVG.style.background = '#0d131a';
                container.appendChild(SceneSVG);
            }
            SceneSVG.setAttribute('viewBox', `${vbMinX} ${vbMinY} ${vbWidth} ${vbHeight}`);
            SceneSVG.innerHTML = '';   // clear the previous scenario's content
            SceneAgentMarkers = {};

            const addEl = (tag, attrs) => {
                const el = document.createElementNS(svgns, tag);
                Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
                SceneSVG.appendChild(el);
                return el;
            };

            // 1. Real lane centerlines + crosswalks — straight from the
            //    dataset's own local map geometry (map_polylines), so they
            //    share the exact frame everything else is drawn in.
            (data?.map_polylines || []).forEach(poly => {
                if (!poly || poly.length === 0) return;
                const isLane = poly[0][2] === 1.0;
                const pts = poly.map(pt => `${pt[0]},${toSvgY(pt[1])}`).join(' ');
                if (isLane) {
                    addEl('polyline', {
                        points: pts, fill: 'none', stroke: '#475569',
                        'stroke-width': 0.3,
                    });
                } else {
                    addEl('polygon', {
                        points: pts, fill: '#0ea5e930', stroke: '#0ea5e9', 'stroke-width': 0.2,
                    });
                }
            });

            // 2. Each agent's real recent history — dotted line (distinct
            //    from prediction line styles, see legend) + current
            //    position dot. Hoverable tooltip confirms which agent is
            //    which regardless of color ambiguity at a glance.
            (data?.history || []).forEach((hist, n) => {
                if (!hist || hist.length === 0) return;
                const isEgo = n === 0;
                const agentType = data?.agent_types?.[n] || 'vehicle';
                const color = isEgo ? '#10b981' : (agentType === 'pedestrian' ? '#38bdf8' : '#f59e0b');
                const pts = hist.map(pt => `${pt[0]},${toSvgY(pt[1])}`).join(' ');
                const histLine = addEl('polyline', {
                    points: pts, fill: 'none', stroke: color, 'stroke-width': 0.35,
                    opacity: 0.7, 'stroke-linecap': 'round', 'stroke-dasharray': '0.15,0.45',
                });
                const histTitle = document.createElementNS(svgns, 'title');
                histTitle.textContent = `${isEgo ? 'Ego Vehicle' : 'Agent #' + n} · real history`;
                histLine.appendChild(histTitle);

                const last = hist[hist.length - 1];
                const marker = addEl('circle', {
                    cx: last[0], cy: toSvgY(last[1]), r: isEgo ? 1.4 : 1.1,
                    fill: color, stroke: '#0d131a', 'stroke-width': 0.2,
                });
                const markerTitle = document.createElementNS(svgns, 'title');
                markerTitle.textContent = isEgo ? 'Ego Vehicle' : `Agent #${n} (${agentType})`;
                marker.appendChild(markerTitle);
                SceneAgentMarkers[n] = marker;
            });

            // 3. Real predicted trajectories — top-confidence mode
            //    solid/thick, other candidate modes dashed (longer dashes
            //    than the dotted history line above, so the two never look
            //    alike), showing genuine model uncertainty across multiple
            //    possible futures.
            const preds = data?.predictions || [];
            const confs = data?.confidences || [];
            preds.forEach((agentModes, n) => {
                if (!Array.isArray(agentModes) || agentModes.length === 0) return;
                const isEgo = n === 0;
                const agentType = data?.agent_types?.[n] || 'vehicle';
                const baseColor = isEgo ? '#10b981' : (agentType === 'pedestrian' ? '#38bdf8' : '#f59e0b');
                const agentConfs = confs[n] || [];
                const bestIdx = agentConfs.length > 0 ? agentConfs.indexOf(Math.max(...agentConfs)) : 0;

                agentModes.forEach((mode, k) => {
                    if (!Array.isArray(mode) || mode.length === 0) return;
                    const isBest = k === bestIdx;
                    const pts = mode.map(pt => `${pt[0]},${toSvgY(pt[1])}`).join(' ');
                    const line = addEl('polyline', {
                        points: pts, fill: 'none', stroke: baseColor,
                        'stroke-width': isBest ? 0.6 : 0.2,
                        opacity: isBest ? 0.95 : 0.4,
                    });
                    if (!isBest) line.setAttribute('stroke-dasharray', '1.5,1.2');
                    if (isBest) {
                        const confPct = ((agentConfs[k] || 0) * 100).toFixed(0);
                        const title = document.createElementNS(svgns, 'title');
                        title.textContent = `${isEgo ? 'Ego' : 'Agent #' + n} · most likely path (${confPct}% confidence)`;
                        line.appendChild(title);
                    }
                });
            });

            // 4. Re-draw any injected (synthetic) agents so they persist
            //    across redraws of this scenario (e.g. after Reset).
            injectedAgents.forEach(entry => drawInjectedAgent(entry));
        }

        // ── Add Agent — click-to-place synthetic agent, predicted via real
        // physics-based extrapolation + the real HCP pruner (see
        // /inject_agent/{s_id} in main.py for what this actually runs —
        // NOT the trained transformer). ──
        function setInjectMode(on) {
            injectMode = on;
            const btn = document.getElementById('btn-add-agent');
            const mapEl = document.getElementById('scene-map');
            if (btn) {
                btn.textContent = on ? 'Click map to place…' : '+ Add Agent';
                btn.classList.toggle('border-accent', on);
                btn.classList.toggle('text-accent', on);
            }
            if (mapEl) mapEl.style.cursor = on ? 'crosshair' : 'default';
        }
        function toggleInjectMode() { setInjectMode(!injectMode); }

        // Click handler — converts a screen click into this scenario's real
        // local (x, y) using the SVG's own coordinate transform, so it's
        // exact regardless of zoom/pan/container size.
        document.addEventListener('DOMContentLoaded', () => {
            const mapEl = document.getElementById('scene-map');
            if (!mapEl) return;
            mapEl.addEventListener('click', (evt) => {
                if (!injectMode || !SceneSVG) return;
                const pt = SceneSVG.createSVGPoint();
                pt.x = evt.clientX;
                pt.y = evt.clientY;
                const svgPt = pt.matrixTransform(SceneSVG.getScreenCTM().inverse());
                const realX = svgPt.x;
                const realY = -svgPt.y;   // undo the Y-flip used for rendering
                openInjectForm(realX, realY, evt.clientX, evt.clientY);
                setInjectMode(false);
            });
        });

        // Small floating form for speed/heading/type — appears at the click
        // point, real x/y already captured from the click itself.
        function openInjectForm(realX, realY, clientX, clientY) {
            const existing = document.getElementById('inject-form');
            if (existing) existing.remove();

            const form = document.createElement('div');
            form.id = 'inject-form';
            form.className = 'glass-card p-3 text-xs space-y-2';
            form.style.position = 'fixed';
            form.style.left = `${Math.min(clientX + 10, window.innerWidth - 220)}px`;
            form.style.top = `${Math.min(clientY + 10, window.innerHeight - 220)}px`;
            form.style.zIndex = 2000;
            form.style.width = '200px';
            form.innerHTML = `
                <div class="font-bold text-slate-300 mb-1">New Agent</div>
                <label class="block text-slate-500">Type</label>
                <select id="inj-type" class="w-full bg-slate-900/80 border border-slate-700/60 rounded px-2 py-1 mono text-[11px]">
                    <option value="vehicle">Vehicle</option>
                    <option value="pedestrian">Pedestrian</option>
                </select>
                <label class="block text-slate-500 mt-1">Speed (m/s)</label>
                <input id="inj-speed" type="number" value="5" step="0.5" min="0" class="w-full bg-slate-900/80 border border-slate-700/60 rounded px-2 py-1 mono text-[11px]" />
                <label class="block text-slate-500 mt-1">Heading (°, 0 = +x)</label>
                <input id="inj-heading" type="number" value="0" step="15" class="w-full bg-slate-900/80 border border-slate-700/60 rounded px-2 py-1 mono text-[11px]" />
                <div class="flex gap-2 mt-2">
                    <button id="inj-confirm" class="flex-1 bg-accent hover:bg-emerald-600 text-darkbg font-bold px-2 py-1.5 rounded text-[11px]">Add</button>
                    <button id="inj-cancel" class="flex-1 glass-card px-2 py-1.5 text-[11px] hover:border-red-500/40">Cancel</button>
                </div>
            `;
            document.body.appendChild(form);

            document.getElementById('inj-cancel').onclick = () => form.remove();
            document.getElementById('inj-confirm').onclick = async () => {
                const agent_type = document.getElementById('inj-type').value;
                const speed_mps = parseFloat(document.getElementById('inj-speed').value) || 0;
                const heading_deg = parseFloat(document.getElementById('inj-heading').value) || 0;
                form.remove();
                await confirmInjectAgent(realX, realY, heading_deg, speed_mps, agent_type);
            };
        }

        async function confirmInjectAgent(x, y, heading_deg, speed_mps, agent_type) {
            try {
                const res = await fetch(`/inject_agent/${currentScenarioId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ x, y, heading_deg, speed_mps, agent_type }),
                });
                if (!res.ok) {
                    console.warn('Agent injection failed:', await res.text());
                    return;
                }
                const result = await res.json();
                const entry = { x, y, heading_deg, speed_mps, agent_type, result };
                injectedAgents.push(entry);
                drawInjectedAgent(entry);
            } catch (err) {
                console.warn('Agent injection request failed:', err);
            }
        }

        // Draws one injected agent: a distinct dashed-outline marker (never
        // styled like a real agent's solid marker) + its real per-candidate
        // pruner verdict — feasible candidates solid, rejected candidates
        // thin/red, so the pruner's actual decision is visible, not just
        // asserted.
        function drawInjectedAgent(entry) {
            if (!SceneSVG) return;
            const svgns = 'http://www.w3.org/2000/svg';
            const toSvgY = (y) => -y;
            const addEl = (tag, attrs) => {
                const el = document.createElementNS(svgns, tag);
                Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
                SceneSVG.appendChild(el);
                return el;
            };

            const color = entry.agent_type === 'pedestrian' ? '#38bdf8' : '#f59e0b';
            const r = entry.result;

            if (r && Array.isArray(r.candidates)) {
                r.candidates.forEach((path, k) => {
                    const feasible = r.survived ? r.survived[k] : true;
                    const pts = path.map(([px, py]) => `${px},${toSvgY(py)}`).join(' ');
                    const line = addEl('polyline', {
                        points: pts, fill: 'none',
                        stroke: feasible ? color : '#ef4444',
                        'stroke-width': feasible ? 0.5 : 0.2,
                        opacity: feasible ? 0.9 : 0.4,
                    });
                    if (!feasible) line.setAttribute('stroke-dasharray', '1.5,1.2');
                    const title = document.createElementNS(svgns, 'title');
                    const label = (r.labels && r.labels[k]) || `Candidate ${k}`;
                    title.textContent = `Injected agent · ${label} · ${feasible ? 'FEASIBLE (real HCP verdict)' : 'REJECTED by HCP'}`;
                    line.appendChild(title);
                });
            }

            // Distinct dashed-outline marker (never a solid dot like real
            // agents) so an injected agent can never be visually confused
            // with a real, dataset-sourced one.
            const marker = addEl('circle', {
                cx: entry.x, cy: toSvgY(entry.y), r: 1.3,
                fill: 'none', stroke: color, 'stroke-width': 0.4, 'stroke-dasharray': '0.4,0.3',
            });
            const markerTitle = document.createElementNS(svgns, 'title');
            markerTitle.textContent = `Injected ${entry.agent_type} · ${entry.speed_mps} m/s @ ${entry.heading_deg}°`;
            marker.appendChild(markerTitle);
        }

        function clearInjectedAgents() {
            injectedAgents = [];
            if (currentScenarioData) renderSceneMap(currentScenarioData);
        }

        // ── Agent Intelligence Feed ──
        function loadAgentFeed(data) {
            const list = document.getElementById('agent-list');
            if (!list) return;
            list.innerHTML = "";
            const countEl = document.getElementById('agent-count');
            if (countEl && data?.history) {
                countEl.textContent = `${data.history.length} active`;
            }

            (data?.history || []).forEach((hist, n) => {
                if (!hist || hist.length === 0) return;
                const lastPt = hist[hist.length - 1];
                const vx = lastPt?.[2] || 0;
                const vy = lastPt?.[3] || 0;
                const speed = Math.sqrt(vx * vx + vy * vy).toFixed(1);
                const type = data?.agent_types?.[n] || 'unknown';
                const isEgo = n === 0;

                const card = document.createElement('div');
                card.className = `p-3 rounded-lg border text-xs flex justify-between items-center transition duration-200 hover:border-accent/40 ${
                    isEgo ? 'bg-emerald-950/15 border-emerald-900/40' : 'bg-slate-900/30 border-slate-800/40'
                }`;

                const riskColor = isEgo ? 'text-accent' : 'text-slate-300';
                card.innerHTML = `
                    <div>
                        <div class="font-bold flex items-center gap-1.5">
                            <span class="${riskColor}">${isEgo ? 'Ego Vehicle' : 'Agent #' + n}</span>
                            <span class="text-[8px] uppercase bg-slate-800/60 px-1.5 py-0.5 rounded text-slate-500 mono">${type}</span>
                        </div>
                        <div class="text-[10px] text-slate-500 mt-1">Speed: <span class="mono text-slate-300">${speed} m/s</span></div>
                    </div>
                    ${isEgo
                        ? '<span class="text-accent text-[9px] border border-accent/30 bg-accent/5 px-2 py-0.5 rounded-full font-bold mono">SDC</span>'
                        : '<span class="text-secondary text-[9px] border border-sky-900/30 bg-sky-950/10 px-2 py-0.5 rounded-full mono font-bold">#' + n + '</span>'
                    }
                `;
                list.appendChild(card);
            });
        }

        // ── NLG State Explainer ──
        async function loadNLGState(s_id) {
            try {
                const res = await fetch(`/motion_states/${s_id}`);
                const states = await res.json();
                const tbody = document.getElementById('explainer-table-body');
                if (!tbody || !Array.isArray(states)) return;
                tbody.innerHTML = "";

                states.forEach(s => {
                    if (!s) return;
                    const tr = document.createElement('tr');
                    tr.className = "hover:bg-slate-900/20 text-xs";

                    const riskBadge = s.risk_level === 'high'
                        ? '<span class="px-2 py-0.5 rounded bg-red-950/30 text-red-400 border border-red-500/20 font-bold">HIGH</span>'
                        : (s.risk_level === 'medium'
                            ? '<span class="px-2 py-0.5 rounded bg-orange-950/30 text-orange-400 border border-orange-500/20 font-bold">MED</span>'
                            : '<span class="px-2 py-0.5 rounded bg-emerald-950/20 text-emerald-400 border border-emerald-500/20 font-bold">LOW</span>');

                    tr.innerHTML = `
                        <td class="p-3 font-bold mono">#${s.agent_id ?? 'N/A'}</td>
                        <td class="p-3 uppercase text-[10px] text-slate-500">${s.category ?? ''}</td>
                        <td class="p-3 mono">${(s.speed_mps ?? 0).toFixed(1)} m/s</td>
                        <td class="p-3 mono">${(s.heading_deg ?? 0).toFixed(0)}°</td>
                        <td class="p-3 mono">${(s.ttc_seconds ?? -1) > 0 ? (s.ttc_seconds).toFixed(1) + 's' : 'N/A'}</td>
                        <td class="p-3">${riskBadge}</td>
                        <td class="p-3 text-slate-400 italic">${s.explanation ?? ''}</td>
                    `;
                    tbody.appendChild(tr);
                });

                updateRecommendationBanner(states);

                const dirImg = document.getElementById('direction-field-img');
                if (dirImg) dirImg.src = `/map/${s_id}`;
            } catch (err) {
                console.warn('NLG load failed:', err);
            }
        }

        // ── Recommended Action — a real decision synthesized from the
        // model's own already-computed risk_level/ttc_seconds per agent
        // (the same data driving the State Explainer table above). Not a
        // new signal, not fabricated — just the single highest-priority
        // real finding surfaced where it's actually useful at a glance.
        function updateRecommendationBanner(states) {
            const iconEl = document.getElementById('rec-icon');
            const actionEl = document.getElementById('rec-action');
            const reasonEl = document.getElementById('rec-reason');
            const bannerEl = document.getElementById('recommendation-banner');
            if (!actionEl || !reasonEl || !bannerEl) return;

            const relevant = (states || []).filter(s => s && s.agent_id !== 0); // exclude ego from its own risk assessment
            const priority = { high: 2, medium: 1, low: 0 };
            let worst = null;
            relevant.forEach(s => {
                if (!worst || (priority[s.risk_level] ?? 0) > (priority[worst.risk_level] ?? 0)) {
                    worst = s;
                }
            });

            if (worst && worst.risk_level === 'high') {
                iconEl.textContent = '⛔';
                actionEl.textContent = 'YIELD / BRAKE';
                actionEl.className = 'text-sm font-extrabold tracking-wide text-red-400';
                bannerEl.style.borderLeftColor = '#ef4444';
                reasonEl.textContent = `Agent #${worst.agent_id} — ${worst.explanation || 'high collision risk'}` +
                    (worst.ttc_seconds > 0 ? ` (TTC ${worst.ttc_seconds.toFixed(1)}s)` : '');
            } else if (worst && worst.risk_level === 'medium') {
                iconEl.textContent = '⚠️';
                actionEl.textContent = 'PROCEED WITH CAUTION';
                actionEl.className = 'text-sm font-extrabold tracking-wide text-orange-400';
                bannerEl.style.borderLeftColor = '#f97316';
                reasonEl.textContent = `Agent #${worst.agent_id} — ${worst.explanation || 'moderate risk detected'}` +
                    (worst.ttc_seconds > 0 ? ` (TTC ${worst.ttc_seconds.toFixed(1)}s)` : '');
            } else {
                iconEl.textContent = '✅';
                actionEl.textContent = 'PROCEED — Path Clear';
                actionEl.className = 'text-sm font-extrabold tracking-wide text-accent';
                bannerEl.style.borderLeftColor = '#10b981';
                reasonEl.textContent = relevant.length > 0
                    ? `No agent currently at elevated risk (${relevant.length} tracked).`
                    : 'No other agents in this scenario.';
            }
        }

        // ── HCP Trigger ──
        async function triggerHCPRun() {
            try {
                const res = await fetch(`/run_hcp/${currentScenarioId}`, { method: 'POST' });
                const stats = await res.json();
                if (!stats) return;

                const update = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
                const updateWidth = (id, w) => { const el = document.getElementById(id); if (el) el.style.width = w; };

                update('raw-stat', `${stats.raw_count} (100%)`);
                updateWidth('raw-bar', `100%`);
                update('kff-stat', `${stats.kff_count} (${Math.round(stats.kff_count / stats.raw_count * 100)}%)`);
                updateWidth('kff-bar', `${Math.round(stats.kff_count / stats.raw_count * 100)}%`);
                update('srf-stat', `${stats.srf_count} (${Math.round(stats.srf_count / stats.raw_count * 100)}%)`);
                updateWidth('srf-bar', `${Math.round(stats.srf_count / stats.raw_count * 100)}%`);
                update('scf-stat', `${stats.scf_count} (${Math.round(stats.scf_count / stats.raw_count * 100)}%)`);
                updateWidth('scf-bar', `${Math.round(stats.scf_count / stats.raw_count * 100)}%`);
                update('latency-ms', `${stats.total_time_ms.toFixed(1)}ms`);
                update('pruning-ratio', `${(stats.pruning_ratio * 100).toFixed(1)}%`);
                // pruner.py deliberately stopped reporting latency_reduction_pct:
                // masking does not skip any decoder computation, so there is no
                // reduction to report. Show the pruner's own measured cost.
                update('latency-saved', `${stats.total_time_ms.toFixed(1)}ms`);

                // Success flash on map
                const mapEl = document.getElementById('scene-map');
                if (mapEl) {
                    const flash = document.createElement('div');
                    flash.className = "absolute bottom-4 right-4 z-[1001] bg-accent/90 text-darkbg font-extrabold px-4 py-2 rounded-lg text-xs shadow-lg";
                    flash.textContent = `HCP ✓ Pruned ${(stats.pruning_ratio * 100).toFixed(1)}%`;
                    mapEl.appendChild(flash);
                    setTimeout(() => flash.remove(), 2500);
                }
            } catch (err) {
                console.warn('HCP run failed:', err);
            }
        }

        // ── SSE Live Stream ──
        function togglePlayback() {
            const btn = document.getElementById('btn-play');
            if (!btn) return;
            if (btn.textContent === "Play Stream") {
                btn.textContent = "Pause Stream";
                sseSource = new EventSource(`/stream/${currentScenarioId}`);
                sseSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        const fc = document.getElementById('frame-counter');
                        if (fc) fc.textContent = `${(data?.step ?? 0) + 1} / 12`;

                        // Move each agent's real streamed position on the
                        // scene map — using the stream's real ego-centric
                        // x/y (same frame as everything else drawn here),
                        // not the earlier approximate lat/lng conversion.
                        if (SceneSVG && data && Array.isArray(data.agents)) {
                            const svgns = 'http://www.w3.org/2000/svg';
                            data.agents.forEach(agent => {
                                if (typeof agent.x !== 'number' || typeof agent.y !== 'number') return;
                                const svgY = -agent.y;
                                if (SceneAgentMarkers[agent.agent_id]) {
                                    // Marker already exists for this agent — just move it.
                                    SceneAgentMarkers[agent.agent_id].setAttribute('cx', agent.x);
                                    SceneAgentMarkers[agent.agent_id].setAttribute('cy', svgY);
                                } else {
                                    // First frame for this agent — create its marker.
                                    const isEgo = agent.agent_id === 0;
                                    const color = isEgo ? '#10b981' /* emerald, matches Ego Vehicle styling */
                                                : agent.type === 'pedestrian' ? '#38bdf8' /* sky blue */
                                                : '#f59e0b' /* amber, other vehicles */;
                                    const marker = document.createElementNS(svgns, 'circle');
                                    marker.setAttribute('cx', agent.x);
                                    marker.setAttribute('cy', svgY);
                                    marker.setAttribute('r', isEgo ? 1.4 : 1.1);
                                    marker.setAttribute('fill', color);
                                    marker.setAttribute('stroke', '#0d131a');
                                    marker.setAttribute('stroke-width', '0.2');
                                    const title = document.createElementNS(svgns, 'title');
                                    title.textContent = isEgo ? 'Ego Vehicle' : `Agent #${agent.agent_id} (${agent.type})`;
                                    marker.appendChild(title);
                                    SceneSVG.appendChild(marker);
                                    SceneAgentMarkers[agent.agent_id] = marker;
                                }
                            });
                        }
                    } catch (e) {}
                };
                sseSource.onerror = function() {
                    if (sseSource) sseSource.close();
                    sseSource = null;
                    btn.textContent = "Play Stream";
                };
            } else {
                btn.textContent = "Play Stream";
                if (sseSource) { sseSource.close(); sseSource = null; }
            }
        }

        function resetPlayback() {
            if (sseSource) { sseSource.close(); sseSource = null; }
            const btn = document.getElementById('btn-play');
            if (btn) btn.textContent = "Play Stream";
            const fc = document.getElementById('frame-counter');
            if (fc) fc.textContent = "0 / 12";
            // Re-render the static scene from the current scenario's real
            // data — restores every agent to its real starting position
            // and clears any drift left over from streamed playback.
            if (currentScenarioData) {
                renderSceneMap(currentScenarioData);
            }
        }
    </script>
</body>
</html>
"""
    return Response(content=html_content, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # NOTE: previously called the old (fake) HCPEvaluator().run_benchmarks()
    # here, which just wrote hardcoded numbers and never touched the model.
    # That class no longer exists — real evaluation results, from actually
    # running hcp_project/eval/evaluate.py, already exist as
    # eval_real_*.json files in OUTPUT_DIR, which /metrics picks up
    # automatically. No need to regenerate anything at every server startup.

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)