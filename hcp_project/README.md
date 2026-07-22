# HCP + MTR: Hierarchical Combinatorial Pruning for Multimodal Motion Transformers

This repository contains the official implementation of **Hierarchical Combinatorial Pruning (HCP)** integrated with a **Multimodal Motion Transformer (MTR)** backbone for real-time trajectory prediction, targeting acceptance at **IEEE ITSC / IV / ICRA 2026**.

```
Dense candidates (N_agents × K_modes × T_steps) 
                     ↓
      Stage 1: KFF (Kinematic constraints)
                     ↓
      Stage 2: SRF (Spatial boundaries)
                     ↓
      Stage 3: SCF (Social interactions GNN)
                     ↓
                Sparse Set
                     ↓
      MTR Decoder (GMM Forecasting)
```

---

## Quick Start (Reproduce in 3 Commands)

Execute the following commands from the root directory to generate datasets, evaluate models, and launch the interactive demo dashboard:

```bash
# 1. Unpack datasets and generate mock splits
python hcp_project/data/extractor.py --test

# 2. Run benchmarking evaluation (minADE/FDE/MR and latency metrics)
python hcp_project/eval/evaluate.py

# 3. Launch live telemetry control room dashboard
python hcp_project/backend/main.py
```
After launching the backend, open **[http://localhost:8000](http://localhost:8000)** in your browser to view the live dashboard.

---

## Directory Structure

```
hcp_project/
├── data/              # Data parsing and Unified Dataset Router
├── hcp/               # Hierarchical Combinatorial Pruning (KFF, SRF, SCF)
├── mtr_core/          # MTR tokenizers, encoder, decoder, and training
├── fusion/            # Geometry-conditioned cross-attention layer
├── outputs/           # Modality results: route graphs, maps, motion states
├── eval/              # Benchmarking and metrics evaluation
├── paper/             # Main LaTeX draft template filled with metrics
├── backend/           # FastAPI telemetries and stream endpoints
└── ui/                # Vite + React frontend dashboard codebase
```

---

## Core Algorithmic Components

### 1. HCP Pruner (`hcp_project/hcp/`)
- **Stage 1: Kinematic Feasibility Filter (KFF):** Prunes trajectories violating maximum curvature ($\kappa_{max}=0.2$ rad/m), jerk ($j_{max}=5.0$ m/s³), and lateral acceleration ($a_{lat\_max}=4.0$ m/s²). Complexity: $O(N \cdot K \cdot T)$
- **Stage 2: Spatial Reachability Filter (SRF):** Employs a SciPy KD-Tree index over static lane boundaries to detect off-road collisions. Complexity: $O(N \cdot K \cdot T \cdot \log M)$
- **Stage 3: Social Compatibility Filter (SCF):** A 3-layer GraphSAGE GNN representing agent-to-agent conflicts. Complexity: $O(N^2 \cdot K^2 \cdot T)$

### 2. MTR Core (`hcp_project/mtr_core/`)
- **Tokenizers:** Image patch ViT-S/16, polyline PointNet, and agent trajectory MLPs.
- **Encoder:** 6-layer Transformer utilizing Rotary Position Embeddings (RoPE).
- **Fusion Layer:** Fuses map and agent representations using a physical distance-conditioned RBF kernel attention bias.

---

## Telemetry Outputs & Live Dashboard

The dashboard serves three distinct output modalities:
1. **Modality 1 (TNT Route Graph):** Directed NetworkX waypoints and auto-playing synthesized text-to-speech navigation cues.
2. **Modality 2 (BEV Map Crop):** Google-style rendering displaying lanes, crosswalks, bounding boxes, and predicted modes.
3. **Modality 3 (Motion State NLG):** Natural Language explanations for vehicle risks and vector field plots.

---

## Docker Execution

To run the complete server stack inside a container:

```bash
docker-compose up --build
```
Open **[http://localhost:8000](http://localhost:8000)** to access the system.
