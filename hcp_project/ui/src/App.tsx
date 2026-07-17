import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Shield, Play, RotateCcw, Award, FileText, Activity, Layers, Compass } from 'lucide-react';
import Map, { Source, Layer } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

// Free dark basemap — no token or account needed
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

// Default anchor point (San Francisco intersection) for converting local meter coords -> GPS
const ANCHOR_LNG = -122.4194;
const ANCHOR_LAT = 37.7749;

interface AgentState {
  agent_id: number;
  category: string;
  speed_mps: number;
  heading_deg: number;
  accel_mps2: number;
  turn_rate_radps: number;
  ttc_seconds: number;
  risk_level: string;
  explanation: string;
}

interface ScenarioData {
  scenario_id: string;
  agent_types: string[];
  history: number[][][][];       // (N, T_hist, 6)
  predictions: number[][][][][]; // (N, K, T_fut, 5)
  confidences: number[][];       // (N, K)
  map_polylines: number[][][];
}

/**
 * Convert local meter offsets (dx, dy) from the ego origin into (lng, lat).
 * Uses simple equirectangular approximation which is accurate at this zoom.
 */
function metersToLngLat(dx: number, dy: number, anchorLng: number, anchorLat: number): [number, number] {
  const metersPerDegreeLat = 111_320;
  const metersPerDegreeLng = 111_320 * Math.cos((anchorLat * Math.PI) / 180);
  return [
    anchorLng + dx / metersPerDegreeLng,
    anchorLat + dy / metersPerDegreeLat,
  ];
}

export default function App() {
  const [tab, setTab] = useState<'dashboard' | 'viewer3d' | 'nlg'>('dashboard');
  const [scenarioId, setScenarioId] = useState<string>('scenario_0');
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [step, setStep] = useState(0);
  
  // HCP Stats State
  const [hcpStats, setHcpStats] = useState({
    raw_count: 128,
    kff_count: 74,
    srf_count: 31,
    scf_count: 9,
    total_time_ms: 32.5,
    pruning_ratio: 0.76,
    latency_reduction_pct: 71.8
  });
  
  // Active Agents and Explanations state
  const [agents, setAgents] = useState<AgentState[]>([]);
  
  // Full scenario data for map trajectories
  const [scenarioData, setScenarioData] = useState<ScenarioData | null>(null);
  
  // Ref for ThreeJS
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Fetch scenario list
    fetch('http://localhost:8000/scenarios')
      .then(res => res.json())
      .then(data => {
        setScenarios(data);
        if (data.length > 0) {
          setScenarioId(data[0]);
          loadScenario(data[0]);
        }
      })
      .catch(() => {
        // Fallback for isolated client rendering
        setScenarios(['scenario_0', 'scenario_1', 'scenario_2']);
        loadScenario('scenario_0');
      });
  }, []);

  const loadScenario = (id: string) => {
    setScenarioId(id);
    setStep(0);
    setIsPlaying(false);
    
    // Fetch motion states
    fetch(`http://localhost:8000/motion_states/${id}`)
      .then(res => res.json())
      .then(data => setAgents(data))
      .catch(() => {
        // Mock fallback
        setAgents([
          {
            agent_id: 0, category: 'ego_vehicle', speed_mps: 12.4, heading_deg: 28.5,
            accel_mps2: -2.1, turn_rate_radps: 0.0, ttc_seconds: -1,
            risk_level: 'low', explanation: 'Ego vehicle tracking target route trajectory.'
          },
          {
            agent_id: 1, category: 'vehicle', speed_mps: 8.5, heading_deg: 340.0,
            accel_mps2: 0.5, turn_rate_radps: -0.15, ttc_seconds: 3.2,
            risk_level: 'medium', explanation: 'Vehicle #1 moving at 8.5 m/s, decelerating at 0.5 m/s². Medium risk close approach.'
          }
        ]);
      });

    // Fetch full scenario data (predictions + history) for map rendering
    fetch(`http://localhost:8000/scenario/${id}`)
      .then(res => res.json())
      .then((data: ScenarioData) => setScenarioData(data))
      .catch(() => {
        // Mock fallback with simple trajectories
        const mockPreds: number[][][][][] = [];
        // Ego agent
        const egoModes: number[][][] = [];
        const egoMode: number[][] = [];
        for (let t = 0; t < 12; t++) {
          egoMode.push([(t + 1) * 2.5, (t + 1) * 0.5, 5.0, 1.0, 0.2]);
        }
        egoModes.push(egoMode);
        mockPreds.push(egoModes);

        // Another agent
        const agentModes: number[][][] = [];
        const agentMode: number[][] = [];
        for (let t = 0; t < 12; t++) {
          agentMode.push([15 + (t + 1) * 1.0, -5 + (t + 1) * 0.3, 2.0, 0.6, -0.1]);
        }
        agentModes.push(agentMode);
        mockPreds.push(agentModes);

        setScenarioData({
          scenario_id: id,
          agent_types: ['ego_vehicle', 'vehicle'],
          history: [
            [[0, 0, 0, 0, 0, 0]].map(h => [h]),
            [[15, -5, 0, 0, 0, 0]].map(h => [h]),
          ] as any,
          predictions: mockPreds,
          confidences: [[1.0], [1.0]],
          map_polylines: [],
        });
      });
  };

  const runHCP = () => {
    fetch(`http://localhost:8000/run_hcp/${scenarioId}`, { method: 'POST' })
      .then(res => res.json())
      .then(data => setHcpStats(data))
      .catch(() => {
        // Mock transition
        setHcpStats({
          raw_count: 128,
          kff_count: 70,
          srf_count: 28,
          scf_count: 8,
          total_time_ms: 31.8,
          pruning_ratio: 0.78,
          latency_reduction_pct: 72.4
        });
      });
  };

  // Build GeoJSON features for the Mapbox map
  const { trajectoryGeoJSON, markersGeoJSON } = useMemo(() => {
    const trajectoryFeatures: GeoJSON.Feature[] = [];
    const markerFeatures: GeoJSON.Feature[] = [];

    if (!scenarioData || !agents.length) {
      return {
        trajectoryGeoJSON: { type: 'FeatureCollection' as const, features: [] },
        markersGeoJSON: { type: 'FeatureCollection' as const, features: [] },
      };
    }

    const preds = scenarioData.predictions;
    const confs = scenarioData.confidences;

    for (let n = 0; n < preds.length && n < agents.length; n++) {
      const agent = agents[n];
      // Pick the highest-confidence mode
      const modeConfs = confs[n];
      let bestMode = 0;
      let bestConf = -1;
      for (let k = 0; k < modeConfs.length; k++) {
        if (modeConfs[k] > bestConf) {
          bestConf = modeConfs[k];
          bestMode = k;
        }
      }

      const traj = preds[n][bestMode]; // (T_fut, 5) -> [x, y, vx, vy, heading]
      if (!traj || traj.length === 0) continue;

      // Convert trajectory points to lng/lat
      const coords: [number, number][] = traj.map((pt: number[]) =>
        metersToLngLat(pt[0], pt[1], ANCHOR_LNG, ANCHOR_LAT)
      );

      // Determine color category
      const isEgo = agent.agent_id === 0 || agent.category === 'ego_vehicle';
      const isHighRisk = agent.risk_level === 'high';
      const colorCategory = isEgo ? 'ego' : isHighRisk ? 'high_risk' : 'other';

      // Line feature
      trajectoryFeatures.push({
        type: 'Feature',
        properties: {
          agentId: agent.agent_id,
          colorCategory,
          riskLevel: agent.risk_level,
        },
        geometry: {
          type: 'LineString',
          coordinates: coords,
        },
      });

      // Marker at current position (first predicted point)
      markerFeatures.push({
        type: 'Feature',
        properties: {
          agentId: agent.agent_id,
          colorCategory,
          riskLevel: agent.risk_level,
          label: isEgo ? 'EGO' : `#${agent.agent_id}`,
        },
        geometry: {
          type: 'Point',
          coordinates: coords[0],
        },
      });
    }

    return {
      trajectoryGeoJSON: { type: 'FeatureCollection' as const, features: trajectoryFeatures },
      markersGeoJSON: { type: 'FeatureCollection' as const, features: markerFeatures },
    };
  }, [scenarioData, agents]);

  // Mapbox layer styles
  const egoLineLayer = {
    id: 'ego-trajectory',
    type: 'line',
    filter: ['==', ['get', 'colorCategory'], 'ego'],
    paint: {
      'line-color': '#1D9E75',
      'line-width': 4,
      'line-opacity': 0.9,
    },
  };

  const otherLineLayer = {
    id: 'other-trajectory',
    type: 'line',
    filter: ['==', ['get', 'colorCategory'], 'other'],
    paint: {
      'line-color': '#378ADD',
      'line-width': 2.5,
      'line-opacity': 0.8,
    },
  };

  const highRiskLineLayer = {
    id: 'high-risk-trajectory',
    type: 'line',
    filter: ['==', ['get', 'colorCategory'], 'high_risk'],
    paint: {
      'line-color': '#D85A30',
      'line-width': 2.5,
      'line-opacity': 0.9,
    },
  };

  const markerLayer = {
    id: 'agent-markers',
    type: 'circle',
    paint: {
      'circle-radius': 7,
      'circle-color': [
        'match',
        ['get', 'colorCategory'],
        'ego', '#1D9E75',
        'high_risk', '#D85A30',
        '#378ADD', // default for 'other'
      ],
      'circle-stroke-width': 2,
      'circle-stroke-color': '#0a0f14',
      'circle-opacity': 1,
    },
  };

  return (
    <div className="min-h-screen bg-[#0a0f14] p-6 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="flex justify-between items-center pb-5 mb-5 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-extrabold text-[#1D9E75] flex items-center gap-2">
            HCP + MTR Telemetry Dashboard
            <span className="text-[10px] uppercase bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 px-2 py-0.5 rounded-full font-bold">React Node</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">Hierarchical Combinatorial Pruning + Motion Transformer Real-Time Control Center</p>
        </div>
        
        <div className="flex items-center gap-4">
          <div>
            <label className="text-[10px] text-slate-400 block mb-1">Select Scenario</label>
            <select 
              value={scenarioId} 
              onChange={e => loadScenario(e.target.value)} 
              className="bg-slate-900 border border-slate-700 text-xs rounded px-3 py-1.5 focus:outline-none focus:border-[#1D9E75]"
            >
              {scenarios.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <button 
            onClick={runHCP} 
            className="bg-[#1D9E75] hover:bg-emerald-600 text-[#0a0f14] font-bold text-xs px-4 py-2 rounded transition"
          >
            Trigger HCP
          </button>
        </div>
      </header>

      {/* Tabs */}
      <div className="flex gap-4 mb-6 border-b border-slate-800 pb-2">
        <button 
          onClick={() => setTab('dashboard')} 
          className={`px-4 py-2 text-xs font-bold border-b-2 transition ${tab === 'dashboard' ? 'border-[#1D9E75] text-[#1D9E75]' : 'border-transparent text-slate-400 hover:text-white'}`}
        >
          1. Control Room BEV
        </button>
        <button 
          onClick={() => setTab('viewer3d')} 
          className={`px-4 py-2 text-xs font-bold border-b-2 transition ${tab === 'viewer3d' ? 'border-[#1D9E75] text-[#1D9E75]' : 'border-transparent text-slate-400 hover:text-white'}`}
        >
          2. 3D Scene Ribbon
        </button>
        <button 
          onClick={() => setTab('nlg')} 
          className={`px-4 py-2 text-xs font-bold border-b-2 transition ${tab === 'nlg' ? 'border-[#1D9E75] text-[#1D9E75]' : 'border-transparent text-slate-400 hover:text-white'}`}
        >
          3. State Explainer
        </button>
      </div>

      {/* Contents */}
      {tab === 'dashboard' && (
        <div className="grid grid-cols-12 gap-6 flex-grow">
          {/* Left: Agent list */}
          <div className="col-span-3 bg-slate-950/50 border border-slate-800 p-4 rounded-xl flex flex-col h-[580px]">
            <h2 className="text-sm font-bold border-b border-slate-800 pb-2 mb-3">Agent Intelligence Feed</h2>
            <div className="flex-grow overflow-y-auto space-y-3 pr-1">
              {agents.map(a => (
                <div key={a.agent_id} className={`p-3 border rounded-lg text-xs flex justify-between items-center ${a.agent_id === 0 ? 'bg-emerald-950/10 border-emerald-900/40' : 'bg-slate-900/30 border-slate-800'}`}>
                  <div>
                    <div className="font-bold flex items-center gap-1.5">
                      <span>{a.agent_id === 0 ? 'Ego Vehicle' : `Agent #${a.agent_id}`}</span>
                      <span className="text-[9px] bg-slate-800 text-slate-400 px-1 py-0.5 rounded uppercase">{a.category}</span>
                    </div>
                    <div className="text-[10px] text-slate-400 mt-1">Speed: <span className="font-mono">{a.speed_mps} m/s</span></div>
                  </div>
                  <span className={`px-2 py-0.5 rounded text-[9px] font-bold ${a.risk_level === 'high' ? 'bg-red-950 text-red-400 border border-red-900/50' : 'bg-slate-800 text-slate-400'}`}>
                    {a.risk_level.toUpperCase()}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Center: MapLibre HD map */}
          <div className="col-span-6 bg-slate-950/50 border border-slate-800 p-4 rounded-xl flex flex-col h-[580px]">
            <h2 className="text-sm font-bold border-b border-slate-800 pb-2 mb-3 flex justify-between items-center">
              <span>Ego-Centric BEV Crop (500m)</span>
              <span className="text-[10px] text-[#1D9E75] font-semibold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-[#1D9E75] animate-pulse"></span> 10Hz Feed
              </span>
            </h2>
            <div className="flex-grow rounded-lg overflow-hidden border border-slate-800 bg-[#0d131a] relative">
              <Map
                initialViewState={{
                  longitude: ANCHOR_LNG,
                  latitude: ANCHOR_LAT,
                  zoom: 16,
                }}
                style={{ width: '100%', height: '100%' }}
                mapStyle={MAP_STYLE}
                attributionControl={false}
              >
                {/* Trajectory lines */}
                <Source id="trajectories" type="geojson" data={trajectoryGeoJSON}>
                  <Layer {...egoLineLayer} />
                  <Layer {...otherLineLayer} />
                  <Layer {...highRiskLineLayer} />
                </Source>

                {/* Agent position markers */}
                <Source id="markers" type="geojson" data={markersGeoJSON}>
                  <Layer {...markerLayer} />
                </Source>
              </Map>
            </div>
            <div className="flex justify-between items-center mt-3 text-xs">
              <div className="flex gap-2">
                <button className="bg-slate-800 px-3 py-1 rounded hover:bg-slate-700">Play</button>
                <button className="bg-slate-800 px-3 py-1 rounded hover:bg-slate-700">Reset</button>
              </div>
              <span className="text-slate-400">Time: <span className="text-[#1D9E75] font-bold">0.0s / 6.0s</span></span>
            </div>
          </div>

          {/* Right: HCP Metrics */}
          <div className="col-span-3 bg-slate-950/50 border border-slate-800 p-4 rounded-xl flex flex-col h-[580px] justify-between">
            <div>
              <h2 className="text-sm font-bold border-b border-slate-800 pb-2 mb-4">HCP Pruning Waterfall</h2>
              <div className="space-y-4 text-xs">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-400">Raw Candidate Search Space</span>
                    <span className="font-mono text-slate-200">128 (100%)</span>
                  </div>
                  <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden">
                    <div className="bg-slate-400 h-full w-[100%]"></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-400">Stage 1: KFF (Kinematic)</span>
                    <span className="font-mono text-slate-200">{hcpStats.kff_count} ({Math.round(hcpStats.kff_count/hcpStats.raw_count*100)}%)</span>
                  </div>
                  <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden">
                    <div className="bg-slate-500 h-full w-[58%]"></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-slate-400">Stage 2: SRF (Spatial)</span>
                    <span className="font-mono text-slate-200">{hcpStats.srf_count} ({Math.round(hcpStats.srf_count/hcpStats.raw_count*100)}%)</span>
                  </div>
                  <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden">
                    <div className="bg-[#378ADD] h-full w-[24%]"></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-[#1D9E75] font-bold">Stage 3: SCF (Social)</span>
                    <span className="font-mono text-emerald-400 font-bold">{hcpStats.scf_count} ({Math.round(hcpStats.scf_count/hcpStats.raw_count*100)}%)</span>
                  </div>
                  <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden">
                    <div className="bg-[#1D9E75] h-full w-[7%]"></div>
                  </div>
                </div>
              </div>
            </div>

            <div className="border-t border-slate-800 pt-4 mt-6">
              <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider block mb-2 text-center">Telemetry Metrics</span>
              <div className="grid grid-cols-2 gap-2 text-center text-xs">
                <div className="bg-slate-900/40 border border-slate-800 p-2.5 rounded-lg">
                  <span className="text-[9px] text-slate-400 block mb-0.5">Inference Latency</span>
                  <span className="text-sm font-black text-white font-mono">{hcpStats.total_time_ms}ms</span>
                </div>
                <div className="bg-slate-900/40 border border-slate-800 p-2.5 rounded-lg">
                  <span className="text-[9px] text-slate-400 block mb-0.5">Latency Reduction</span>
                  <span className="text-sm font-black text-[#1D9E75] font-mono">{hcpStats.latency_reduction_pct.toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {tab === 'viewer3d' && (
        <div className="bg-slate-950/50 border border-slate-800 p-5 rounded-xl flex flex-col h-[580px]">
          <h2 className="text-sm font-bold border-b border-slate-800 pb-2 mb-3">3D Trajectory Ribbon Viewer</h2>
          <div className="flex-grow rounded-lg overflow-hidden border border-slate-800 bg-[#0d131a] relative flex items-center justify-center">
            <span className="text-xs text-slate-400">3D WebGL renderer compiles under Vite. In-browser demo preview serves via FastAPI landing page directly.</span>
          </div>
        </div>
      )}

      {tab === 'nlg' && (
        <div className="bg-slate-950/50 border border-slate-800 p-5 rounded-xl flex flex-col h-[580px]">
          <h2 className="text-sm font-bold border-b border-slate-800 pb-2 mb-4">State Explainer</h2>
          <div className="overflow-x-auto flex-grow">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900 text-slate-300 font-semibold border-b border-slate-800">
                <tr>
                  <th className="p-3">Agent</th>
                  <th className="p-3">Type</th>
                  <th className="p-3">Speed</th>
                  <th className="p-3">TTC (s)</th>
                  <th className="p-3">Risk</th>
                  <th className="p-3">NLG Explanation</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {agents.map(a => (
                  <tr key={a.agent_id} className="hover:bg-slate-900/30">
                    <td className="p-3 font-bold">#{a.agent_id}</td>
                    <td className="p-3 text-slate-400">{a.category}</td>
                    <td className="p-3 font-mono">{a.speed_mps} m/s</td>
                    <td className="p-3 font-mono">{a.ttc_seconds > 0 ? a.ttc_seconds + 's' : 'N/A'}</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded font-bold ${a.risk_level === 'high' ? 'bg-red-950/40 text-red-500 border border-red-500/30' : 'bg-slate-900 text-slate-400'}`}>
                        {a.risk_level.toUpperCase()}
                      </span>
                    </td>
                    <td className="p-3 italic text-slate-300">{a.explanation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
