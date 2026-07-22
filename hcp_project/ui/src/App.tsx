import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Activity, Layers, Compass, Box, Zap, Shield,
  Radio, ChevronRight, Gauge, Timer, TrendingDown,
  Play, Square, RotateCcw, AlertTriangle, MapPin, Wifi, WifiOff,
} from 'lucide-react';
import Map, { Source, Layer, Marker } from 'react-map-gl/maplibre';
import type { MapRef } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

/* ═══════════════════ CONSTANTS ═══════════════════ */
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
// API base: empty string = same origin (works with Vite proxy and production)
const API = '';

/* ═══════════════════ TYPES ═══════════════════ */
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

interface HCPStats {
  raw_count: number;
  kff_count: number;
  srf_count: number;
  scf_count: number;
  total_time_ms: number;
  pruning_ratio: number;
  latency_reduction_pct: number;
}

interface ScenarioData {
  scenario_id: string;
  agent_types: string[];
  history: number[][][];
  predictions: number[][][][];
  confidences: number[][];
  ego_origin: { lat: number; lng: number; city: string };
  geo_predictions: number[][][][]; // [agent][mode][step][lng,lat]
  agent_positions: { lng: number; lat: number; type: string }[];
}

interface StreamFrame {
  step: number;
  agents: { agent_id: number; type: string; x: number; y: number; lng: number; lat: number; heading: number }[];
}

/* ═══════════════════ HELPERS ═══════════════════ */
const riskColor = (level?: string) => {
  switch (level?.toLowerCase()) {
    case 'high': return { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', dot: 'bg-red-500' };
    case 'medium': return { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', dot: 'bg-amber-500' };
    default: return { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', dot: 'bg-emerald-500' };
  }
};

const pct = (val: number, total: number) => total > 0 ? Math.round((val / total) * 100) : 0;

const TRAJ_COLORS = [
  { hex: '#06b6d4', label: 'Best Mode' },     // cyan
  { hex: '#38bdf8', label: 'Alt Mode 1' },    // sky
  { hex: '#f59e0b', label: 'Alt Mode 2' },    // amber
  { hex: '#a78bfa', label: 'Alt Mode 3' },    // violet
  { hex: '#94a3b8', label: 'Alt Mode 4' },    // slate
  { hex: '#64748b', label: 'Alt Mode 5' },    // gray
];

const AGENT_COLORS: Record<string, string> = {
  ego_vehicle: '#10b981',
  vehicle: '#f43f5e',
  pedestrian: '#f59e0b',
};

/* ═══════════════════ GLASS CARD COMPONENT ═══════════════════ */
const Glass: React.FC<{ children: React.ReactNode; className?: string; style?: React.CSSProperties }> = ({ children, className = '', style }) => (
  <div className={`glass ${className}`} style={style}>{children}</div>
);

/* ═══════════════════ PRUNING BAR ═══════════════════ */
const PruneBar: React.FC<{ label: string; value: number; total: number; color: string; dotColor: string }> = ({
  label, value, total, color, dotColor,
}) => (
  <div className="space-y-1.5">
    <div className="flex justify-between text-[11px]">
      <span className="text-slate-400 flex items-center gap-1.5">
        <span className={`w-1.5 h-1.5 rounded-full ${dotColor}`} />
        {label}
      </span>
      <span className="font-mono font-bold text-slate-300">{value} <span className="text-slate-500">({pct(value, total)}%)</span></span>
    </div>
    <div className="w-full h-1.5 rounded-full bg-white/[0.04] overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-700 ease-out ${color}`}
        style={{ width: `${pct(value, total)}%` }}
      />
    </div>
  </div>
);

/* ═══════════════════ 3D SCENE COMPONENT ═══════════════════ */
const Scene3D: React.FC<{ scenario: ScenarioData | null }> = ({ scenario }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameIdRef = useRef<number>(0);
  const sceneRef = useRef<any>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    let Three: typeof import('three');
    let renderer: any, scene: any, camera: any, controls: any;
    let animationId: number;

    const init = async () => {
      Three = await import('three');
      const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js');

      const canvas = canvasRef.current!;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;

      // Renderer
      renderer = new Three.WebGLRenderer({ canvas, antialias: true, alpha: true });
      renderer.setSize(w, h);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setClearColor(0x060a10, 1);

      // Scene
      scene = new Three.Scene();
      scene.fog = new Three.FogExp2(0x060a10, 0.008);
      sceneRef.current = scene;

      // Camera
      camera = new Three.PerspectiveCamera(50, w / h, 0.1, 500);
      camera.position.set(0, 60, 80);
      camera.lookAt(0, 0, 0);

      // Controls
      controls = new OrbitControls(camera, canvas);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.maxDistance = 200;
      controls.minDistance = 10;

      // Grid
      const gridHelper = new Three.GridHelper(200, 40, 0x1e293b, 0x0f172a);
      scene.add(gridHelper);

      // Axes
      const axesHelper = new Three.AxesHelper(10);
      scene.add(axesHelper);

      // Lighting
      const ambient = new Three.AmbientLight(0xffffff, 0.3);
      scene.add(ambient);
      const dirLight = new Three.DirectionalLight(0xffffff, 0.8);
      dirLight.position.set(30, 50, 30);
      scene.add(dirLight);
      const pointLight = new Three.PointLight(0x06b6d4, 1, 100);
      pointLight.position.set(0, 20, 0);
      scene.add(pointLight);

      // Build scene from scenario data
      if (scenario) {
        buildSceneFromData(Three, scene, scenario);
      } else {
        buildDemoScene(Three, scene);
      }

      // Animate
      const animate = () => {
        animationId = requestAnimationFrame(animate);
        controls.update();

        // Subtle animation of trajectory ribbons
        scene.traverse((obj: any) => {
          if (obj.userData?.isTrajectory) {
            obj.material.opacity = 0.5 + 0.15 * Math.sin(Date.now() * 0.002 + (obj.userData.index || 0));
          }
        });

        renderer.render(scene, camera);
      };
      animate();

      // Handle resize
      const onResize = () => {
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };
      window.addEventListener('resize', onResize);

      return () => {
        window.removeEventListener('resize', onResize);
        cancelAnimationFrame(animationId);
        renderer.dispose();
      };
    };

    const cleanup = init();
    return () => { cleanup.then(fn => fn?.()); };
  }, [scenario]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full rounded-xl"
      style={{ minHeight: 'calc(100vh - 200px)' }}
    />
  );
};

function buildSceneFromData(Three: typeof import('three'), scene: any, data: ScenarioData) {
  const predictions = data.predictions;
  const agentTypes = data.agent_types;

  // Draw agent bounding boxes at last history position
  for (let n = 0; n < agentTypes.length; n++) {
    const hist = data.history[n];
    const lastPt = hist[hist.length - 1];
    const x = lastPt[0], y = lastPt[1];
    const isEgo = n === 0;
    const isPed = agentTypes[n] === 'pedestrian';

    const boxW = isPed ? 0.6 : 1.8;
    const boxH = isPed ? 1.7 : 1.5;
    const boxD = isPed ? 0.6 : 4.5;

    const color = isEgo ? 0x10b981 : (isPed ? 0xf59e0b : 0xf43f5e);
    const geometry = new Three.BoxGeometry(boxW, boxH, boxD);
    const material = new Three.MeshStandardMaterial({
      color, transparent: true, opacity: 0.85,
      emissive: new Three.Color(color), emissiveIntensity: 0.3,
    });
    const mesh = new Three.Mesh(geometry, material);
    mesh.position.set(x, boxH / 2, -y);
    scene.add(mesh);

    // Wireframe outline
    const edges = new Three.EdgesGeometry(geometry);
    const lineMat = new Three.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.4 });
    const wireframe = new Three.LineSegments(edges, lineMat);
    wireframe.position.copy(mesh.position);
    scene.add(wireframe);

    // Label
    if (isEgo) {
      const ringGeo = new Three.RingGeometry(2.5, 3.0, 32);
      const ringMat = new Three.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.4, side: Three.DoubleSide });
      const ring = new Three.Mesh(ringGeo, ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, 0.05, -y);
      scene.add(ring);
    }
  }

  // Draw trajectory ribbons
  for (let n = 0; n < Math.min(predictions.length, 4); n++) {
    const agentPreds = predictions[n];
    const numModes = Math.min(agentPreds.length, 3);
    for (let k = 0; k < numModes; k++) {
      const modeTraj = agentPreds[k];
      const points: any[] = [];
      for (let t = 0; t < modeTraj.length; t++) {
        points.push(new Three.Vector3(modeTraj[t][0], 0.3 + k * 0.2, -modeTraj[t][1]));
      }

      if (points.length < 2) continue;
      const curve = new Three.CatmullRomCurve3(points);
      const tubeGeo = new Three.TubeGeometry(curve, 30, 0.15 - k * 0.03, 6, false);
      const isEgo = n === 0;
      const colorHex = isEgo
        ? [0x06b6d4, 0x38bdf8, 0xf59e0b][k]
        : [0xf43f5e, 0xfb923c, 0xfbbf24][k];

      const tubeMat = new Three.MeshStandardMaterial({
        color: colorHex, transparent: true, opacity: 0.6 - k * 0.15,
        emissive: new Three.Color(colorHex), emissiveIntensity: 0.5,
      });
      const tube = new Three.Mesh(tubeGeo, tubeMat);
      tube.userData = { isTrajectory: true, index: n * 3 + k };
      scene.add(tube);

      // Endpoint sphere
      const lastPt = points[points.length - 1];
      const sphereGeo = new Three.SphereGeometry(0.3, 16, 16);
      const sphereMat = new Three.MeshStandardMaterial({ color: colorHex, emissive: new Three.Color(colorHex), emissiveIntensity: 0.6 });
      const sphere = new Three.Mesh(sphereGeo, sphereMat);
      sphere.position.copy(lastPt);
      scene.add(sphere);
    }
  }

  // Draw map polylines as lines on the ground
  for (const poly of data.history.length > 0 ? (data as any).map_polylines || [] : []) {
    /* skip — map_polylines aren't always in data at this point */
  }
}

function buildDemoScene(Three: typeof import('three'), scene: any) {
  // Fallback demo: a few boxes and lines
  const colors = [0x10b981, 0xf43f5e, 0xf59e0b];
  for (let i = 0; i < 3; i++) {
    const geo = new Three.BoxGeometry(1.8, 1.5, 4.5);
    const mat = new Three.MeshStandardMaterial({
      color: colors[i], transparent: true, opacity: 0.85,
      emissive: new Three.Color(colors[i]), emissiveIntensity: 0.3,
    });
    const mesh = new Three.Mesh(geo, mat);
    mesh.position.set(i * 8 - 8, 0.75, 0);
    scene.add(mesh);
  }

  // Demo trajectory curves
  for (let k = 0; k < 3; k++) {
    const pts = [];
    for (let t = 0; t < 12; t++) {
      pts.push(new Three.Vector3(t * 3, 0.3, -t * (k - 1) * 0.8));
    }
    const curve = new Three.CatmullRomCurve3(pts);
    const tubeGeo = new Three.TubeGeometry(curve, 30, 0.12, 6, false);
    const tubeMat = new Three.MeshStandardMaterial({
      color: [0x06b6d4, 0x38bdf8, 0xf59e0b][k],
      transparent: true, opacity: 0.6,
      emissive: new Three.Color([0x06b6d4, 0x38bdf8, 0xf59e0b][k]),
      emissiveIntensity: 0.5,
    });
    const tube = new Three.Mesh(tubeGeo, tubeMat);
    tube.userData = { isTrajectory: true, index: k };
    scene.add(tube);
  }
}

/* ═══════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════ */
export default function App() {
  const [tab, setTab] = useState<'dashboard' | 'viewer3d' | 'nlg'>('dashboard');
  const [scenarioId, setScenarioId] = useState('scenario_0');
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [agents, setAgents] = useState<AgentState[]>([]);
  const [scenarioData, setScenarioData] = useState<ScenarioData | null>(null);
  const [hcp, setHcp] = useState<HCPStats>({
    raw_count: 128, kff_count: 74, srf_count: 31, scf_count: 9,
    total_time_ms: 32.5, pruning_ratio: 0.76, latency_reduction_pct: 71.8,
  });
  const [isLiveData, setIsLiveData] = useState(false);
  const [isHcpLive, setIsHcpLive] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);

  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [liveAgentPositions, setLiveAgentPositions] = useState<{ lng: number; lat: number; type: string }[]>([]);
  const sseRef = useRef<EventSource | null>(null);

  // Map ref
  const mapRef = useRef<MapRef>(null);
  const abortRef = useRef<AbortController | null>(null);

  /* ── Map viewport state ── */
  const [mapCenter, setMapCenter] = useState<{ lng: number; lat: number }>({ lng: -122.4194, lat: 37.7749 });

  /* ── Init ── */
  useEffect(() => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    fetch(`${API}/scenarios`, { signal: ctrl.signal })
      .then(r => r.json())
      .then((d: string[]) => {
        if (!Array.isArray(d)) return;
        setScenarios(d);
        setBackendOnline(true);
        setIsLiveData(true);
        if (d.length > 0) { setScenarioId(d[0]); loadScenario(d[0]); }
      })
      .catch(e => {
        if (e?.name === 'AbortError') return;
        setBackendOnline(false);
        setIsLiveData(false);
        setScenarios(['scenario_0', 'scenario_1', 'scenario_2']);
        loadScenario('scenario_0');
      });
    return () => ctrl.abort();
  }, []);

  const loadScenario = useCallback((id: string) => {
    setScenarioId(id);
    stopPlayback();
    setCurrentFrame(0);

    // Load full scenario data
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    fetch(`${API}/scenario/${id}`, { signal: ctrl.signal })
      .then(r => r.json())
      .then((d: ScenarioData) => {
        setScenarioData(d);
        setBackendOnline(true);
        setIsLiveData(true);
        // Center map on ego origin
        if (d.ego_origin) {
          setMapCenter({ lng: d.ego_origin.lng, lat: d.ego_origin.lat });
          mapRef.current?.flyTo({
            center: [d.ego_origin.lng, d.ego_origin.lat],
            zoom: 16,
            duration: 1200,
          });
        }
        // Set initial live agent positions
        if (d.agent_positions) {
          setLiveAgentPositions(d.agent_positions);
        }
      })
      .catch(e => {
        if (e?.name === 'AbortError') return;
        setBackendOnline(false);
        setIsLiveData(false);
        setScenarioData(null);
      });

    // Load motion states
    fetch(`${API}/motion_states/${id}`, { signal: ctrl.signal })
      .then(r => r.json())
      .then(d => { if (Array.isArray(d)) setAgents(d); })
      .catch(e => {
        if (e?.name === 'AbortError') return;
        setAgents([
          { agent_id: 0, category: 'ego_vehicle', speed_mps: 12.4, heading_deg: 28.5, accel_mps2: -2.1, turn_rate_radps: 0, ttc_seconds: -1, risk_level: 'low', explanation: 'Ego vehicle tracking target route trajectory.' },
          { agent_id: 1, category: 'vehicle', speed_mps: 8.5, heading_deg: 340, accel_mps2: 0.5, turn_rate_radps: -0.15, ttc_seconds: 3.2, risk_level: 'medium', explanation: 'Vehicle #1 approaching. Medium risk.' },
          { agent_id: 2, category: 'pedestrian', speed_mps: 1.2, heading_deg: 90, accel_mps2: 0, turn_rate_radps: 0, ttc_seconds: 8.1, risk_level: 'low', explanation: 'Pedestrian on sidewalk, safe distance.' },
        ]);
      });
  }, []);

  /* ── Playback (SSE stream) ── */
  const startPlayback = useCallback(() => {
    stopPlayback();
    setIsPlaying(true);
    setCurrentFrame(0);

    const sse = new EventSource(`${API}/stream/${scenarioId}`);
    sseRef.current = sse;

    sse.onmessage = (event) => {
      try {
        const frame: StreamFrame = JSON.parse(event.data);
        setCurrentFrame(frame.step + 1);

        // Update agent positions on map
        const newPositions = frame.agents.map(a => ({
          lng: a.lng, lat: a.lat, type: a.type,
        }));
        setLiveAgentPositions(newPositions);
      } catch (e) { /* ignore parse errors */ }
    };

    sse.onerror = () => {
      sse.close();
      sseRef.current = null;
      setIsPlaying(false);
      // Fallback: step through frames locally
      if (scenarioData) {
        startLocalPlayback();
      }
    };
  }, [scenarioId, scenarioData]);

  const startLocalPlayback = useCallback(() => {
    if (!scenarioData) return;
    setIsPlaying(true);
    let frame = 0;
    const iv = setInterval(() => {
      frame++;
      setCurrentFrame(frame);
      // Move agent markers to prediction positions at this frame
      const newPos = scenarioData.geo_predictions.map((agentModes, n) => ({
        lng: agentModes[0]?.[Math.min(frame - 1, agentModes[0].length - 1)]?.[0] ?? scenarioData.agent_positions[n]?.lng ?? 0,
        lat: agentModes[0]?.[Math.min(frame - 1, agentModes[0].length - 1)]?.[1] ?? scenarioData.agent_positions[n]?.lat ?? 0,
        type: scenarioData.agent_types[n] ?? 'vehicle',
      }));
      setLiveAgentPositions(newPos);
      if (frame >= 12) {
        clearInterval(iv);
        setIsPlaying(false);
      }
    }, 500);
    // Store interval id for cleanup
    (sseRef as any).__localIv = iv;
  }, [scenarioData]);

  const stopPlayback = useCallback(() => {
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
    if ((sseRef as any).__localIv) { clearInterval((sseRef as any).__localIv); }
    setIsPlaying(false);
  }, []);

  const resetPlayback = useCallback(() => {
    stopPlayback();
    setCurrentFrame(0);
    if (scenarioData?.agent_positions) {
      setLiveAgentPositions(scenarioData.agent_positions);
    }
  }, [scenarioData, stopPlayback]);

  /* ── HCP Run ── */
  const runHCP = useCallback(() => {
    fetch(`${API}/run_hcp/${scenarioId}`, { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        if (d && typeof d === 'object') {
          setHcp(d as HCPStats);
          setIsHcpLive(true);
        }
      })
      .catch(() => {
        setHcp({ raw_count: 128, kff_count: 70, srf_count: 28, scf_count: 8, total_time_ms: 31.8, pruning_ratio: 0.78, latency_reduction_pct: 72.4 });
        setIsHcpLive(false);
      });
  }, [scenarioId]);

  /* ── Build trajectory GeoJSON ── */
  const trajectoryGeoJSON = useMemo(() => {
    if (!scenarioData?.geo_predictions) return null;
    const features: any[] = [];

    scenarioData.geo_predictions.forEach((agentModes, agentIdx) => {
      const numModes = Math.min(agentModes.length, 3);
      for (let k = 0; k < numModes; k++) {
        const coords = agentModes[k];
        if (!coords || coords.length < 2) continue;
        features.push({
          type: 'Feature',
          properties: {
            agentIdx,
            modeIdx: k,
            isEgo: agentIdx === 0,
            color: agentIdx === 0 ? TRAJ_COLORS[k]?.hex : '#f43f5e',
            width: agentIdx === 0 ? (4 - k) : 2,
            opacity: agentIdx === 0 ? (0.9 - k * 0.2) : 0.5,
          },
          geometry: {
            type: 'LineString',
            coordinates: coords,
          },
        });
      }
    });

    return { type: 'FeatureCollection' as const, features };
  }, [scenarioData]);

  const tabs = [
    { key: 'dashboard' as const, label: 'Control Room', icon: Compass },
    { key: 'viewer3d' as const, label: '3D Scene', icon: Box },
    { key: 'nlg' as const, label: 'Explainer', icon: Layers },
  ];

  /* ═══════════════════ RENDER ═══════════════════ */
  return (
    <div className="min-h-screen bg-[#060a10] text-white flex flex-col">

      {/* ═══════════ HEADER ═══════════ */}
      <header className="px-6 pt-5 pb-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Logo mark */}
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-emerald-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight flex items-center gap-2">
              HCP + MTR
              <span className={`text-[9px] uppercase px-2 py-0.5 rounded-full font-bold tracking-widest border ${
                backendOnline
                  ? 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
                  : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
              }`}>
                {backendOnline ? 'Live' : 'Offline'}
              </span>
            </h1>
            <p className="text-[11px] text-slate-500 tracking-wide">
              Hierarchical Combinatorial Pruning · Motion Transformer
              {scenarioData?.ego_origin?.city && (
                <span className="ml-2 text-cyan-400/60">
                  📍 {scenarioData.ego_origin.city}
                </span>
              )}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Backend status */}
          <div className="glass px-2.5 py-1.5 flex items-center gap-1.5">
            {backendOnline
              ? <Wifi className="w-3 h-3 text-emerald-400" />
              : <WifiOff className="w-3 h-3 text-amber-400" />
            }
            <span className={`text-[9px] font-bold ${backendOnline ? 'text-emerald-400' : 'text-amber-400'}`}>
              {backendOnline ? 'CONNECTED' : 'MOCK DATA'}
            </span>
          </div>

          {/* Scenario selector */}
          <div className="glass px-3 py-2 flex items-center gap-2">
            <span className="text-[9px] text-slate-500 uppercase tracking-wider font-semibold">Scenario</span>
            <select
              value={scenarioId}
              onChange={e => loadScenario(e.target.value)}
              className="bg-transparent text-xs font-mono font-bold text-white focus:outline-none cursor-pointer"
            >
              {(scenarios ?? []).map(s => <option key={s} value={s} className="bg-[#0a0f14]">{s}</option>)}
            </select>
          </div>
          {/* HCP Trigger */}
          <button onClick={runHCP} className="group relative px-5 py-2.5 rounded-xl font-bold text-xs overflow-hidden transition-all hover:shadow-lg hover:shadow-cyan-500/20">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-emerald-500 transition-opacity" />
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-emerald-400 opacity-0 group-hover:opacity-100 transition-opacity" />
            <span className="relative flex items-center gap-2 text-[#060a10]">
              <Zap className="w-3.5 h-3.5" />
              Run HCP Pipeline
              {isHcpLive && <span className="w-1.5 h-1.5 rounded-full bg-[#060a10] animate-pulse" />}
            </span>
          </button>
        </div>
      </header>

      {/* ═══════════ NAV TABS ═══════════ */}
      <nav className="px-6 flex items-center gap-1 border-b border-white/[0.04]">
        {tabs.map(t => {
          const active = tab === t.key;
          const Icon = t.icon;
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`relative px-4 py-3 text-xs font-semibold flex items-center gap-2 transition-colors ${
                active ? 'text-cyan-400' : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {t.label}
              {active && <div className="absolute bottom-0 left-2 right-2 h-[2px] rounded-full bg-gradient-to-r from-cyan-500 to-emerald-500" />}
            </button>
          );
        })}
      </nav>

      {/* ═══════════ CONTENT ═══════════ */}
      <main className="flex-1 p-5 overflow-hidden">

        {/* ─────── TAB: CONTROL ROOM ─────── */}
        {tab === 'dashboard' && (
          <div className="grid grid-cols-12 gap-4 h-full" style={{ minHeight: 'calc(100vh - 140px)' }}>

            {/* LEFT — Agent Feed */}
            <div className="col-span-3 flex flex-col gap-4">
              <Glass className="p-4 flex-1 flex flex-col min-h-0">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <Radio className="w-3.5 h-3.5 text-cyan-400" />
                    Agent Feed
                  </h2>
                  <span className="text-[10px] font-mono text-slate-500">{agents?.length ?? 0} active</span>
                </div>
                <div className="flex-1 overflow-y-auto space-y-2 pr-1 min-h-0">
                  {(agents ?? []).map(a => {
                    const rc = riskColor(a?.risk_level);
                    const isEgo = a?.agent_id === 0;
                    return (
                      <div
                        key={a?.agent_id ?? Math.random()}
                        className={`p-3 rounded-lg border transition-all duration-200 hover:border-cyan-500/30 hover:bg-cyan-500/[0.02] cursor-pointer ${
                          isEgo ? 'border-emerald-500/20 bg-emerald-500/[0.04]' : 'border-white/[0.06] bg-white/[0.02]'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1.5">
                          <span className={`text-xs font-bold ${isEgo ? 'text-emerald-400' : 'text-white'}`}>
                            {isEgo ? '◉ Ego Vehicle' : `Agent #${a?.agent_id ?? '?'}`}
                          </span>
                          <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-md ${rc.bg} ${rc.border} ${rc.text} border`}>
                            {(a?.risk_level ?? 'low').toUpperCase()}
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-[10px] text-slate-500">
                          <span className="font-mono">{a?.speed_mps ?? 0} m/s</span>
                          <span>·</span>
                          <span className="font-mono">{(a?.heading_deg ?? 0).toFixed(0)}°</span>
                          <span>·</span>
                          <span className="uppercase text-[8px] tracking-wider">{a?.category ?? 'unknown'}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </Glass>
            </div>

            {/* CENTER — Map */}
            <div className="col-span-6 flex flex-col gap-4">
              <Glass className="p-4 flex-1 flex flex-col min-h-0">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Ego-Centric BEV · 500m Crop
                  </h2>
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-emerald-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 hud-blink" />
                    {isPlaying ? `${currentFrame}/12` : '10Hz'}
                  </div>
                </div>

                <div className="flex-1 rounded-xl overflow-hidden border border-white/[0.06] relative min-h-0">
                  <Map
                    ref={mapRef}
                    initialViewState={{ longitude: mapCenter.lng, latitude: mapCenter.lat, zoom: 16 }}
                    style={{ width: '100%', height: '100%' }}
                    mapStyle={MAP_STYLE}
                    attributionControl={false}
                  >
                    {/* Trajectory polylines */}
                    {trajectoryGeoJSON && (
                      <>
                        {/* Ego trajectories (multiple modes) */}
                        {trajectoryGeoJSON.features
                          .filter((f: any) => f.properties.isEgo)
                          .map((f: any, i: number) => (
                            <Source key={`ego-traj-${i}`} id={`ego-traj-${i}`} type="geojson" data={{ type: 'FeatureCollection', features: [f] }}>
                              <Layer
                                id={`ego-traj-line-${i}`}
                                type="line"
                                paint={{
                                  'line-color': f.properties.color,
                                  'line-width': f.properties.width,
                                  'line-opacity': f.properties.opacity,
                                  'line-dasharray': f.properties.modeIdx > 0 ? [2, 2] : [1],
                                }}
                                layout={{ 'line-cap': 'round', 'line-join': 'round' }}
                              />
                            </Source>
                          ))
                        }
                        {/* Other agent trajectories */}
                        {trajectoryGeoJSON.features
                          .filter((f: any) => !f.properties.isEgo)
                          .map((f: any, i: number) => (
                            <Source key={`agent-traj-${i}`} id={`agent-traj-${i}`} type="geojson" data={{ type: 'FeatureCollection', features: [f] }}>
                              <Layer
                                id={`agent-traj-line-${i}`}
                                type="line"
                                paint={{
                                  'line-color': f.properties.color,
                                  'line-width': 2,
                                  'line-opacity': 0.5,
                                  'line-dasharray': [3, 3],
                                }}
                                layout={{ 'line-cap': 'round', 'line-join': 'round' }}
                              />
                            </Source>
                          ))
                        }
                      </>
                    )}

                    {/* Agent position markers */}
                    {liveAgentPositions.map((pos, i) => (
                      <Marker key={`agent-marker-${i}`} longitude={pos.lng} latitude={pos.lat} anchor="center">
                        <div
                          className={`rounded-full border-2 shadow-lg transition-all duration-300 ${
                            i === 0
                              ? 'w-4 h-4 bg-emerald-400 border-emerald-300 shadow-emerald-500/40'
                              : pos.type === 'pedestrian'
                                ? 'w-3 h-3 bg-amber-400 border-amber-300 shadow-amber-500/40'
                                : 'w-3.5 h-3.5 bg-rose-400 border-rose-300 shadow-rose-500/40'
                          }`}
                          style={{
                            boxShadow: i === 0
                              ? '0 0 12px rgba(16, 185, 129, 0.6)'
                              : '0 0 8px rgba(244, 63, 94, 0.4)',
                          }}
                        />
                      </Marker>
                    ))}
                  </Map>

                  {/* HUD Overlay */}
                  <div className="absolute top-0 left-0 right-0 z-10 flex justify-center pointer-events-none">
                    <div className="mt-3 px-4 py-1.5 rounded-full bg-black/60 backdrop-blur-xl border border-cyan-500/10">
                      <span className="text-[9px] font-mono font-bold text-cyan-400 tracking-[0.25em] uppercase hud-blink">
                        🛰️ LIVE GEOGRAPHIC ENVIRONMENT STREAM
                      </span>
                    </div>
                  </div>

                  {/* Trajectory legend */}
                  <div className="absolute bottom-4 left-4 z-10 pointer-events-none">
                    <div className="glass px-3 py-2 space-y-1">
                      {TRAJ_COLORS.slice(0, 3).map((c, i) => (
                        <div key={i} className="flex items-center gap-2 text-[9px] text-slate-400">
                          <span className="w-3 h-0.5 rounded-full" style={{ backgroundColor: c.hex }} />
                          {c.label}
                        </div>
                      ))}
                      <div className="flex items-center gap-2 text-[9px] text-slate-400">
                        <span className="w-2 h-2 rounded-full bg-rose-400" />
                        Other agents
                      </div>
                    </div>
                  </div>

                  {/* Bottom gradient */}
                  <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-[#060a10] to-transparent pointer-events-none" />
                </div>

                {/* Controls */}
                <div className="flex items-center justify-between mt-3">
                  <div className="flex gap-2">
                    <button
                      onClick={isPlaying ? stopPlayback : startPlayback}
                      className={`px-3 py-1.5 text-[10px] font-bold rounded-lg border transition-all flex items-center gap-1.5 ${
                        isPlaying
                          ? 'border-amber-500/30 bg-amber-500/10 text-amber-400 hover:bg-amber-500/20'
                          : 'border-white/[0.06] bg-white/[0.02] hover:border-cyan-500/30 hover:bg-cyan-500/[0.04]'
                      }`}
                    >
                      {isPlaying ? <Square className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                      {isPlaying ? 'Pause' : 'Play'}
                    </button>
                    <button
                      onClick={resetPlayback}
                      className="px-3 py-1.5 text-[10px] font-bold rounded-lg border border-white/[0.06] bg-white/[0.02] hover:border-cyan-500/30 hover:bg-cyan-500/[0.04] transition-all flex items-center gap-1.5"
                    >
                      <RotateCcw className="w-3 h-3" />
                      Reset
                    </button>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono text-slate-500">
                      Frame <span className="text-emerald-400 font-bold">{currentFrame}</span>/12
                    </span>
                    {/* Frame progress bar */}
                    <div className="w-20 h-1 rounded-full bg-white/[0.04] overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-500 transition-all duration-300"
                        style={{ width: `${(currentFrame / 12) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </Glass>
            </div>

            {/* RIGHT — HCP Metrics */}
            <div className="col-span-3 flex flex-col gap-4">
              {/* Pruning Cascade */}
              <Glass className="p-4 flex-1 flex flex-col min-h-0">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <TrendingDown className="w-3.5 h-3.5 text-cyan-400" />
                    Pruning Cascade
                  </h2>
                  <span className={`text-[8px] font-bold px-1.5 py-0.5 rounded-full border ${
                    isHcpLive
                      ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                      : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
                  }`}>
                    {isHcpLive ? 'LIVE' : 'MOCK'}
                  </span>
                </div>
                <div className="space-y-3 flex-1">
                  <PruneBar label="Raw Candidates" value={hcp?.raw_count ?? 128} total={hcp?.raw_count ?? 128} color="bg-slate-500" dotColor="bg-slate-500" />
                  <PruneBar label="KFF · Kinematic" value={hcp?.kff_count ?? 0} total={hcp?.raw_count || 1} color="bg-slate-400" dotColor="bg-slate-400" />
                  <PruneBar label="SRF · Spatial" value={hcp?.srf_count ?? 0} total={hcp?.raw_count || 1} color="bg-sky-500" dotColor="bg-sky-500" />
                  <PruneBar label="SCF · Social" value={hcp?.scf_count ?? 0} total={hcp?.raw_count || 1} color="bg-emerald-500" dotColor="bg-emerald-500" />
                </div>

                {/* Latency Ring */}
                <div className="mt-4 pt-4 border-t border-white/[0.04] flex items-center gap-4">
                  <div className="relative w-16 h-16 flex-shrink-0">
                    <svg className="w-full h-full -rotate-90" viewBox="0 0 36 36">
                      <circle cx="18" cy="18" r="15.9" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="2.5" />
                      <circle
                        cx="18" cy="18" r="15.9" fill="none"
                        stroke="url(#grad)" strokeWidth="2.5" strokeLinecap="round"
                        strokeDasharray={`${Math.min(((hcp?.total_time_ms ?? 32.5) / 120) * 100, 100)}, 100`}
                        className="transition-all duration-700"
                      />
                      <defs>
                        <linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
                          <stop offset="0%" stopColor="#06b6d4" />
                          <stop offset="100%" stopColor="#10b981" />
                        </linearGradient>
                      </defs>
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-sm font-black font-mono">{(hcp?.total_time_ms ?? 0).toFixed(0)}</span>
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Latency</div>
                    <div className="text-sm font-bold font-mono text-white">{(hcp?.total_time_ms ?? 0).toFixed(1)}ms</div>
                  </div>
                </div>
              </Glass>

              {/* Stats Row */}
              <div className="grid grid-cols-2 gap-3">
                <Glass className="p-3 text-center">
                  <Gauge className="w-4 h-4 text-emerald-400 mx-auto mb-1" />
                  <div className="text-lg font-black font-mono bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
                    {((hcp?.pruning_ratio ?? 0) * 100).toFixed(0)}%
                  </div>
                  <div className="text-[8px] text-slate-500 uppercase tracking-widest font-semibold mt-0.5">Pruned</div>
                </Glass>
                <Glass className="p-3 text-center">
                  <Timer className="w-4 h-4 text-sky-400 mx-auto mb-1" />
                  <div className="text-lg font-black font-mono bg-gradient-to-r from-sky-400 to-violet-400 bg-clip-text text-transparent">
                    {(hcp?.latency_reduction_pct ?? 0).toFixed(0)}%
                  </div>
                  <div className="text-[8px] text-slate-500 uppercase tracking-widest font-semibold mt-0.5">Faster</div>
                </Glass>
              </div>
            </div>
          </div>
        )}

        {/* ─────── TAB: 3D SCENE ─────── */}
        {tab === 'viewer3d' && (
          <Glass className="p-4 flex flex-col" style={{ minHeight: 'calc(100vh - 140px)' }}>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Box className="w-3.5 h-3.5 text-cyan-400" />
                3D Trajectory Ribbon Viewer
              </h2>
              <div className="flex items-center gap-4">
                {[
                  { color: 'bg-emerald-500', label: 'Ego Vehicle' },
                  { color: 'bg-cyan-500', label: 'Best Mode' },
                  { color: 'bg-sky-400', label: 'Alt Modes' },
                  { color: 'bg-rose-500', label: 'Other Agents' },
                ].map(c => (
                  <div key={c.label} className="flex items-center gap-1.5 text-[9px] text-slate-500">
                    <span className={`w-2 h-2 rounded-sm ${c.color}`} />
                    {c.label}
                  </div>
                ))}
              </div>
            </div>
            <div className="flex-1 rounded-xl overflow-hidden border border-white/[0.06] relative">
              <Scene3D scenario={scenarioData} />
              {/* Camera controls hint */}
              <div className="absolute bottom-4 right-4 z-10 glass px-3 py-1.5 pointer-events-none">
                <span className="text-[9px] font-mono text-slate-500">
                  🖱️ Orbit: Drag · Zoom: Scroll · Pan: Right-click
                </span>
              </div>
            </div>
          </Glass>
        )}

        {/* ─────── TAB: STATE EXPLAINER ─────── */}
        {tab === 'nlg' && (
          <Glass className="p-5" style={{ minHeight: 'calc(100vh - 140px)' }}>
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4 flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              Motion State Explainer
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-white/[0.06]">
                    {['Agent', 'Type', 'Speed', 'Heading', 'TTC', 'Risk', 'Explanation'].map(h => (
                      <th key={h} className="p-3 text-[10px] uppercase tracking-wider text-slate-500 font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(agents ?? []).map(a => {
                    const rc = riskColor(a?.risk_level);
                    return (
                      <tr key={a?.agent_id ?? Math.random()} className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors">
                        <td className="p-3 text-xs font-bold font-mono">#{a?.agent_id ?? 'N/A'}</td>
                        <td className="p-3 text-[10px] uppercase text-slate-500 tracking-wider">{a?.category ?? ''}</td>
                        <td className="p-3 text-xs font-mono">{(a?.speed_mps ?? 0).toFixed(1)} m/s</td>
                        <td className="p-3 text-xs font-mono">{(a?.heading_deg ?? 0).toFixed(0)}°</td>
                        <td className="p-3 text-xs font-mono">{(a?.ttc_seconds ?? -1) > 0 ? `${a.ttc_seconds.toFixed(1)}s` : '—'}</td>
                        <td className="p-3">
                          <span className={`text-[9px] font-bold px-2 py-0.5 rounded-md ${rc.bg} ${rc.border} ${rc.text} border`}>
                            {(a?.risk_level ?? 'low').toUpperCase()}
                          </span>
                        </td>
                        <td className="p-3 text-xs text-slate-400 italic max-w-xs">{a?.explanation ?? ''}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Glass>
        )}
      </main>
    </div>
  );
}
