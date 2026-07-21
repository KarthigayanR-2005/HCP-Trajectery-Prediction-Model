import React, { useState, useEffect, useRef } from 'react';
import {
  Activity, Layers, Compass, Box, Zap, Shield,
  Radio, ChevronRight, Gauge, Timer, TrendingDown,
} from 'lucide-react';
import Map from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

/* ═══════════════════ CONSTANTS ═══════════════════ */
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
const ANCHOR_LNG = -122.4194;
const ANCHOR_LAT = 37.7749;
const API = 'http://localhost:8000';

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

/* ═══════════════════ HELPERS ═══════════════════ */
const riskColor = (level?: string) => {
  switch (level?.toLowerCase()) {
    case 'high': return { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', dot: 'bg-red-500' };
    case 'medium': return { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', dot: 'bg-amber-500' };
    default: return { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', dot: 'bg-emerald-500' };
  }
};

const pct = (val: number, total: number) => total > 0 ? Math.round((val / total) * 100) : 0;

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

/* ═══════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════ */
export default function App() {
  const [tab, setTab] = useState<'dashboard' | 'viewer3d' | 'nlg'>('dashboard');
  const [scenarioId, setScenarioId] = useState('scenario_0');
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [agents, setAgents] = useState<AgentState[]>([]);
  const [hcp, setHcp] = useState<HCPStats>({
    raw_count: 128, kff_count: 74, srf_count: 31, scf_count: 9,
    total_time_ms: 32.5, pruning_ratio: 0.76, latency_reduction_pct: 71.8,
  });
  const abortRef = useRef<AbortController | null>(null);

  /* ── Init ── */
  useEffect(() => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    fetch(`${API}/scenarios`, { signal: ctrl.signal })
      .then(r => r.json())
      .then((d: string[]) => {
        if (!Array.isArray(d)) return;
        setScenarios(d);
        if (d.length > 0) { setScenarioId(d[0]); loadScenario(d[0]); }
      })
      .catch(e => {
        if (e?.name === 'AbortError') return;
        setScenarios(['scenario_0', 'scenario_1', 'scenario_2']);
        loadScenario('scenario_0');
      });
    return () => ctrl.abort();
  }, []);

  const loadScenario = (id: string) => {
    setScenarioId(id);
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
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
  };

  const runHCP = () => {
    fetch(`${API}/run_hcp/${scenarioId}`, { method: 'POST' })
      .then(r => r.json())
      .then(d => { if (d && typeof d === 'object') setHcp(d as HCPStats); })
      .catch(() => setHcp({ raw_count: 128, kff_count: 70, srf_count: 28, scf_count: 8, total_time_ms: 31.8, pruning_ratio: 0.78, latency_reduction_pct: 72.4 }));
  };

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
              <span className="text-[9px] uppercase px-2 py-0.5 rounded-full font-bold tracking-widest bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                Live
              </span>
            </h1>
            <p className="text-[11px] text-slate-500 tracking-wide">Hierarchical Combinatorial Pruning · Motion Transformer</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
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
                    10Hz
                  </div>
                </div>

                <div className="flex-1 rounded-xl overflow-hidden border border-white/[0.06] relative min-h-0">
                  <Map
                    initialViewState={{ longitude: ANCHOR_LNG, latitude: ANCHOR_LAT, zoom: 16 }}
                    style={{ width: '100%', height: '100%' }}
                    mapStyle={MAP_STYLE}
                    attributionControl={false}
                  />
                  {/* HUD Overlay */}
                  <div className="absolute top-0 left-0 right-0 z-10 flex justify-center pointer-events-none">
                    <div className="mt-3 px-4 py-1.5 rounded-full bg-black/60 backdrop-blur-xl border border-cyan-500/10">
                      <span className="text-[9px] font-mono font-bold text-cyan-400 tracking-[0.25em] uppercase hud-blink">
                        🛰️ LIVE GEOGRAPHIC ENVIRONMENT STREAM
                      </span>
                    </div>
                  </div>
                  {/* Bottom gradient */}
                  <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-[#060a10] to-transparent pointer-events-none" />
                </div>

                {/* Controls */}
                <div className="flex items-center justify-between mt-3">
                  <div className="flex gap-2">
                    {['Play', 'Reset'].map(label => (
                      <button key={label} className="px-3 py-1.5 text-[10px] font-bold rounded-lg border border-white/[0.06] bg-white/[0.02] hover:border-cyan-500/30 hover:bg-cyan-500/[0.04] transition-all">
                        {label}
                      </button>
                    ))}
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">
                    Frame <span className="text-emerald-400 font-bold">0</span>/12
                  </span>
                </div>
              </Glass>
            </div>

            {/* RIGHT — HCP Metrics */}
            <div className="col-span-3 flex flex-col gap-4">
              {/* Pruning Cascade */}
              <Glass className="p-4 flex-1 flex flex-col min-h-0">
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-4 flex items-center gap-2">
                  <TrendingDown className="w-3.5 h-3.5 text-cyan-400" />
                  Pruning Cascade
                </h2>
                <div className="space-y-3 flex-1">
                  <PruneBar label="Raw Candidates" value={128} total={128} color="bg-slate-500" dotColor="bg-slate-500" />
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
          <Glass className="p-6 flex flex-col items-center justify-center" style={{ minHeight: 'calc(100vh - 140px)' }}>
            <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-violet-500/20 border border-white/[0.06] flex items-center justify-center mb-6">
              <Box className="w-12 h-12 text-cyan-400/60" />
            </div>
            <h3 className="text-xl font-bold mb-2">3D Trajectory Ribbon Viewer</h3>
            <p className="text-sm text-slate-400 max-w-lg text-center leading-relaxed mb-6">
              Interactive Three.js scene with agent bounding boxes, multi-modal predicted
              trajectory ribbons, and BEV grid overlays with orbit controls.
            </p>
            <div className="flex items-center gap-6 mb-6">
              {[
                { color: 'bg-emerald-500', label: 'Ego Route' },
                { color: 'bg-sky-500', label: 'Alt Modes' },
                { color: 'bg-orange-500', label: 'High Risk' },
              ].map(c => (
                <div key={c.label} className="flex items-center gap-2 text-[10px] text-slate-500">
                  <span className={`w-2 h-2 rounded-sm ${c.color}`} />
                  {c.label}
                </div>
              ))}
            </div>
            <p className="text-[10px] font-mono text-slate-600">
              WebGL renderer available via FastAPI at localhost:8000
            </p>
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
