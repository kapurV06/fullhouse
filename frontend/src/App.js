/* global THREE */
import { useState, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";

const FONTS = `
@import url('https://fonts.googleapis.com/css2?family=Permanent+Marker&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
`;

const HOUSE_QUOTES = [
  "It's never lupus.",
  "Everybody lies.",
  "When you hear hoofbeats, think horses. Unless you work here.",
  "I don't care about the patient, I care about the puzzle.",
  "Simplicity is the best disguise.",
  "Treating illness is why we became doctors.",
];

function ClusterGlobe({ visible, highlightCluster }) {
  const mountRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    if (!visible || !mountRef.current) return;
    if (typeof THREE === "undefined") return;

    const w = mountRef.current.clientWidth;
    const h = mountRef.current.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 1000);
    camera.position.z = 3.5;

    const geometry = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];
    const color = new THREE.Color();

    const clusterColors = [
      "#4ade80", "#60a5fa", "#a78bfa", "#f87171",
      "#fbbf24", "#34d399", "#f472b6", "#38bdf8",
    ];

    for (let c = 0; c < 80; c++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 1.2 + Math.random() * 0.8;
      const cx = r * Math.sin(phi) * Math.cos(theta);
      const cy = r * Math.sin(phi) * Math.sin(theta);
      const cz = r * Math.cos(phi);
      const clusterSize = Math.floor(Math.random() * 12) + 3;
      color.set(clusterColors[c % clusterColors.length]);

      for (let p = 0; p < clusterSize; p++) {
        positions.push(
          cx + (Math.random() - 0.5) * 0.15,
          cy + (Math.random() - 0.5) * 0.15,
          cz + (Math.random() - 0.5) * 0.15
        );
        const b = 0.4 + Math.random() * 0.4;
        colors.push(color.r * b, color.g * b, color.b * b);
      }
    }

    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({ size: 0.04, vertexColors: true, transparent: true, opacity: 0.85 });
    const points = new THREE.Points(geometry, material);
    scene.add(points);
    scene.add(new THREE.AmbientLight(0x1e2d3d, 2));

    let mouseX = 0, mouseY = 0;
    const onMouseMove = (e) => {
      const rect = mountRef.current?.getBoundingClientRect();
      if (!rect) return;
      mouseX = ((e.clientX - rect.left) / rect.width - 0.5) * 2;
      mouseY = -((e.clientY - rect.top) / rect.height - 0.5) * 2;
    };
    mountRef.current.addEventListener("mousemove", onMouseMove);

    const animate = () => {
      animRef.current = requestAnimationFrame(animate);
      points.rotation.y += 0.002 + mouseX * 0.001;
      points.rotation.x += 0.0005 + mouseY * 0.0005;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animRef.current);
      renderer.dispose();
      if (mountRef.current && renderer.domElement.parentNode === mountRef.current) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, [visible, highlightCluster]);

  return <div ref={mountRef} style={{ width: "100%", height: "100%", cursor: "crosshair" }} />;
}

function Heartbeat() {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 3 }}>
      {[0, 0.2, 0.4].map((d, i) => (
        <span key={i} style={{
          width: 4, height: 4, borderRadius: "50%", background: "#4ade80",
          animation: `hb 1.2s ease-in-out ${d}s infinite`,
        }} />
      ))}
    </span>
  );
}

function SymptomTag({ hpo_id, name, onRemove }) {
  return (
    <span style={{
      background: "#1e2d3d", border: "1px solid #2a4a6b", borderRadius: 4,
      padding: "4px 10px", fontSize: 11, color: "#93c5fd",
      fontFamily: "'IBM Plex Mono', monospace",
      display: "flex", alignItems: "center", gap: 6,
    }}>
      {hpo_id} · {name}
      <span onClick={() => onRemove(hpo_id)} style={{ color: "#4b5563", cursor: "pointer", fontSize: 10 }}>×</span>
    </span>
  );
}

function Whiteboard({ results, loading }) {
  if (loading) {
    return (
      <div style={{
        background: "#f5f0e8", borderRadius: 8, padding: "16px 20px",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontFamily: "'IBM Plex Mono', monospace", color: "#9ca3af", fontSize: 12,
        minHeight: 120,
      }}>
        <span style={{ animation: "pulse 1.5s ease-in-out infinite" }}>running differential...</span>
      </div>
    );
  }

  if (!results || results.length === 0) {
    return (
      <div style={{
        background: "#f5f0e8", borderRadius: 8, padding: "16px 20px", minHeight: 100,
        display: "flex", flexDirection: "column",
      }}>
        <div style={{ fontSize: 10, color: "#6b7280", letterSpacing: 2, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 10 }}>
          differential · whiteboard
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", flex: 1, fontFamily: "'Permanent Marker', cursive", fontSize: 16, color: "#d1d5db" }}>
          add symptoms to begin
        </div>
      </div>
    );
  }

  const top = results[0];
  const rest = results.slice(1);
  const maxScore = top.score || 0.001;

  return (
    <div style={{ background: "#f5f0e8", borderRadius: 8, padding: "16px 20px", position: "relative", overflowY: "auto", maxHeight: 400 }}>
      <div style={{
        position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
        fontSize: 10, color: "rgba(0,0,0,0.04)", letterSpacing: 3, whiteSpace: "nowrap",
        fontWeight: 600, pointerEvents: "none",
      }}>PRINCETON-PLAINSBORO TEACHING HOSPITAL</div>

      <div style={{ fontSize: 10, color: "#6b7280", letterSpacing: 2, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 12 }}>
        differential · whiteboard
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {rest.map((r) => (
          <div key={r.disease_id} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              fontFamily: "'Permanent Marker', cursive", fontSize: 14, color: "#9ca3af",
              flex: 1, minWidth: 0, textDecoration: "line-through",
              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}>{r.disease_name}</div>
            <div style={{ width: 60, height: 4, background: "#e5e7eb", borderRadius: 2, overflow: "hidden", flexShrink: 0 }}>
              <div style={{ height: "100%", background: "#d1d5db", borderRadius: 2, width: `${(r.score / maxScore) * 100}%` }} />
            </div>
            <div style={{ fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", color: "#9ca3af", width: 28, textAlign: "right", flexShrink: 0 }}>
              {r.n_matched}/{r.n_query}
            </div>
          </div>
        ))}

        <div style={{ marginTop: 6, paddingTop: 10, borderTop: "1px dashed #d1d5db", display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ fontFamily: "'Permanent Marker', cursive", fontSize: 17, color: "#1d4ed8", flex: 1, minWidth: 0, position: "relative" }}>
            <span style={{ display: "inline-block", position: "relative", padding: "2px 6px" }}>
              {top.disease_name}
              <svg style={{ position: "absolute", top: -4, left: -8, width: "calc(100% + 16px)", height: "calc(100% + 8px)", pointerEvents: "none", overflow: "visible" }}
                viewBox="0 0 100 100" preserveAspectRatio="none" fill="none">
                <ellipse cx="50" cy="50" rx="47" ry="44" stroke="#1d4ed8" strokeWidth="4" strokeDasharray="8 4" strokeLinecap="round" />
              </svg>
            </span>
          </div>
          <div style={{ width: 60, height: 4, background: "#e5e7eb", borderRadius: 2, overflow: "hidden", flexShrink: 0 }}>
            <div style={{ height: "100%", background: "#1d4ed8", borderRadius: 2, width: `${Math.max((top.n_matched / Math.max(top.n_query, 1)) * 100, 10)}%` }} />
          </div>
          <div style={{ fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: "#1d4ed8", fontWeight: 600, width: 28, textAlign: "right", flexShrink: 0 }}>
            {top.n_matched}/{top.n_query}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 10, fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: "#9ca3af", borderTop: "1px solid #e5e7eb", paddingTop: 8 }}>
        {top.disease_id} · cluster — {top.n_matched}/{top.n_query} symptoms matched
      </div>
    </div>
  );
}

function SHAPSection({ results }) {
  if (!results || results.length === 0) return null;
  const top = results[0];
  const symptoms = top.matched_symptoms.slice(0, 6);
  if (symptoms.length === 0) return null;
  const maxFreq = Math.max(...symptoms.map(s => s.disease_freq), 0.01);
  const colors = ["#3b82f6", "#818cf8", "#818cf8", "#a78bfa", "#a78bfa", "#c084fc"];

  return (
    <div style={{ background: "#111827", border: "1px solid #1e2d3d", borderRadius: 8, padding: "14px 16px" }}>
      <div style={{ fontSize: 10, color: "#4b5563", letterSpacing: 2, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 12 }}>
        symptom match · {top.disease_name}
      </div>
      {symptoms.map((s, i) => (
        <div key={s.hpo_id} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 7 }}>
          <div style={{ fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: "#93c5fd", width: 150, flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {s.name}
          </div>
          <div style={{ flex: 1, height: 5, background: "#1e2d3d", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ height: "100%", background: colors[i] || "#a78bfa", borderRadius: 3, width: `${(s.disease_freq / maxFreq) * 100}%` }} />
          </div>
          <div style={{ fontSize: 10, fontFamily: "'IBM Plex Mono', monospace", color: "#4b5563", width: 32, textAlign: "right" }}>
            {s.disease_freq.toFixed(2)}
          </div>
        </div>
      ))}
    </div>
  );
}

function TeamPanel({ team, loading, onRun }) {
  const members = [
    { key: "house", name: "Gregory House", role: "head of diagnostics", badge: "attending", isHouse: true },
    { key: "foreman", name: "Eric Foreman", role: "neurology", badge: "fellow", isHouse: false },
    { key: "chase", name: "Robert Chase", role: "general medicine", badge: "fellow", isHouse: false },
    { key: "cameron", name: "Allison Cameron", role: "immunology", badge: "fellow", isHouse: false },
  ];

  return (
    <div style={{ background: "#0d1117", display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div style={{ padding: "12px 16px 10px", fontSize: 10, color: "#4b5563", letterSpacing: 2, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace", borderBottom: "1px solid #1e2d3d", flexShrink: 0 }}>
        the team
      </div>

      <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
        {members.map(m => (
          <div key={m.key} style={{
            padding: "12px 14px", borderBottom: "1px solid #1a2535",
            borderLeft: m.isHouse ? "2px solid #f87171" : "2px solid transparent",
            background: m.isHouse ? "#0f1822" : "transparent",
          }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "#c9d1d9", marginBottom: 2, display: "flex", alignItems: "center", gap: 6 }}>
              {m.name}
              <span style={{
                fontSize: 9, padding: "1px 5px", borderRadius: 3,
                fontFamily: "'IBM Plex Mono', monospace", textTransform: "uppercase", letterSpacing: 1,
                background: m.isHouse ? "#2d1b1b" : "#1e2d3d",
                color: m.isHouse ? "#f87171" : "#a78bfa",
              }}>{m.badge}</span>
            </div>
            <div style={{ fontSize: 9, color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 6, textTransform: "uppercase", letterSpacing: 1 }}>
              {m.role}
            </div>
            <div style={{
              fontSize: 11, lineHeight: 1.6, wordBreak: "break-word",
              color: m.isHouse ? "#fca5a5" : "#9ca3af",
              fontStyle: m.isHouse ? "normal" : "italic",
            }}>
              {loading
                ? <span style={{ color: "#374151", fontStyle: "normal" }}>thinking...</span>
                : team?.[m.key]
                  ? `"${team[m.key]}"`
                  : <span style={{ color: "#374151", fontStyle: "normal" }}>awaiting differential</span>
              }
            </div>
          </div>
        ))}
      </div>

      <div onClick={onRun} style={{
        margin: "12px 14px", background: loading ? "#0d1117" : "#111827",
        border: "1px solid #1e2d3d", borderRadius: 6, padding: "10px 14px",
        color: loading ? "#374151" : "#4ade80",
        fontFamily: "'IBM Plex Mono', monospace", fontSize: 12,
        cursor: loading ? "not-allowed" : "pointer",
        textAlign: "center", letterSpacing: 1, flexShrink: 0,
      }}>
        {loading ? "running..." : "↗ run differential"}
      </div>
    </div>
  );
}

function HPOSearch({ onAdd, existing }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const debounceRef = useRef(null);

  const search = useCallback((q) => {
    if (!q || q.length < 2) { setResults([]); setOpen(false); return; }
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/hpo/search?q=${encodeURIComponent(q)}&limit=8`);
        const data = await res.json();
        setResults(data.results || []);
        setOpen(true);
      } catch { setResults([]); }
    }, 200);
  }, []);

  useEffect(() => { search(query); }, [query, search]);

  const handleAdd = (item) => {
    if (!existing.find(e => e.hpo_id === item.hpo_id)) onAdd(item);
    setQuery(""); setResults([]); setOpen(false);
  };

  return (
    <div style={{ position: "relative" }}>
      <input
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="add symptom or HPO term..."
        style={{
          background: "transparent", border: "none",
          borderTop: "1px solid #1e2d3d", paddingTop: 10,
          color: "#e8e8e8", fontSize: 13,
          fontFamily: "'IBM Plex Mono', monospace",
          width: "100%", outline: "none",
        }}
      />
      {open && results.length > 0 && (
        <div style={{
          position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0,
          background: "#111827", border: "1px solid #1e2d3d",
          borderRadius: 6, zIndex: 100, overflow: "hidden",
          maxHeight: 280, overflowY: "auto",
        }}>
          {results.map(r => (
            <div key={r.hpo_id} onClick={() => handleAdd(r)}
              style={{ padding: "8px 12px", cursor: "pointer", borderBottom: "1px solid #1e2d3d", display: "flex", gap: 8, alignItems: "center" }}
              onMouseEnter={e => e.currentTarget.style.background = "#1e2d3d"}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}
            >
              <span style={{ fontSize: 10, color: "#60a5fa", fontFamily: "'IBM Plex Mono', monospace", width: 90, flexShrink: 0 }}>{r.hpo_id}</span>
              <span style={{ fontSize: 12, color: "#e8e8e8" }}>{r.name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function FullHouse() {
  const [symptoms, setSymptoms] = useState([]);
  const [results, setResults] = useState(null);
  const [team, setTeam] = useState(null);
  const [houseQuote, setHouseQuote] = useState(HOUSE_QUOTES[0]);
  const [loading, setLoading] = useState(false);
  const [showGlobe, setShowGlobe] = useState(false);
  const [threeLoaded, setThreeLoaded] = useState(false);
  const [lupusEaster, setLupusEaster] = useState(false);

  useEffect(() => {
    if (window.THREE) { setThreeLoaded(true); return; }
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";
    script.onload = () => setThreeLoaded(true);
    document.head.appendChild(script);
  }, []);

  const removeSymptom = (hpo_id) => setSymptoms(s => s.filter(x => x.hpo_id !== hpo_id));
  const addSymptom = (item) => setSymptoms(s => [...s, item]);

  const runDiagnosis = async () => {
    if (symptoms.length === 0 || loading) return;
    setLoading(true); setResults(null); setTeam(null); setLupusEaster(false);
    try {
      const res = await fetch(`${API}/diagnose`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hpo_ids: symptoms.map(s => s.hpo_id), top_n: 10 }),
      });
      const data = await res.json();
      setResults(data.differential || []);
      setTeam(data.team || null);
      setHouseQuote(data.house_quote || HOUSE_QUOTES[0]);
      if (data.differential?.some(d => d.disease_name.toLowerCase().includes("lupus"))) setLupusEaster(true);
    } catch {
      setTeam({ house: "Backend unreachable. Is the server running on port 8000?", foreman: "", chase: "", cameron: "" });
    } finally { setLoading(false); }
  };

  return (
    <>
      <style>{`
        ${FONTS}
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #060a10; }
        @keyframes hb { 0%,80%,100%{opacity:.3;transform:scale(.8)} 40%{opacity:1;transform:scale(1)} }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
        @keyframes pulse-fill { 0%,100%{opacity:1} 50%{opacity:.7} }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #111827; }
        ::-webkit-scrollbar-thumb { background: #1e2d3d; border-radius: 2px; }
      `}</style>

      <div style={{ background: "#0d1117", color: "#e8e8e8", fontFamily: "'Inter', sans-serif", height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* TOP BAR */}
        <div style={{ background: "#111827", borderBottom: "1px solid #1e2d3d", padding: "10px 20px", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
          <div>
            <div style={{ fontFamily: "'Permanent Marker', cursive", fontSize: 20, color: "#e8e8e8", letterSpacing: 1 }}>
              FULL<span style={{ color: "#4ade80" }}>HOUSE</span>
            </div>
            <div style={{ fontSize: 9, color: "#4b5563", letterSpacing: 2, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace" }}>
              Princeton-Plainsboro · Rare Disease Dx Engine
            </div>
          </div>
          <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
            {[{ label: "diseases", value: "11,572" }, { label: "HPO terms", value: "11,557" }, { label: "clusters", value: "863", color: "#fbbf24" }].map(v => (
              <div key={v.label} style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                <div style={{ fontSize: 9, color: "#4b5563", letterSpacing: 1, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace" }}>{v.label}</div>
                <div style={{ fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", color: v.color || "#4ade80", fontWeight: 500 }}>{v.value}</div>
              </div>
            ))}
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <div style={{ fontSize: 9, color: "#4b5563", letterSpacing: 1, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace" }}>status</div>
              <div style={{ fontSize: 12, fontFamily: "'IBM Plex Mono', monospace", color: "#4ade80", display: "flex", alignItems: "center", gap: 6 }}>
                <Heartbeat /> LIVE
              </div>
            </div>
            <div onClick={() => setShowGlobe(g => !g)} style={{
              fontSize: 10, fontFamily: "'IBM Plex Mono', monospace",
              color: showGlobe ? "#4ade80" : "#4b5563", cursor: "pointer", letterSpacing: 1,
              border: `1px solid ${showGlobe ? "#4ade80" : "#1e2d3d"}`,
              padding: "4px 10px", borderRadius: 4, transition: "all 0.2s",
            }}>
              {showGlobe ? "← back" : "cluster view"}
            </div>
          </div>
        </div>

        {/* MAIN */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden", minHeight: 0 }}>

          {showGlobe ? (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", background: "#060a10" }}>
              <div style={{ padding: "10px 20px", borderBottom: "1px solid #1e2d3d", fontSize: 10, color: "#4b5563", letterSpacing: 2, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace" }}>
                phenotype cluster space · 863 clusters · UMAP 3D projection
              </div>
              <div style={{ flex: 1 }}>
                {threeLoaded
                  ? <ClusterGlobe visible={showGlobe} />
                  : <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12 }}>loading three.js...</div>
                }
              </div>
              <div style={{ padding: "8px 20px", background: "#0a0f16", borderTop: "1px solid #1e2d3d", fontSize: 10, color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace", letterSpacing: 1 }}>
                each point = a rare disease · colors = cluster groups · move mouse to steer
              </div>
            </div>
          ) : (
            <>
              {/* LEFT PANEL */}
              <div style={{ flex: 1, minWidth: 0, padding: 16, borderRight: "1px solid #1e2d3d", display: "flex", flexDirection: "column", gap: 12, overflow: "auto" }}>

                {/* Symptom input */}
                <div style={{ background: "#111827", border: "1px solid #1e2d3d", borderRadius: 8, padding: "12px 14px", flexShrink: 0 }}>
                  <div style={{ fontSize: 10, color: "#4b5563", letterSpacing: 2, textTransform: "uppercase", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 8 }}>
                    patient symptoms · HPO encoded
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: symptoms.length > 0 ? 8 : 0 }}>
                    {symptoms.map(s => <SymptomTag key={s.hpo_id} {...s} onRemove={removeSymptom} />)}
                  </div>
                  <HPOSearch onAdd={addSymptom} existing={symptoms} />
                </div>

                {/* Whiteboard */}
                <Whiteboard results={results} loading={loading} />

                {/* SHAP */}
                <SHAPSection results={results} />
              </div>

              {/* RIGHT PANEL */}
              <div style={{ width: 420, flexShrink: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>                <TeamPanel team={team} loading={loading} onRun={runDiagnosis} />
              </div>
            </>
          )}
        </div>

        {/* CONFIDENCE BAR */}
        {!showGlobe && (
          <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 20px", background: "#0a0f16", borderTop: "1px solid #1e2d3d", flexShrink: 0 }}>
            <div style={{ fontSize: 9, color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace", letterSpacing: 1, textTransform: "uppercase" }}>confidence</div>
            <div style={{ flex: 1, height: 3, background: "#1e2d3d", borderRadius: 2, overflow: "hidden" }}>
              <div style={{
                height: "100%",
                width: results ? `${Math.min((results[0]?.score || 0) * 500, 100)}%` : "0%",
                background: "linear-gradient(90deg, #4ade80, #22c55e)", borderRadius: 2,
                transition: "width 1s ease",
                animation: loading ? "pulse-fill 2s ease-in-out infinite" : "none",
              }} />
            </div>
            <div style={{ fontSize: 11, fontFamily: "'IBM Plex Mono', monospace", color: "#4ade80", fontWeight: 500 }}>
              {results ? `${results[0]?.n_matched}/${results[0]?.n_query} matched` : "—"}
            </div>
            <div style={{ width: 1, height: 14, background: "#1e2d3d", margin: "0 4px" }} />
            <div style={{ fontSize: 9, color: "#4b5563", fontFamily: "'IBM Plex Mono', monospace", letterSpacing: 1, textTransform: "uppercase" }}>
              orphanet · hpo · monarch
            </div>
          </div>
        )}

        {/* LUPUS EASTER EGG */}
        {lupusEaster && (
          <div style={{ padding: "3px 20px", background: "#2d1b1b", fontSize: 9, color: "#f87171", fontFamily: "'IBM Plex Mono', monospace", letterSpacing: 1, textTransform: "uppercase", flexShrink: 0 }}>
            ⚠ lupus detected in differential — it's never lupus
          </div>
        )}

        {/* QUOTE BAR */}
        <div style={{ background: "#111827", borderTop: "1px solid #1e2d3d", padding: "8px 20px", fontSize: 11, color: "#4b5563", fontStyle: "italic", fontFamily: "'IBM Plex Mono', monospace", display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
          <span style={{ color: "#1e2d3d" }}>—</span>
          <span>"{houseQuote}"</span>
        </div>
      </div>
    </>
  );
}