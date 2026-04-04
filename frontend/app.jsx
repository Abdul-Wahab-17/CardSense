const { useEffect, useMemo, useRef, useState } = React;

const THULLA_HAND_SIZE = 13;
const SUIT_ORDER = ["S", "H", "D", "C"];
const SUIT_LABEL = {
  S: "Spades",
  H: "Hearts",
  D: "Diamonds",
  C: "Clubs",
  OTHER: "Other",
};
const SUIT_CLASS = {
  S: "spades",
  H: "hearts",
  D: "diamonds",
  C: "clubs",
};

function suitOf(label) {
  if (!label || typeof label !== "string") return "OTHER";
  const s = label.trim().toUpperCase().slice(-1);
  return ["S", "H", "D", "C"].includes(s) ? s : "OTHER";
}

function cardTopLabel(det) {
  if (!det) return "-";
  return det.card_label_top1 || det.cls_name || "-";
}

function formatClock(tsMs) {
  if (!tsMs) return "-";
  return new Date(Number(tsMs)).toLocaleTimeString();
}

function App() {
  const [gameId, setGameId] = useState("default-game");
  const [cameraId, setCameraId] = useState("cam-1");
  const [cameraType, setCameraType] = useState("hand");
  const [leaderSeat, setLeaderSeat] = useState("me");
  const [toPlaySeat, setToPlaySeat] = useState("me");
  const [leadSuit, setLeadSuit] = useState("");
  const [cameraSource, setCameraSource] = useState("0");
  const [registerStatus, setRegisterStatus] = useState("");
  const [streamStatus, setStreamStatus] = useState("Register camera then start webcam stream.");
  const [fps, setFps] = useState(2);
  const [batchSize, setBatchSize] = useState(1);
  const [events, setEvents] = useState([]);
  const [hand, setHand] = useState(null);
  const [advice, setAdvice] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);
  const [manualCards, setManualCards] = useState("AS,10H,10D,10S,9C,9D,9H,8S,7S,6S,5S,4S,3S");
  const [isStreaming, setIsStreaming] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [backendOk, setBackendOk] = useState(false);
  const [lastEventAt, setLastEventAt] = useState(null);
  const [lastFrameSentAt, setLastFrameSentAt] = useState(null);
  const [framesSent, setFramesSent] = useState(0);
  const [lastError, setLastError] = useState("");
  const [handDebug, setHandDebug] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [gameState, setGameState] = useState(null);
  const [gameDiagnostics, setGameDiagnostics] = useState(null);
  const [beliefView, setBeliefView] = useState(null);
  const [simulationDebug, setSimulationDebug] = useState(null);
  const [mustFollowSuit, setMustFollowSuit] = useState(true);
  const [mustTrumpIfVoid, setMustTrumpIfVoid] = useState(false);
  const [trumpSuit, setTrumpSuit] = useState("");
  const [calibrationJson, setCalibrationJson] = useState('{"seat_regions":{"left":[[0,0],[320,720]],"across":[[321,0],[640,720]],"right":[[641,0],[960,720]],"me":[[961,0],[1280,720]]},"trick_region":[[300,220],[980,560]],"won_pile_regions":{"team_me":[[0,560],[640,720]],"team_opp":[[640,560],[1280,720]]},"table_homography":[],"occlusion_timeout_ms":2000}');

  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const streamTimerRef = useRef(null);
  const frameBatchRef = useRef([]);
  const captureCanvasRef = useRef(document.createElement("canvas"));

  const handCards = useMemo(() => hand?.cards || [], [hand]);

  const groupedHand = useMemo(() => {
    const groups = { S: [], H: [], D: [], C: [], OTHER: [] };
    for (const card of handCards) {
      groups[suitOf(card.label)].push(card);
    }
    return groups;
  }, [handCards]);

  const currentEvent = useMemo(() => {
    return events.find((item) => item.camera_id === cameraId.trim()) || events[0] || null;
  }, [events, cameraId]);

  const bestAction = recommendation?.best_action || null;

  const alternatives = useMemo(() => {
    const bestEv = Number(bestAction?.ev ?? 0);
    return (recommendation?.alternatives || []).slice(0, 4).map((alt) => {
      const ev = Number(alt?.ev ?? 0);
      return {
        card: alt?.card || "-",
        ev,
        delta: ev - bestEv,
      };
    });
  }, [recommendation, bestAction]);

  const pipelineStatus = useMemo(() => {
    if (!backendOk) return "Backend offline";
    if (!wsConnected) return "Realtime link disconnected";
    if (!isStreaming) return "Ready";
    if (!lastEventAt) return "Streaming, waiting for detections";
    if (Date.now() - lastEventAt > 15000) return "Streaming, worker may be stalled";
    return "Running";
  }, [backendOk, wsConnected, isStreaming, lastEventAt]);

  const freshness = useMemo(() => {
    const ts = Number(recommendation?.freshness_ts_ms || lastEventAt || 0);
    if (!ts) return { label: "No data", cls: "freshness-old" };
    const age = Date.now() - ts;
    if (age < 2500) return { label: "Fresh", cls: "freshness-fresh" };
    if (age < 6000) return { label: "Stale", cls: "freshness-stale" };
    return { label: "Old", cls: "freshness-old" };
  }, [recommendation, lastEventAt, events]);

  const lowConfidence = Number(bestAction?.confidence ?? 0) < 0.55;

  const legalPlays = Array.isArray(recommendation?.action_scores)
    ? recommendation.action_scores.length
    : null;

  async function apiFetch(url, options = {}) {
    return fetch(url, options);
  }

  async function registerCamera(event) {
    event.preventDefault();
    try {
      const res = await apiFetch("/api/cameras/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_id: cameraId.trim(), source: cameraSource.trim() }),
      });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus(`Registered ${cameraId.trim()}`);
    } catch (err) {
      setRegisterStatus(`Registration failed: ${err}`);
    }
  }

  async function loadInitialData() {
    try {
      const [eventsRes, handsRes] = await Promise.all([
        apiFetch("/api/detections/latest"),
        apiFetch("/api/hands/latest"),
      ]);

      if (eventsRes.ok) {
        const data = await eventsRes.json();
        const latest = (data.events || []).slice().reverse();
        setEvents((prev) => [...latest, ...prev].slice(0, 60));
        const debugEvent = latest.find((item) => item.camera_id === cameraId.trim()) || latest[0] || null;
        if (debugEvent?.debug) setDebugInfo(debugEvent.debug);
      }

      if (handsRes.ok) {
        const handData = await handsRes.json();
        const selected = (handData.hands || []).find((h) => h.camera_id === cameraId.trim()) || handData.hands?.[0] || null;
        setHand(selected);

        if (selected?.camera_id) {
          const singleHandRes = await apiFetch(`/api/hands/${selected.camera_id}`);
          if (singleHandRes.ok) {
            const singleHandData = await singleHandRes.json();
            setHandDebug(singleHandData.debug || null);
          }

          const adviceRes = await apiFetch(`/api/hands/${selected.camera_id}/advice`);
          if (adviceRes.ok) {
            const adviceData = await adviceRes.json();
            setAdvice(adviceData.advice || null);
          }
        }
      }
    } catch (_) {}
  }

  useEffect(() => {
    loadInitialData();

    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/ws/detections`);

    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data);
        setEvents((prev) => [event, ...prev].slice(0, 60));
        setLastEventAt(Date.now());
        setLastError("");
        if (event.hand_state) setHand(event.hand_state);
        if (event.advice) setAdvice(event.advice);
        if (event.hand_debug) setHandDebug(event.hand_debug);
        if (event.camera_id === cameraId.trim() && event.debug) setDebugInfo(event.debug);
      } catch (_) {}
    };

    ws.onopen = () => setWsConnected(true);

    ws.onerror = () => setLastError("WebSocket error.");

    ws.onclose = () => {
      setWsConnected(false);
      setTimeout(() => {
        if (!window.__cardsenseReloaded) {
          window.__cardsenseReloaded = true;
          location.reload();
        }
      }, 1000);
    };

    return () => {
      ws.close();
      stopWebcamStream();
    };
  }, []);

  useEffect(() => {
    let timer = null;
    const checkHealth = async () => {
      try {
        const res = await apiFetch("/api/health");
        if (!res.ok) throw new Error();
        const body = await res.json();
        setBackendOk(Boolean(body?.ok));
      } catch (_) {
        setBackendOk(false);
      }
    };

    checkHealth();
    timer = setInterval(checkHealth, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let timer = null;
    const loadRecommendation = async () => {
      const gid = gameId.trim() || "default-game";
      try {
        const [recRes, stateRes, diagRes, beliefRes, simRes] = await Promise.all([
          apiFetch(`/api/games/${encodeURIComponent(gid)}/recommendation`),
          apiFetch(`/api/games/${encodeURIComponent(gid)}/state`),
          apiFetch(`/api/games/${encodeURIComponent(gid)}/diagnostics`),
          apiFetch(`/api/games/${encodeURIComponent(gid)}/belief`),
          apiFetch(`/api/games/${encodeURIComponent(gid)}/simulation/debug?simulation_seed=1337&rollout_count=96`),
        ]);
        if (recRes.ok) setRecommendation(await recRes.json());
        if (stateRes.ok) {
          const gs = await stateRes.json();
          setGameState(gs.state || null);
        }
        if (diagRes.ok) setGameDiagnostics(await diagRes.json());
        if (beliefRes.ok) setBeliefView(await beliefRes.json());
        if (simRes.ok) setSimulationDebug(await simRes.json());
      } catch (_) {}
    };

    loadRecommendation();
    timer = setInterval(loadRecommendation, 2500);
    return () => clearInterval(timer);
  }, [gameId]);

  function frameToBase64(videoEl) {
    const width = videoEl.videoWidth;
    const height = videoEl.videoHeight;
    if (!width || !height) return null;

    const canvas = captureCanvasRef.current;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, width, height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
    return dataUrl.split(",")[1] || null;
  }

  async function flushBatch() {
    const batch = frameBatchRef.current;
    if (!batch.length) return;

    const payload = batch.length === 1 ? batch[0] : { frames: batch };
    const endpoint = batch.length === 1 ? "/api/frames" : "/api/frames/batch";

    try {
      const res = await apiFetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
    } finally {
      frameBatchRef.current = [];
    }
  }

  async function startWebcamStream() {
    const id = cameraId.trim();
    if (!id) {
      setStreamStatus("Set camera id first.");
      return;
    }

    if (mediaStreamRef.current) {
      setStreamStatus("Stream already running.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "environment",
        },
        audio: false,
      });
      mediaStreamRef.current = stream;
      const videoEl = videoRef.current;
      videoEl.srcObject = stream;
      await videoEl.play();

      const intervalMs = Math.round(1000 / Math.max(1, Math.min(4, Number(fps) || 2)));
      const batchLimit = Math.max(1, Math.min(2, Number(batchSize) || 1));
      setIsStreaming(true);
      setLastError("");
      setStreamStatus(`Streaming ${Math.round(1000 / intervalMs)} FPS, batch=${batchLimit}`);

      streamTimerRef.current = setInterval(async () => {
        const frameB64 = frameToBase64(videoEl);
        if (!frameB64) return;

        frameBatchRef.current.push({
          game_id: gameId.trim() || "default-game",
          camera_id: id,
          camera_type: cameraType,
          frame_id: `${id}:${Date.now()}:${Math.random().toString(16).slice(2, 8)}`,
          frame_b64: frameB64,
          ts_ms: Date.now(),
        });

        if (frameBatchRef.current.length >= batchLimit) {
          try {
            await flushBatch();
            setFramesSent((prev) => prev + batchLimit);
            setLastFrameSentAt(Date.now());
          } catch (err) {
            setLastError(String(err));
            setStreamStatus(`Streaming error: ${err}`);
          }
        }
      }, intervalMs);
    } catch (err) {
      setStreamStatus(`Cannot access webcam: ${err}`);
      mediaStreamRef.current = null;
      setIsStreaming(false);
    }
  }

  async function stopWebcamStream() {
    if (streamTimerRef.current) {
      clearInterval(streamTimerRef.current);
      streamTimerRef.current = null;
    }

    const mediaStream = mediaStreamRef.current;
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    try {
      await flushBatch();
    } catch (_) {}

    setIsStreaming(false);
    setStreamStatus("Stream stopped.");
  }

  async function applyManualHandOverride() {
    const cards = manualCards
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
      .slice(0, THULLA_HAND_SIZE);

    try {
      const res = await apiFetch(`/api/hands/${cameraId.trim()}/override`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cards }),
      });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus("Manual hand override applied.");
      await loadInitialData();
    } catch (err) {
      setRegisterStatus(`Manual override failed: ${err}`);
    }
  }

  async function clearManualHandOverride() {
    try {
      const res = await apiFetch(`/api/hands/${cameraId.trim()}/override`, { method: "DELETE" });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus("Manual hand override cleared.");
      await loadInitialData();
    } catch (err) {
      setRegisterStatus(`Clear override failed: ${err}`);
    }
  }

  async function startGame() {
    try {
      const gid = gameId.trim() || "default-game";
      const res = await apiFetch("/api/games/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          game_id: gid,
          leader: leaderSeat,
          to_play: toPlaySeat,
          must_follow_suit: mustFollowSuit,
          must_trump_if_void: mustTrumpIfVoid,
          trump_mode: "variant_config",
          trump_suit: trumpSuit || "",
          variant_name: "default",
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus(`Started game ${gid}`);
    } catch (err) {
      setRegisterStatus(`Start game failed: ${err}`);
    }
  }

  async function applyGameConfig() {
    try {
      const gid = gameId.trim() || "default-game";
      const res = await apiFetch(`/api/games/${encodeURIComponent(gid)}/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          leader: leaderSeat,
          to_play: toPlaySeat,
          lead_suit: leadSuit || "",
          must_follow_suit: mustFollowSuit,
          must_trump_if_void: mustTrumpIfVoid,
          trump_suit: trumpSuit || "",
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus(`Updated game config for ${gid}`);
    } catch (err) {
      setRegisterStatus(`Config failed: ${err}`);
    }
  }

  async function resetGame() {
    try {
      const gid = gameId.trim() || "default-game";
      const res = await apiFetch(`/api/games/${encodeURIComponent(gid)}/reset`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus(`Reset game ${gid}`);
    } catch (err) {
      setRegisterStatus(`Reset failed: ${err}`);
    }
  }

  async function applyTableCalibration() {
    try {
      const gid = gameId.trim() || "default-game";
      const payload = JSON.parse(calibrationJson);
      const res = await apiFetch(`/api/games/${encodeURIComponent(gid)}/calibration/table`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus(`Applied table calibration for ${gid}`);
    } catch (err) {
      setRegisterStatus(`Calibration failed: ${err}`);
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <p className="kicker">Thulla CV Assistant</p>
        <h1>CardSense Game Engine UI</h1>
        <p className="subtitle">Decision-first interface for hand + table camera signals, legal move filtering, and live recommendation guidance.</p>
      </header>

      <section className="status-strip" role="status" aria-live="polite">
        <div className={`status-chip ${pipelineStatus === "Running" ? "ok" : "warn"}`}>{pipelineStatus}</div>
        <div className={`status-chip ${backendOk ? "ok" : "bad"}`}>Backend {backendOk ? "OK" : "DOWN"}</div>
        <div className={`status-chip ${wsConnected ? "ok" : "bad"}`}>Realtime {wsConnected ? "Connected" : "Disconnected"}</div>
        <div className={`status-chip ${isStreaming ? "ok" : "warn"}`}>Stream {isStreaming ? "ON" : "OFF"}</div>
        <div className="status-chip">Model {currentEvent?.model_name || "-"}</div>
        <div className="status-chip">Detections {currentEvent?.detections?.length ?? 0}</div>
        <div className="status-chip">Last event {lastEventAt ? formatClock(lastEventAt) : "-"}</div>
        {lastError ? <div className="status-chip bad">Error {lastError}</div> : null}
      </section>

      <div className="workspace">
        <aside className="control-rail">
          <section className="panel">
            <h2 className="panel-title">Camera & Stream</h2>
            <form className="form-grid" onSubmit={registerCamera}>
              <label className="field">
                <span>Game ID</span>
                <input value={gameId} onChange={(e) => setGameId(e.target.value)} required />
              </label>
              <label className="field">
                <span>Camera ID</span>
                <input value={cameraId} onChange={(e) => setCameraId(e.target.value)} required />
              </label>
              <label className="field">
                <span>Camera Type</span>
                <select value={cameraType} onChange={(e) => setCameraType(e.target.value)}>
                  <option value="hand">hand</option>
                  <option value="table">table</option>
                </select>
              </label>
              <label className="field">
                <span>Source</span>
                <input value={cameraSource} onChange={(e) => setCameraSource(e.target.value)} placeholder="0 / rtsp://..." />
              </label>
              <button className="btn btn-primary" type="submit">Register Camera</button>
            </form>

            <div className="control-grid">
              <label className="field compact">
                <span>FPS</span>
                <input type="number" min="1" max="4" value={fps} onChange={(e) => setFps(Number(e.target.value))} />
              </label>
              <label className="field compact">
                <span>Batch</span>
                <input type="number" min="1" max="2" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
              </label>
            </div>

            <div className="button-row">
              <button className="btn btn-primary" type="button" onClick={startWebcamStream} disabled={isStreaming}>Start Stream</button>
              <button className="btn btn-secondary" type="button" onClick={stopWebcamStream} disabled={!isStreaming}>Stop</button>
            </div>

            <p className="inline-note">{streamStatus}</p>
            <p className="inline-note">{registerStatus}</p>
            <video id="local-preview" className="video-preview" ref={videoRef} autoPlay muted playsInline />
          </section>

          <section className="panel">
            <h2 className="panel-title">Game Controls</h2>
            <div className="form-grid">
              <label className="field">
                <span>Leader</span>
                <select value={leaderSeat} onChange={(e) => setLeaderSeat(e.target.value)}>
                  <option value="me">me</option>
                  <option value="left">left</option>
                  <option value="across">across</option>
                  <option value="right">right</option>
                </select>
              </label>
              <label className="field">
                <span>To Play</span>
                <select value={toPlaySeat} onChange={(e) => setToPlaySeat(e.target.value)}>
                  <option value="me">me</option>
                  <option value="left">left</option>
                  <option value="across">across</option>
                  <option value="right">right</option>
                </select>
              </label>
              <label className="field">
                <span>Lead Suit</span>
                <input value={leadSuit} onChange={(e) => setLeadSuit(e.target.value.toUpperCase())} placeholder="S/H/D/C" maxLength="1" />
              </label>
              <label className="field">
                <span>Trump Suit</span>
                <input value={trumpSuit} onChange={(e) => setTrumpSuit(e.target.value.toUpperCase())} placeholder="S/H/D/C" maxLength="1" />
              </label>
            </div>

            <div className="toggle-row">
              <label><input type="checkbox" checked={mustFollowSuit} onChange={(e) => setMustFollowSuit(e.target.checked)} /> Must follow suit</label>
              <label><input type="checkbox" checked={mustTrumpIfVoid} onChange={(e) => setMustTrumpIfVoid(e.target.checked)} /> Must trump if void</label>
            </div>

            <div className="button-row">
              <button className="btn btn-primary" type="button" onClick={startGame}>Start</button>
              <button className="btn btn-secondary" type="button" onClick={applyGameConfig}>Apply</button>
              <button className="btn btn-secondary" type="button" onClick={resetGame}>Reset</button>
            </div>
          </section>

          <details className="panel advanced-panel">
            <summary>Advanced Tools</summary>
            <label className="field">
              <span>Table Calibration JSON</span>
              <textarea value={calibrationJson} onChange={(e) => setCalibrationJson(e.target.value)} rows={5} />
            </label>
            <div className="button-row">
              <button className="btn btn-secondary" type="button" onClick={applyTableCalibration}>Apply Calibration</button>
            </div>

            <label className="field">
              <span>Manual Hand Override</span>
              <input value={manualCards} onChange={(e) => setManualCards(e.target.value)} placeholder="AS,10H,10D,..." />
            </label>
            <div className="button-row">
              <button className="btn btn-secondary" type="button" onClick={applyManualHandOverride}>Apply Manual Hand</button>
              <button className="btn btn-secondary" type="button" onClick={clearManualHandOverride}>Clear</button>
            </div>

            <details className="sub-debug">
              <summary>Advanced Diagnostics</summary>
              <pre>{JSON.stringify({ debugInfo, handDebug, gameDiagnostics }, null, 2)}</pre>
              <pre>{JSON.stringify({ beliefDiagnostics: beliefView?.diagnostics || {}, simulationTop: (simulationDebug?.action_scores || []).slice(0, 5) }, null, 2)}</pre>
            </details>
          </details>
        </aside>

        <section className="game-stage">
          <section className="panel recommendation-hero" aria-live="polite">
            <div className="panel-head-row">
              <h2 className="panel-title">Best Move</h2>
              <span className={`freshness-chip ${freshness.cls}`}>{freshness.label}</span>
            </div>

            {!recommendation ? (
              <p className="inline-note">No recommendation yet. Start stream and game setup first.</p>
            ) : (
              <>
                {lowConfidence ? <div className="confidence-warning">Low confidence recommendation. Play conservatively.</div> : null}
                <div className="best-card">
                  <div className="best-card-label">{bestAction?.card || "-"}</div>
                  <div className="best-card-meta">EV {bestAction?.ev ?? "-"} | Conf {bestAction?.confidence ?? "-"}</div>
                </div>
                <div className="badge-row">
                  <span className="badge">Mode {recommendation.recommendation_mode || "-"}</span>
                  <span className="badge">Legal plays {legalPlays ?? "-"}</span>
                  <span className="badge">Must follow {mustFollowSuit ? "Yes" : "No"}</span>
                  <span className="badge">Fresh at {formatClock(recommendation.freshness_ts_ms)}</span>
                </div>
                <ul className="clean-list">
                  {(recommendation.why || []).slice(0, 3).map((w, i) => (
                    <li key={`why-${i}`}>{w}</li>
                  ))}
                </ul>
                {alternatives.length > 0 ? (
                  <div className="alternatives">
                    {alternatives.map((alt, idx) => (
                      <div key={`${alt.card}-${idx}`} className="alt-row">
                        <span>{alt.card}</span>
                        <span>EV {alt.ev.toFixed(3)}</span>
                        <span>{alt.delta >= 0 ? "+" : ""}{alt.delta.toFixed(3)} vs best</span>
                      </div>
                    ))}
                  </div>
                ) : null}
              </>
            )}
          </section>

          <section className="panel">
            <div className="panel-head-row">
              <h2 className="panel-title">Your Hand</h2>
              <span className="panel-subtitle">{handCards.length}/{THULLA_HAND_SIZE} cards | {formatClock(hand?.updated_ts_ms)}</span>
            </div>
            <div className="suit-columns">
              {SUIT_ORDER.map((suit) => (
                <div className="suit-group" key={suit}>
                  <div className={`suit-title suit-${SUIT_CLASS[suit]}`}>{SUIT_LABEL[suit]}</div>
                  {(groupedHand[suit] || []).length === 0 ? (
                    <div className="inline-note">No cards</div>
                  ) : (
                    <div className="card-list">
                      {groupedHand[suit].map((card, idx) => (
                        <div className={`card-pill suit-${SUIT_CLASS[suit]}`} key={`${card.label}-${card.slot_id || idx}`}>
                          <span>{card.label}</span>
                          <span className="mono">{(Number(card.confidence || 0) * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <h2 className="panel-title">Table & Trick State</h2>
            {!gameState ? (
              <p className="inline-note">No table state yet.</p>
            ) : (
              <>
                <div className="badge-row">
                  <span className="badge">Leader {gameState.turn?.leader || "-"}</span>
                  <span className="badge">To play {gameState.turn?.to_play || "-"}</span>
                  <span className="badge">Lead suit {gameState.turn?.lead_suit || "-"}</span>
                  <span className="badge">Trick {gameState.trick_id ?? "-"}</span>
                </div>
                <div className="badge-row">
                  <span className="badge">Won team_me {(gameState.won_piles?.team_me || []).length}</span>
                  <span className="badge">Won team_opp {(gameState.won_piles?.team_opp || []).length}</span>
                </div>
                <div className="table-cards">
                  {(gameState.table_trick_cards || []).length === 0 ? (
                    <p className="inline-note">No table cards yet.</p>
                  ) : (
                    gameState.table_trick_cards.map((c, idx) => (
                      <div key={`${c.seat}-${c.card}-${idx}`} className="table-card-item">
                        <strong>{c.card}</strong>
                        <span>{c.seat}</span>
                        <span className="mono">{(Number(c.confidence || 0) * 100).toFixed(1)}%</span>
                      </div>
                    ))
                  )}
                </div>
              </>
            )}
          </section>

          <section className="panel">
            <h2 className="panel-title">Strategy Notes</h2>
            <p className="inline-note">{advice ? `${advice.summary} | discard candidates: ${(advice.discard_candidates || []).join(", ") || "N/A"}` : "No strategic advice yet."}</p>
            <ul className="clean-list">
              {(advice?.tips || []).map((tip, i) => (
                <li key={i}><strong>{tip.title}:</strong> {tip.detail}</li>
              ))}
            </ul>
            <p className="inline-note">Uncertainty: {(recommendation?.uncertainty_notes || []).join(" | ") || "none"}</p>
          </section>

          <section className="panel">
            <h2 className="panel-title">Activity Timeline</h2>
            <div className="timeline">
              {events.length === 0 ? (
                <p className="inline-note">No live events yet.</p>
              ) : (
                events.slice(0, 20).map((event, idx) => {
                  const top = (event.detections || []).slice(0, 3).map((d) => cardTopLabel(d)).join(", ");
                  return (
                    <div className="timeline-item" key={`${event.ts_ms}-${idx}`}>
                      <div className="timeline-main">
                        <span>{formatClock(event.ts_ms)}</span>
                        <span>cam {event.camera_id || "-"}</span>
                        <span>{event.detections?.length || 0} det</span>
                        <span>{top || "-"}</span>
                      </div>
                      <div className="timeline-meta">
                        <span>worker {event.worker_name || "-"}</span>
                        <span>latency {event.latency_ms ?? "-"}ms</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </section>
        </section>
      </div>

      <div className="best-mini" role="status" aria-live="polite">
        <span>Best</span>
        <strong>{bestAction?.card || "-"}</strong>
        <span>{bestAction?.confidence ? `${(Number(bestAction.confidence) * 100).toFixed(0)}%` : "-"}</span>
      </div>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
