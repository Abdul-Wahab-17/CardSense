const { useEffect, useMemo, useRef, useState } = React;

const THULLA_HAND_SIZE = 13;

function App() {
  const [gameId, setGameId] = useState('default-game');
  const [cameraId, setCameraId] = useState('cam-1');
  const [cameraType, setCameraType] = useState('hand');
  const [leaderSeat, setLeaderSeat] = useState('me');
  const [toPlaySeat, setToPlaySeat] = useState('me');
  const [leadSuit, setLeadSuit] = useState('');
  const [cameraSource, setCameraSource] = useState('0');
  const [registerStatus, setRegisterStatus] = useState('');
  const [streamStatus, setStreamStatus] = useState('Register camera then start webcam stream.');
  const [fps, setFps] = useState(2);
  const [batchSize, setBatchSize] = useState(1);
  const [events, setEvents] = useState([]);
  const [hand, setHand] = useState(null);
  const [advice, setAdvice] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);
  const [manualCards, setManualCards] = useState('AS,10H,10D,10S,9C,9D,9H,8S,7S,6S,5S,4S,3S');
  const [isStreaming, setIsStreaming] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const [backendOk, setBackendOk] = useState(false);
  const [lastEventAt, setLastEventAt] = useState(null);
  const [lastFrameSentAt, setLastFrameSentAt] = useState(null);
  const [framesSent, setFramesSent] = useState(0);
  const [lastError, setLastError] = useState('');
  const [workerStatuses, setWorkerStatuses] = useState([]);
  const [workerLogs, setWorkerLogs] = useState([]);
  const [handDebug, setHandDebug] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [gameState, setGameState] = useState(null);
  const [gameDiagnostics, setGameDiagnostics] = useState(null);
  const [beliefView, setBeliefView] = useState(null);
  const [simulationDebug, setSimulationDebug] = useState(null);
  const [mustFollowSuit, setMustFollowSuit] = useState(true);
  const [mustTrumpIfVoid, setMustTrumpIfVoid] = useState(false);
  const [trumpSuit, setTrumpSuit] = useState('');
  const [calibrationJson, setCalibrationJson] = useState('{"seat_regions":{"left":[[0,0],[320,720]],"across":[[321,0],[640,720]],"right":[[641,0],[960,720]],"me":[[961,0],[1280,720]]},"trick_region":[[300,220],[980,560]],"won_pile_regions":{"team_me":[[0,560],[640,720]],"team_opp":[[640,560],[1280,720]]},"table_homography":[],"occlusion_timeout_ms":2000}');

  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const streamTimerRef = useRef(null);
  const frameBatchRef = useRef([]);
  const captureCanvasRef = useRef(document.createElement('canvas'));

  const handCards = useMemo(() => {
    const cards = hand?.cards || [];
    return Array.from({ length: THULLA_HAND_SIZE }, (_, i) => cards[i] || null);
  }, [hand]);

  const currentEvent = useMemo(() => {
    return events.find((item) => item.camera_id === cameraId.trim()) || events[0] || null;
  }, [events, cameraId]);

  const pipelineStatus = useMemo(() => {
    if (!backendOk) return 'Backend offline';
    if (!wsConnected) return 'Realtime link disconnected';
    if (!isStreaming) return 'Ready (stream stopped)';
    if (!lastEventAt) return 'Streaming, waiting for detections';
    if (Date.now() - lastEventAt > 15000) return 'Streaming, worker seems stalled';
    return 'Running';
  }, [backendOk, wsConnected, isStreaming, lastEventAt]);

  async function registerCamera(event) {
    event.preventDefault();
    try {
      const res = await fetch('/api/cameras/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
        fetch('/api/detections/latest'),
        fetch('/api/hands/latest'),
      ]);

      if (eventsRes.ok) {
        const data = await eventsRes.json();
        const latest = (data.events || []).slice().reverse();
        setEvents((prev) => [...latest, ...prev].slice(0, 40));
        const debugEvent = latest.find((item) => item.camera_id === cameraId.trim()) || latest[0] || null;
        if (debugEvent?.debug) {
          setDebugInfo(debugEvent.debug);
        }
      }

      if (handsRes.ok) {
        const handData = await handsRes.json();
        const selected = (handData.hands || []).find((h) => h.camera_id === cameraId.trim()) || handData.hands?.[0] || null;
        setHand(selected);
        if (selected?.camera_id) {
          const singleHandRes = await fetch(`/api/hands/${selected.camera_id}`);
          if (singleHandRes.ok) {
            const singleHandData = await singleHandRes.json();
            setHandDebug(singleHandData.debug || null);
          }

          const adviceRes = await fetch(`/api/hands/${selected.camera_id}/advice`);
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

    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${protocol}://${location.host}/ws/detections`);

    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data);
        setEvents((prev) => [event, ...prev].slice(0, 40));
        setLastEventAt(Date.now());
        setLastError('');
        if (event.hand_state) setHand(event.hand_state);
        if (event.advice) setAdvice(event.advice);
        if (event.hand_debug) setHandDebug(event.hand_debug);
        if (event.camera_id === cameraId.trim() && event.debug) setDebugInfo(event.debug);
      } catch (_) {}
    };

    ws.onopen = () => setWsConnected(true);

    ws.onerror = () => {
      setLastError('WebSocket error.');
    };

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
        const res = await fetch('/api/health');
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
      const gid = gameId.trim() || 'default-game';
      try {
        const [recRes, stateRes, diagRes, beliefRes, simRes] = await Promise.all([
          fetch(`/api/games/${encodeURIComponent(gid)}/recommendation`),
          fetch(`/api/games/${encodeURIComponent(gid)}/state`),
          fetch(`/api/games/${encodeURIComponent(gid)}/diagnostics`),
          fetch(`/api/games/${encodeURIComponent(gid)}/belief`),
          fetch(`/api/games/${encodeURIComponent(gid)}/simulation/debug?simulation_seed=1337&rollout_count=96`),
        ]);
        if (recRes.ok) {
          const rec = await recRes.json();
          setRecommendation(rec);
        }
        if (stateRes.ok) {
          const gs = await stateRes.json();
          setGameState(gs.state || null);
        }
        if (diagRes.ok) {
          const dg = await diagRes.json();
          setGameDiagnostics(dg);
        }
        if (beliefRes.ok) {
          const bv = await beliefRes.json();
          setBeliefView(bv);
        }
        if (simRes.ok) {
          const sv = await simRes.json();
          setSimulationDebug(sv);
        }
      } catch (_) {}
    };

    loadRecommendation();
    timer = setInterval(loadRecommendation, 2500);
    return () => clearInterval(timer);
  }, [gameId]);

  useEffect(() => {
    let timer = null;
    const loadWorkerDiagnostics = async () => {
      try {
        const [statusRes, logsRes] = await Promise.all([
          fetch('/api/workers/status'),
          fetch('/api/workers/logs?limit=60'),
        ]);

        if (statusRes.ok) {
          const data = await statusRes.json();
          setWorkerStatuses(data.workers || []);
        }

        if (logsRes.ok) {
          const data = await logsRes.json();
          setWorkerLogs((data.logs || []).slice(0, 60));
        }
      } catch (_) {}
    };

    loadWorkerDiagnostics();
    timer = setInterval(loadWorkerDiagnostics, 3000);
    return () => clearInterval(timer);
  }, []);

  function frameToBase64(videoEl) {
    const width = videoEl.videoWidth;
    const height = videoEl.videoHeight;
    if (!width || !height) return null;

    const canvas = captureCanvasRef.current;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, width, height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    return dataUrl.split(',')[1] || null;
  }

  async function flushBatch() {
    const batch = frameBatchRef.current;
    if (!batch.length) return;

    const payload = batch.length === 1 ? batch[0] : { frames: batch };
    const endpoint = batch.length === 1 ? '/api/frames' : '/api/frames/batch';

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
      setStreamStatus('Set camera id first.');
      return;
    }

    if (mediaStreamRef.current) {
      setStreamStatus('Stream already running.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'environment',
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
      setLastError('');
      setStreamStatus(`Streaming ${Math.round(1000 / intervalMs)} FPS, batch=${batchLimit}`);

      streamTimerRef.current = setInterval(async () => {
        const frameB64 = frameToBase64(videoEl);
        if (!frameB64) return;

        frameBatchRef.current.push({
          game_id: gameId.trim() || 'default-game',
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
    setStreamStatus('Stream stopped.');
  }

  async function applyManualHandOverride() {
    const cards = manualCards
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
      .slice(0, THULLA_HAND_SIZE);

    try {
      const res = await fetch(`/api/hands/${cameraId.trim()}/override`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cards }),
      });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus('Manual hand override applied.');
      await loadInitialData();
    } catch (err) {
      setRegisterStatus(`Manual override failed: ${err}`);
    }
  }

  async function clearManualHandOverride() {
    try {
      const res = await fetch(`/api/hands/${cameraId.trim()}/override`, { method: 'DELETE' });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus('Manual hand override cleared.');
      await loadInitialData();
    } catch (err) {
      setRegisterStatus(`Clear override failed: ${err}`);
    }
  }

  async function startGame() {
    try {
      const gid = gameId.trim() || 'default-game';
      const res = await fetch('/api/games/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          game_id: gid,
          leader: leaderSeat,
          to_play: toPlaySeat,
          must_follow_suit: mustFollowSuit,
          must_trump_if_void: mustTrumpIfVoid,
          trump_mode: 'variant_config',
          trump_suit: trumpSuit || '',
          variant_name: 'default',
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
      const gid = gameId.trim() || 'default-game';
      const res = await fetch(`/api/games/${encodeURIComponent(gid)}/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          leader: leaderSeat,
          to_play: toPlaySeat,
          lead_suit: leadSuit || '',
          must_follow_suit: mustFollowSuit,
          must_trump_if_void: mustTrumpIfVoid,
          trump_suit: trumpSuit || '',
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
      const gid = gameId.trim() || 'default-game';
      const res = await fetch(`/api/games/${encodeURIComponent(gid)}/reset`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus(`Reset game ${gid}`);
    } catch (err) {
      setRegisterStatus(`Reset failed: ${err}`);
    }
  }

  async function applyTableCalibration() {
    try {
      const gid = gameId.trim() || 'default-game';
      const payload = JSON.parse(calibrationJson);
      const res = await fetch(`/api/games/${encodeURIComponent(gid)}/calibration/table`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      setRegisterStatus(`Applied table calibration for ${gid}`);
    } catch (err) {
      setRegisterStatus(`Calibration failed: ${err}`);
    }
  }

  return (
    <main>
      <h1>CardSense Thulla Hand Monitor (React)</h1>

      <section className="status-bar">
        <div className={`status-chip ${pipelineStatus === 'Running' ? 'ok' : 'warn'}`}>{pipelineStatus}</div>
        <div className={`status-chip ${backendOk ? 'ok' : 'bad'}`}>Backend: {backendOk ? 'OK' : 'DOWN'}</div>
        <div className={`status-chip ${wsConnected ? 'ok' : 'bad'}`}>Realtime: {wsConnected ? 'Connected' : 'Disconnected'}</div>
        <div className={`status-chip ${isStreaming ? 'ok' : 'warn'}`}>Stream: {isStreaming ? 'ON' : 'OFF'}</div>
        <div className="status-chip">Model: {currentEvent?.model_name || '-'}</div>
        <div className="status-chip">Worker: {currentEvent?.worker_name || '-'}</div>
        <div className="status-chip">Detections: {currentEvent?.detections?.length ?? 0}</div>
        <div className="status-chip">Frames sent: {framesSent}</div>
        <div className="status-chip">Last frame: {lastFrameSentAt ? new Date(lastFrameSentAt).toLocaleTimeString() : '-'}</div>
        <div className="status-chip">Last event: {lastEventAt ? new Date(lastEventAt).toLocaleTimeString() : '-'}</div>
        {lastError ? <div className="status-chip bad">Error: {lastError}</div> : null}
      </section>

      <section className="card">
        <h2>Register Camera</h2>
        <form onSubmit={registerCamera}>
          <input
            placeholder="game id"
            value={gameId}
            onChange={(e) => setGameId(e.target.value)}
            required
          />
          <input
            placeholder="camera id"
            value={cameraId}
            onChange={(e) => setCameraId(e.target.value)}
            required
          />
          <select value={cameraType} onChange={(e) => setCameraType(e.target.value)}>
            <option value="hand">hand</option>
            <option value="table">table</option>
          </select>
          <input
            placeholder="source (rtsp/url/device)"
            value={cameraSource}
            onChange={(e) => setCameraSource(e.target.value)}
          />
          <button type="submit">Register</button>
        </form>
        <p>{registerStatus}</p>

        <div className="stream-controls">
          <label>
            FPS
            <input type="number" min="1" max="4" value={fps} onChange={(e) => setFps(Number(e.target.value))} />
          </label>
          <label>
            Batch
            <input type="number" min="1" max="2" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
          </label>
          <button type="button" onClick={startWebcamStream} disabled={isStreaming}>Start webcam stream</button>
          <button type="button" onClick={stopWebcamStream} disabled={!isStreaming}>Stop</button>
        </div>
        <p className="meta">{streamStatus}</p>
        <video id="local-preview" ref={videoRef} autoPlay muted playsInline />
      </section>

      <section className="card">
        <h2>Game Setup</h2>
        <div className="stream-controls">
          <label>
            Leader
            <select value={leaderSeat} onChange={(e) => setLeaderSeat(e.target.value)}>
              <option value="me">me</option>
              <option value="left">left</option>
              <option value="across">across</option>
              <option value="right">right</option>
            </select>
          </label>
          <label>
            To play
            <select value={toPlaySeat} onChange={(e) => setToPlaySeat(e.target.value)}>
              <option value="me">me</option>
              <option value="left">left</option>
              <option value="across">across</option>
              <option value="right">right</option>
            </select>
          </label>
          <label>
            Lead suit
            <input value={leadSuit} onChange={(e) => setLeadSuit(e.target.value.toUpperCase())} placeholder="S/H/D/C" maxLength="1" />
          </label>
          <label>
            Trump suit
            <input value={trumpSuit} onChange={(e) => setTrumpSuit(e.target.value.toUpperCase())} placeholder="S/H/D/C" maxLength="1" />
          </label>
          <label>
            <input type="checkbox" checked={mustFollowSuit} onChange={(e) => setMustFollowSuit(e.target.checked)} />
            Must follow suit
          </label>
          <label>
            <input type="checkbox" checked={mustTrumpIfVoid} onChange={(e) => setMustTrumpIfVoid(e.target.checked)} />
            Must trump if void
          </label>
          <button type="button" onClick={startGame}>Start Game</button>
          <button type="button" onClick={applyGameConfig}>Apply Config</button>
          <button type="button" onClick={resetGame}>Reset Game</button>
        </div>
        <textarea
          value={calibrationJson}
          onChange={(e) => setCalibrationJson(e.target.value)}
          rows={5}
          style={{ width: '100%', marginTop: 10, background: '#0f172a', color: '#e5e7eb' }}
        />
        <div className="stream-controls">
          <button type="button" onClick={applyTableCalibration}>Apply Table Calibration</button>
        </div>
      </section>

      <section className="card">
        <h2>Your Hand (Camera View)</h2>
        <p className="muted">This view assumes the camera only sees the cards in your hand. Max 13 cards are kept for Thulla.</p>
        <div className="meta">
          camera={hand?.camera_id || cameraId} cards={(hand?.cards || []).length}/{THULLA_HAND_SIZE}
          {' '}updated={hand?.updated_ts_ms ? new Date(hand.updated_ts_ms).toLocaleTimeString() : '-'}
        </div>
        <div className="hand-grid">
          {handCards.map((card, idx) => (
            <div key={idx} className={`hand-slot ${card ? 'filled' : 'empty'}`}>
              <div className="label">{card ? `${card.label}` : '—'}</div>
              <div className="meta">slot={card?.slot_id ?? idx + 1}</div>
              <div className="conf">{card ? `${(card.confidence * 100).toFixed(1)}%` : ''}</div>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Thulla Help</h2>
        <div className="meta">
          {advice ? `${advice.summary} | discard candidates: ${(advice.discard_candidates || []).join(', ') || 'N/A'}` : 'No advice yet.'}
        </div>
        <ul className="tips">
          {(advice?.tips || []).map((tip, i) => (
            <li key={i}><strong>{tip.title}:</strong> {tip.detail}</li>
          ))}
        </ul>
      </section>

      <section className="card">
        <h2>Move Recommendation</h2>
        {!recommendation ? (
          <div className="meta">No recommendation yet.</div>
        ) : (
          <>
            <div className="meta">
              mode={recommendation.recommendation_mode} best={recommendation.best_action?.card || '-'}
              {' '}ev={recommendation.best_action?.ev ?? '-'} conf={recommendation.best_action?.confidence ?? '-'}
            </div>
            <div className="meta">freshness={recommendation.freshness_ts_ms ? new Date(recommendation.freshness_ts_ms).toLocaleTimeString() : '-'}</div>
            <ul className="tips">
              {(recommendation.why || []).map((w, i) => <li key={`why-${i}`}>{w}</li>)}
            </ul>
            <div className="meta">uncertainty: {(recommendation.uncertainty_notes || []).join(' | ') || 'none'}</div>
            <pre>{JSON.stringify(recommendation.alternatives || [], null, 2)}</pre>
          </>
        )}
        {gameState ? (
          <div className="meta">
            turn={gameState.turn?.to_play || '-'} lead_suit={gameState.turn?.lead_suit || '-'} table_cards={(gameState.table_trick_cards || []).length}
          </div>
        ) : null}
      </section>

      <section className="card">
        <h2>Table State</h2>
        {!gameState ? (
          <div className="meta">No game state yet.</div>
        ) : (
          <>
            <div className="meta">trick_id={gameState.trick_id} leader={gameState.turn?.leader} to_play={gameState.turn?.to_play}</div>
            <div className="meta">won team_me={(gameState.won_piles?.team_me || []).length} team_opp={(gameState.won_piles?.team_opp || []).length}</div>
            <pre>{JSON.stringify(gameState.table_trick_cards || [], null, 2)}</pre>
          </>
        )}
      </section>

      <section className="card">
        <h2>Belief Panel</h2>
        {!beliefView ? (
          <div className="meta">No belief output yet.</div>
        ) : (
          <>
            <pre>{JSON.stringify(beliefView.top_hidden_by_seat || {}, null, 2)}</pre>
            <pre>{JSON.stringify(beliefView.diagnostics || {}, null, 2)}</pre>
          </>
        )}
      </section>

      <section className="card">
        <h2>Recommendation Explainability</h2>
        {!simulationDebug ? (
          <div className="meta">No simulation breakdown yet.</div>
        ) : (
          <pre>{JSON.stringify((simulationDebug.action_scores || []).slice(0, 5), null, 2)}</pre>
        )}
      </section>

      <section className="card">
        <h2>Transition/Conflict Diagnostics</h2>
        {!gameDiagnostics ? (
          <div className="meta">No diagnostics yet.</div>
        ) : (
          <pre>{JSON.stringify(gameDiagnostics, null, 2)}</pre>
        )}
      </section>

      <section className="card">
        <h2>Test Mode (Manual Hand)</h2>
        <p className="muted">Use this to test Thulla advice even if model detections are weak.</p>
        <input
          value={manualCards}
          onChange={(e) => setManualCards(e.target.value)}
          placeholder="AS,10H,10D,..."
          style={{ width: '100%' }}
        />
        <div className="stream-controls">
          <button type="button" onClick={applyManualHandOverride}>Apply Manual Hand</button>
          <button type="button" onClick={clearManualHandOverride}>Clear Manual Hand</button>
        </div>
      </section>

      <section className="card">
        <h2>Live Debug</h2>
        {!debugInfo ? (
          <div className="meta">No debug data yet. Start stream and wait for worker output.</div>
        ) : (
          <>
            <div className="meta">
              conf={debugInfo.conf_thres} debug_conf={debugInfo.debug_conf} selected_pass={String(debugInfo.selected_pass)}
            </div>
            <div className="meta">
              frame={debugInfo.quality?.width}x{debugInfo.quality?.height}
              {' '}brightness={debugInfo.quality?.brightness}
              {' '}blur_var={debugInfo.quality?.blur_var}
              {' '}age_ms={debugInfo.age_ms}
            </div>
            <pre>{JSON.stringify(debugInfo.passes || [], null, 2)}</pre>
            <pre>{JSON.stringify(debugInfo.probe_top || [], null, 2)}</pre>
          </>
        )}
        <h3>Hand Aggregation Debug</h3>
        {!handDebug ? (
          <div className="meta">No hand aggregation debug yet.</div>
        ) : (
          <pre>{JSON.stringify(handDebug, null, 2)}</pre>
        )}
      </section>

      <section className="card">
        <h2>Live Detections & Workers</h2>
        <div id="events">
          {events.map((event, idx) => (
            <div className="event" key={`${event.ts_ms}-${idx}`}>
              <div>
                Camera {event.camera_id} | {event.model_name} | worker={event.worker_name || '-'} | {event.detections?.length || 0} detections
              </div>
              <div className="meta">ts={event.ts_ms} latency={event.latency_ms ?? '-'}ms</div>
              <pre>{JSON.stringify(event.detections || [], null, 2)}</pre>
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <h2>Worker Status (Heartbeat)</h2>
        {workerStatuses.length === 0 ? (
          <div className="meta">No worker heartbeat yet.</div>
        ) : (
          workerStatuses.map((ws, idx) => (
            <div className="event" key={`${ws.worker_name || 'worker'}-${idx}`}>
              <div>
                <strong>{ws.worker_name || '-'}</strong> | model={ws.model_name || '-'} | resolved={ws.resolved_model || '-'}
              </div>
              <div className="meta">
                processed={ws.processed_frames || 0} detections={ws.total_detections || 0} with_detections={ws.frames_with_detections || 0}
                {' '}errors={ws.error_frames || 0} stale={ws.skipped_stale || 0} avg_latency={ws.avg_latency_ms || 0}ms
              </div>
              <div className="meta">
                card_like_classes={ws.model_summary?.card_like_count ?? '-'} of {ws.model_summary?.class_count ?? '-'}
                {' '}last_camera={ws.last_camera_id || '-'} updated={ws.updated_ts_ms ? new Date(Number(ws.updated_ts_ms)).toLocaleTimeString() : '-'}
              </div>
              {ws.last_error ? <div className="meta">last_error={ws.last_error}</div> : null}
            </div>
          ))
        )}
      </section>

      <section className="card">
        <h2>Worker Logs (Recent)</h2>
        {workerLogs.length === 0 ? (
          <div className="meta">No worker logs yet.</div>
        ) : (
          <div id="events">
            {workerLogs.map((log, idx) => (
              <div className="event" key={`${log.id || idx}`}>
                <div>
                  [{log.level || '-'}] {log.worker_name || '-'} — {log.message || '-'}
                </div>
                <div className="meta">{log.ts_ms ? new Date(Number(log.ts_ms)).toLocaleTimeString() : '-'}</div>
                <pre>{JSON.stringify(log, null, 2)}</pre>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
