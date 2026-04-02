const { useEffect, useMemo, useRef, useState } = React;

const THULLA_HAND_SIZE = 13;

function App() {
  const [cameraId, setCameraId] = useState('cam-1');
  const [cameraSource, setCameraSource] = useState('0');
  const [registerStatus, setRegisterStatus] = useState('');
  const [streamStatus, setStreamStatus] = useState('Register camera then start webcam stream.');
  const [fps, setFps] = useState(10);
  const [batchSize, setBatchSize] = useState(4);
  const [events, setEvents] = useState([]);
  const [hand, setHand] = useState(null);
  const [advice, setAdvice] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const streamTimerRef = useRef(null);
  const frameBatchRef = useRef([]);
  const captureCanvasRef = useRef(document.createElement('canvas'));

  const handCards = useMemo(() => {
    const cards = hand?.cards || [];
    return Array.from({ length: THULLA_HAND_SIZE }, (_, i) => cards[i] || null);
  }, [hand]);

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
        if (event.hand_state) setHand(event.hand_state);
        if (event.advice) setAdvice(event.advice);
        if (event.camera_id === cameraId.trim() && event.debug) setDebugInfo(event.debug);
      } catch (_) {}
    };

    ws.onclose = () => {
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

      const intervalMs = Math.round(1000 / Math.max(1, Math.min(30, Number(fps) || 10)));
      const batchLimit = Math.max(1, Math.min(16, Number(batchSize) || 4));
      setIsStreaming(true);
      setStreamStatus(`Streaming ${Math.round(1000 / intervalMs)} FPS, batch=${batchLimit}`);

      streamTimerRef.current = setInterval(async () => {
        const frameB64 = frameToBase64(videoEl);
        if (!frameB64) return;

        frameBatchRef.current.push({
          camera_id: id,
          frame_b64: frameB64,
          ts_ms: Date.now(),
        });

        if (frameBatchRef.current.length >= batchLimit) {
          try {
            await flushBatch();
          } catch (err) {
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

  return (
    <main>
      <h1>CardSense Thulla Hand Monitor (React)</h1>

      <section className="card">
        <h2>Register Camera</h2>
        <form onSubmit={registerCamera}>
          <input
            placeholder="camera id"
            value={cameraId}
            onChange={(e) => setCameraId(e.target.value)}
            required
          />
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
            <input type="number" min="1" max="30" value={fps} onChange={(e) => setFps(Number(e.target.value))} />
          </label>
          <label>
            Batch
            <input type="number" min="1" max="16" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} />
          </label>
          <button type="button" onClick={startWebcamStream} disabled={isStreaming}>Start webcam stream</button>
          <button type="button" onClick={stopWebcamStream} disabled={!isStreaming}>Stop</button>
        </div>
        <p className="meta">{streamStatus}</p>
        <video id="local-preview" ref={videoRef} autoPlay muted playsInline />
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
              <div className="label">{card ? card.label : '—'}</div>
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
    </main>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
