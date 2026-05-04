/* CardSense — Blackjack HUD page (mobile-first redesign).
   - No auth. AUTH_ENABLED=0 server-side so requests work anonymously.
   - Sticky giant HUD on top, camera collapsible below.
   - Big touch-friendly buttons. Bottom sheet for settings/history/tipsi.
*/

const { useEffect, useRef, useState, useCallback } = React;

// ---------------- backend infra ----------------------------------------------

function backendHttpBase() {
  // Always use the same origin the page was served from.
  // (cloudflared tunnel maps /<*>  ->  api on 8000.)
  return `${location.protocol}//${location.host}`;
}
function backendWsBase() {
  return backendHttpBase().replace(/^http/, "ws");
}
function backendUrl(p) { return `${backendHttpBase()}${p}`; }

async function apiFetch(path, opts = {}) {
  const headers = new Headers(opts.headers || {});
  if (!headers.has("Content-Type") && opts.body) headers.set("Content-Type", "application/json");
  const res = await fetch(backendUrl(path), { ...opts, headers });
  let data = null;
  try { data = await res.json(); } catch (_) {}
  if (!res.ok) {
    const msg = (data && data.detail) ? data.detail : `${res.status} ${res.statusText}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

// ---------------- helpers -----------------------------------------------------

function totalLabel(cards) {
  if (!cards || cards.length === 0) return "0";
  let total = 0; let aces = 0;
  for (const r of cards) {
    if (r === "A") { total += 11; aces += 1; }
    else if (r === "K" || r === "Q" || r === "J" || r === "10") total += 10;
    else total += parseInt(r, 10) || 0;
  }
  while (total > 21 && aces > 0) { total -= 10; aces -= 1; }
  const isSoft = aces > 0 && total <= 21;
  if (isSoft && cards.length === 2 && total === 21) return "BJ";
  return isSoft ? `${total} (soft)` : `${total}`;
}

function outcomeColor(o) {
  if (!o) return "";
  if (o === "win" || o === "player_blackjack") return "bj-out-win";
  if (o === "lose" || o === "player_bust") return "bj-out-lose";
  if (o === "push") return "bj-out-push";
  return "bj-out-other";
}
function outcomePretty(o) {
  if (!o) return "";
  if (o === "player_blackjack") return "BLACKJACK!";
  if (o === "player_bust") return "BUST";
  if (o === "dealer_bust") return "DEALER BUST";
  return o.replace(/_/g, " ").toUpperCase();
}

// ---------------- ROI editor --------------------------------------------------

// Hardcoded ROIs with a small buffer band around y=0.5 — cards whose center
// lands in the no-man's-land are dropped (prevents the same physical card
// from being counted twice when its bbox wobbles across the boundary).
const DEFAULT_ROIS = {
  dealer: { x1: 0, y1: 0,    x2: 1, y2: 0.42 },
  player: { x1: 0, y1: 0.58, x2: 1, y2: 1.0  },
};

// ---------------- camera + uploader ------------------------------------------

function useCamera(videoRef) {
  const [error, setError] = useState("");
  const [streaming, setStreaming] = useState(false);

  const start = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) { setError("Camera API unavailable"); return; }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const v = videoRef.current;
      if (v) { v.srcObject = stream; await v.play(); }
      setStreaming(true); setError("");
    } catch (exc) { setError(exc.message || String(exc)); setStreaming(false); }
  }, [videoRef]);

  const stop = useCallback(() => {
    const v = videoRef.current;
    if (v && v.srcObject) {
      v.srcObject.getTracks().forEach(t => t.stop()); v.srcObject = null;
    }
    setStreaming(false);
  }, [videoRef]);

  return { error, streaming, start, stop };
}

function useFrameUploader({ enabled, videoRef, gameId, cameraId, fps }) {
  const [framesSent, setFramesSent] = useState(0);
  const [uploadError, setUploadError] = useState("");
  useEffect(() => {
    if (!enabled || !gameId || !cameraId) return;
    let cancelled = false;
    const intervalMs = Math.max(33, Math.round(1000 / Math.max(1, fps || 6)));
    const tick = async () => {
      if (cancelled) return;
      const v = videoRef.current;
      if (!v || v.readyState < 2) { setTimeout(tick, intervalMs); return; }
      try {
        const w = v.videoWidth, h = v.videoHeight;
        if (w <= 0 || h <= 0) { setTimeout(tick, intervalMs); return; }
        const c = document.createElement("canvas");
        c.width = w; c.height = h;
        c.getContext("2d").drawImage(v, 0, 0, w, h);
        const dataUrl = c.toDataURL("image/jpeg", 0.85);
        const b64 = dataUrl.split(",")[1] || "";
        await apiFetch("/api/frames/batch", {
          method: "POST",
          body: JSON.stringify({ frames: [{
            game_id: gameId, camera_id: cameraId, camera_type: "table",
            frame_id: `${cameraId}-${Date.now()}`, frame_b64: b64, ts_ms: Date.now(),
          }] }),
        });
        setFramesSent(n => n + 1); setUploadError("");
      } catch (exc) { setUploadError(exc.message || String(exc)); }
      setTimeout(tick, intervalMs);
    };
    setTimeout(tick, intervalMs);
    return () => { cancelled = true; };
  }, [enabled, videoRef, gameId, cameraId, fps]);
  return { framesSent, uploadError };
}

function useBlackjackWebsocket(gameId, onState) {
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    if (!gameId) return;
    const url = `${backendWsBase()}/ws/detections`;
    let ws = null, closed = false, retry = null;
    const connect = () => {
      try {
        ws = new WebSocket(url);
        ws.onopen = () => !closed && setConnected(true);
        ws.onclose = () => { setConnected(false); if (!closed) retry = setTimeout(connect, 1500); };
        ws.onerror = () => { try { ws.close(); } catch (_) {} };
        ws.onmessage = (evt) => {
          try {
            const d = JSON.parse(evt.data);
            if (d && d.type === "blackjack_state" && d.game_id === gameId) onState(d);
          } catch (_) {}
        };
      } catch (_) { retry = setTimeout(connect, 1500); }
    };
    connect();
    return () => { closed = true; if (retry) clearTimeout(retry); if (ws) ws.close(); };
  }, [gameId, onState]);
  return { connected };
}

// =============================================================================
// MAIN APP
// =============================================================================

function BlackjackApp() {
  const videoRef = useRef(null);

  const [gameId, setGameId] = useState(localStorage.getItem("bj_game_id") || "demo");
  const [cameraId] = useState("cam-bj"); // hidden — auto-managed
  const [bjState, setBjState] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [history, setHistory] = useState([]);
  const [tipsi, setTipsi] = useState("");
  const [tipsiLoading, setTipsiLoading] = useState(false);
  const [error, setError] = useState("");
  const [drawer, setDrawer] = useState(null);  // null | "settings" | "history" | "manual"
  const [cameraStarted, setCameraStarted] = useState(false);
  const [uploadEnabled, setUploadEnabled] = useState(false);
  const [cameraCollapsed, setCameraCollapsed] = useState(false);
  const fps = 5;

  const [agentLastAction, setAgentLastAction] = useState(null);
  const [agentLastError, setAgentLastError] = useState("");
  const [agentBusy, setAgentBusy] = useState(false);

  const cam = useCamera(videoRef);
  const { framesSent, uploadError } = useFrameUploader({
    enabled: uploadEnabled, videoRef, gameId, cameraId, fps,
  });

  // Agent uploader: 1 fps to /process_frame. Backend does cheap frame-diff
  // and only calls Claude when the scene changed. State drives the HUD.
  useEffect(() => {
    if (!uploadEnabled || !gameId) return;
    let cancelled = false;
    const intervalMs = 1000;
    const tick = async () => {
      if (cancelled) return;
      const v = videoRef.current;
      if (!v || v.readyState < 2) { setTimeout(tick, intervalMs); return; }
      try {
        const w = v.videoWidth, h = v.videoHeight;
        if (w <= 0 || h <= 0) { setTimeout(tick, intervalMs); return; }
        // 1024px @ 85% JPEG: above Claude's "high res" threshold, card corners
        // remain readable (~250px per card, ~40-50px corner indicators), and
        // image-token count drops ~2.4x vs 1568px for noticeably faster calls.
        const MAX_DIM = 1024;
        const scale = Math.min(1, MAX_DIM / Math.max(w, h));
        const dw = Math.round(w * scale), dh = Math.round(h * scale);
        const c = document.createElement("canvas");
        c.width = dw; c.height = dh;
        c.getContext("2d").drawImage(v, 0, 0, dw, dh);
        const dataUrl = c.toDataURL("image/jpeg", 0.85);
        const b64 = dataUrl.split(",")[1] || "";
        setAgentBusy(true);
        const result = await apiFetch(
          `/api/blackjack/games/${encodeURIComponent(gameId)}/process_frame`,
          { method: "POST", body: JSON.stringify({ frame_b64: b64 }) },
        );
        setAgentBusy(false);
        if (result?.actions?.length) {
          // Show the last tool the agent called
          const last = result.actions[result.actions.length - 1];
          setAgentLastAction({
            tool: last.tool,
            args: last.args,
            ok: !!last.result?.ok,
            error: last.result?.error,
            at: Date.now(),
          });
          if (last.result?.error) setAgentLastError(last.result.error);
          else setAgentLastError("");
        } else if (result?.skipped) {
          // No-op: scene unchanged. Don't update agentLastAction.
        }
      } catch (exc) {
        setAgentBusy(false);
        setAgentLastError(exc.message || String(exc));
      }
      setTimeout(tick, intervalMs);
    };
    setTimeout(tick, intervalMs);
    return () => { cancelled = true; };
  }, [uploadEnabled, gameId]);

  // Persist game id
  useEffect(() => { localStorage.setItem("bj_game_id", gameId); }, [gameId]);

  // ROIs are now hardcoded (top half = dealer, bottom half = player) and
  // posted to the backend as part of startGame(). No editor needed.

  // WS state stream -----------------------------------------------------
  const handleStateEvent = useCallback((evt) => {
    if (evt.state) setBjState(evt.state);
    setRecommendation(evt.recommendation || null);
  }, []);
  const ws = useBlackjackWebsocket(gameId, handleStateEvent);

  // ------------- API actions -----------------------------------------

  const refreshState = useCallback(async () => {
    try {
      const data = await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/state?llm=true`);
      setBjState(data.state); setRecommendation(data.recommendation || null);
    } catch (_) {}
  }, [gameId]);

  const refreshHistory = useCallback(async () => {
    try {
      const data = await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/history?limit=20`);
      setHistory(data.history || []);
    } catch (_) {}
  }, [gameId]);

  const startGame = useCallback(async () => {
    try {
      await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/start`, {
        method: "POST", body: JSON.stringify({ decks_in_shoe: 1, bet: 1 }),
      });
      try {
        await apiFetch("/api/cameras/register", {
          method: "POST", body: JSON.stringify({ camera_id: cameraId }),
        });
      } catch (_) {}
      try {
        await apiFetch(`/api/games/${encodeURIComponent(gameId)}/cameras/bind`, {
          method: "POST", body: JSON.stringify({ camera_id: cameraId, camera_type: "table" }),
        });
      } catch (_) {}
      // Set the default ROIs (top half = dealer, bottom half = player) so
      // the user never has to configure regions manually.
      try {
        await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/roi`, {
          method: "POST",
          body: JSON.stringify({
            player_roi: DEFAULT_ROIS.player,
            dealer_roi: DEFAULT_ROIS.dealer,
            auto_deal: true,
          }),
        });
      } catch (_) {}
      await refreshState(); await refreshHistory();
      setError("");
    } catch (exc) { setError(`Start: ${exc.message}`); }
  }, [gameId, cameraId, refreshState, refreshHistory]);

  const startGameRef = useRef(startGame);
  useEffect(() => { startGameRef.current = startGame; }, [startGame]);

  const newRound = useCallback(async () => {
    try {
      await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/new_round`, {
        method: "POST", body: JSON.stringify({ bet: 1 }),
      });
      setTipsi("");
      await refreshState();
    } catch (exc) { setError(`New round: ${exc.message}`); }
  }, [gameId, refreshState]);

  const playerStand = useCallback(async () => {
    try {
      await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/action`, {
        method: "POST", body: JSON.stringify({ action: "stand" }),
      });
      await refreshState(); await refreshHistory();
    } catch (exc) { setError(`Stand: ${exc.message}`); }
  }, [gameId, refreshState, refreshHistory]);

  const requestTipsi = useCallback(async () => {
    setTipsiLoading(true);
    try {
      const data = await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/tipsi`, { method: "POST" });
      setTipsi(data.review || data.reason || "(no review)");
      setDrawer("tipsi");
    } catch (exc) { setTipsi(`Error: ${exc.message}`); setDrawer("tipsi"); }
    finally { setTipsiLoading(false); }
  }, [gameId]);

  // Manual deal/hit/double (only used via drawer)
  const manualDeal = useCallback(async (p1, p2, d1, d2) => {
    try {
      await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/deal`, {
        method: "POST", body: JSON.stringify({ player_cards: [p1, p2], dealer_cards: [d1, d2] }),
      });
      await refreshState(); setDrawer(null);
    } catch (exc) { setError(`Deal: ${exc.message}`); }
  }, [gameId, refreshState]);

  const manualHit = useCallback(async (label) => {
    try {
      await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/action`, {
        method: "POST", body: JSON.stringify({ action: "hit", drawn_label: label }),
      });
      await refreshState();
    } catch (exc) { setError(`Hit: ${exc.message}`); }
  }, [gameId, refreshState]);

  const manualDealerDraw = useCallback(async (label) => {
    try {
      await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/observe`, {
        method: "POST", body: JSON.stringify({ label, target: "dealer" }),
      });
      await refreshState(); await refreshHistory();
    } catch (exc) { setError(`Dealer draw: ${exc.message}`); }
  }, [gameId, refreshState, refreshHistory]);

  // Initial mount: load existing game state, or auto-start if none exists.
  // Also defensively re-POSTs the default ROIs on every page load — needed
  // because in-memory state across API restarts can leave a game record
  // intact but ROIs cleared, which silently drops every detection.
  useEffect(() => {
    let mounted = true;
    (async () => {
      let needRoiSync = false;
      try {
        const data = await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/state`);
        if (!mounted) return;
        if (data?.state) {
          setBjState(data.state);
          setRecommendation(data.recommendation || null);
          const cfg = data.state.config || {};
          if (!cfg.player_roi || !cfg.dealer_roi) {
            needRoiSync = true;
          }
        } else {
          await startGame();
        }
      } catch (_) {
        try { await startGame(); } catch (_) {}
      }
      // Always make sure ROIs are present — defensive against any path that
      // could leave them null (manual reset, container restart, race, etc.)
      if (mounted && needRoiSync) {
        try {
          await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/roi`, {
            method: "POST",
            body: JSON.stringify({
              player_roi: DEFAULT_ROIS.player,
              dealer_roi: DEFAULT_ROIS.dealer,
              auto_deal: true,
            }),
          });
        } catch (_) {}
      }
      if (mounted) await refreshHistory();
    })();
    return () => { mounted = false; };
  }, [gameId, refreshHistory, startGame]);

  // Periodic health-check: every 5 seconds, re-fetch state. If the game
  // disappeared (e.g. backend restart wiped in-memory state), auto-recreate
  // it so the user doesn't have to manually refresh the page.
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const data = await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/state`);
        // If state exists but ROIs are null, defensively re-POST them
        const cfg = data?.state?.config || {};
        if (data?.state && (!cfg.player_roi || !cfg.dealer_roi)) {
          try {
            await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/roi`, {
              method: "POST",
              body: JSON.stringify({
                player_roi: DEFAULT_ROIS.player,
                dealer_roi: DEFAULT_ROIS.dealer,
                auto_deal: true,
              }),
            });
          } catch (_) {}
        }
      } catch (_) {
        // 404: game vanished. Recreate it.
        try { await startGame(); } catch (_) {}
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [gameId, startGame]);
  // Refresh history when round outcome flips terminal
  useEffect(() => {
    if (bjState && bjState.round_outcome &&
        ["win","lose","push","player_blackjack","player_bust"].includes(bjState.round_outcome)) {
      refreshHistory();
    }
  }, [bjState && bjState.round_outcome, refreshHistory]);

  // ---------------- derived view state -------------------------------------
  const playerCards = bjState?.player?.cards || [];
  const dealerCards = bjState?.dealer?.cards || [];
  const dealerHoleRevealed = !!bjState?.dealer_hole_revealed;
  const dealerVisible = dealerHoleRevealed ? dealerCards : (dealerCards[0] ? [dealerCards[0]] : []);
  const playerTotal = totalLabel(playerCards);
  const dealerTotal = totalLabel(dealerVisible);
  const roundOver = bjState?.round_outcome &&
    ["win","lose","push","player_blackjack","player_bust"].includes(bjState.round_outcome);

  const diag = recommendation?.diagnostics || null;
  const seenCount = bjState?.seen_cards?.length || 0;
  const totalCards = (bjState?.decks_in_shoe || 1) * 52;
  const remaining = bjState ? (totalCards - seenCount) : totalCards;
  const runningCount = diag?.running_count ?? 0;
  const trueCount = diag?.true_count ?? 0;

  const recAction = recommendation?.action || "";
  const actionClass = recAction === "stand" ? "bj-act-stand"
                    : recAction === "double" ? "bj-act-double"
                    : recAction === "hit" ? "bj-act-hit" : "";

  // ---------------- render --------------------------------------------------

  return (
    <div className="bj-app">
      {/* SLIM TOP STATUS STRIP */}
      <div className="bj-strip">
        <span className="bj-strip-game">{gameId}</span>
        <span className={`bj-strip-stat ${runningCount>0?"pos":runningCount<0?"neg":""}`}>
          RC <strong>{runningCount>0?`+${runningCount}`:runningCount}</strong>
        </span>
        <span className={`bj-strip-stat ${trueCount>0?"pos":trueCount<0?"neg":""}`}>
          TC <strong>{Number(trueCount).toFixed(1)}</strong>
        </span>
        <span className="bj-strip-stat">
          Deck <strong>{remaining}/{totalCards}</strong>
        </span>
        <span className="bj-strip-stat" title="Are ROIs configured on the backend?">
          ROI <strong style={{color: bjState?.config?.player_roi && bjState?.config?.dealer_roi ? "#7ef0a8" : "#ff8c8c"}}>
            {bjState?.config?.player_roi && bjState?.config?.dealer_roi ? "✓" : "✗"}
          </strong>
        </span>
        <span className={`bj-strip-conn ${ws.connected?"ok":""}`} title={ws.connected ? "live" : "offline"} />
      </div>

      {/* THE BIG HUD — the whole point of the page */}
      <div className={`bj-hud ${actionClass}`}>
        {recommendation ? (
          <>
            <div className="bj-hud-action">{recommendation.short_label}</div>
            <div className="bj-hud-confidence">
              {Math.round((recommendation.confidence||0)*100)}% confidence
              {recommendation.used_deviation && <span className="bj-tag">count override</span>}
            </div>
            {recommendation.llm_narration ? (
              <div className="bj-hud-llm">"{recommendation.llm_narration}"</div>
            ) : (
              <div className="bj-hud-reason">{recommendation.reason}</div>
            )}
            {diag && (
              <div className="bj-hud-stats">
                <span>Bust if hit: <strong>{Math.round((diag.bust_prob_on_hit||0)*100)}%</strong></span>
                <span>Dealer bust: <strong>{Math.round((diag.dealer_bust_prob||0)*100)}%</strong></span>
              </div>
            )}
          </>
        ) : roundOver ? (
          <>
            <div className="bj-hud-action bj-hud-outcome">
              {outcomePretty(bjState.round_outcome)}
            </div>
            <div className="bj-hud-confidence">Tap NEW ROUND below</div>
          </>
        ) : (
          <>
            <div className="bj-hud-action bj-hud-empty">WAITING</div>
            <div className="bj-hud-reason">
              {!bjState ? "Tap START GAME below" : "Place your cards on the table"}
            </div>
          </>
        )}
      </div>

      {/* HANDS */}
      <div className="bj-hands">
        <div className="bj-hand bj-hand-dealer">
          <div className="bj-hand-label">Dealer</div>
          <div className="bj-hand-cards">
            {dealerVisible.length === 0 && <span className="bj-hand-empty">—</span>}
            {dealerVisible.map((c, i) => <span key={i} className="bj-card">{c}</span>)}
            {(!dealerHoleRevealed && dealerCards.length === 2) && <span className="bj-card bj-card-back">?</span>}
          </div>
          <div className="bj-hand-total">{dealerTotal}</div>
        </div>
        <div className="bj-hand bj-hand-player">
          <div className="bj-hand-label">You</div>
          <div className="bj-hand-cards">
            {playerCards.length === 0 && <span className="bj-hand-empty">—</span>}
            {playerCards.map((c, i) => <span key={i} className="bj-card">{c}</span>)}
          </div>
          <div className="bj-hand-total">{playerTotal}</div>
        </div>
      </div>

      {/* CAMERA — collapsible */}
      <div className={`bj-cam-section ${cameraCollapsed ? "bj-cam-collapsed" : ""}`}>
        <div className="bj-cam-header" onClick={() => setCameraCollapsed(c => !c)}>
          <span>{cameraStarted ? "📷 Camera" : "📷 Camera (off)"}</span>
          <span className="bj-cam-arrow">{cameraCollapsed ? "▾" : "▴"}</span>
        </div>
        {!cameraCollapsed && (
          <div className="bj-cam-body">
            <div className="bj-cam-wrap">
              <video ref={videoRef} muted playsInline />
              {/* Static labeled regions — top half dealer, bottom half player */}
              <div className="bj-zone bj-zone-dealer"><span className="bj-zone-label">Dealer</span></div>
              <div className="bj-zone bj-zone-player"><span className="bj-zone-label">You</span></div>
              {!cameraStarted && (
                <div className="bj-cam-hint bj-cam-hint-center">
                  Tap <b>Start Camera</b> below
                </div>
              )}
            </div>
            <div className="bj-cam-controls">
              {!cam.streaming ? (
                <button className="bj-pill" onClick={async () => { await cam.start(); setCameraStarted(true); setUploadEnabled(true); }}>
                  Start Camera
                </button>
              ) : (
                <button className="bj-pill" onClick={() => { cam.stop(); setCameraStarted(false); setUploadEnabled(false); }}>
                  Stop Camera
                </button>
              )}
              <button className={`bj-pill ${uploadEnabled ? "bj-pill-on" : ""}`}
                disabled={!cam.streaming}
                onClick={() => setUploadEnabled(u => !u)}>
                {uploadEnabled ? `Streaming (${framesSent})` : "Streaming off"}
              </button>
            </div>
            {(cam.error || uploadError) && (
              <div className="bj-cam-err">{cam.error || uploadError}</div>
            )}
            {/* Agent activity strip */}
            {uploadEnabled && (
              <div className="bj-agent-strip">
                <span className={`bj-agent-dot ${agentBusy ? "bj-agent-busy" : ""}`} />
                <span className="bj-agent-text">
                  {agentBusy ? "Agent looking…" :
                   agentLastAction ? (
                     `Agent: ${agentLastAction.tool}` +
                     (agentLastAction.args && Object.keys(agentLastAction.args).length
                       ? ` (${JSON.stringify(agentLastAction.args)})` : "")
                   ) : "Agent ready"}
                </span>
                {agentLastError && (
                  <span className="bj-agent-err">{agentLastError}</span>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* BIG ACTION BAR — sticky bottom, this is what you tap during play */}
      <div className="bj-actions">
        <button className="bj-action bj-action-primary" onClick={startGame}>Start Game</button>
        <button className="bj-action" onClick={newRound} disabled={!bjState}>New Round</button>
        <button className="bj-action" onClick={playerStand} disabled={!bjState || roundOver}>Stand</button>
        <button className="bj-action bj-action-tipsi" onClick={requestTipsi}
          disabled={tipsiLoading || history.length === 0}>
          {tipsiLoading ? "..." : "💡 Tipsi"}
        </button>
        <button className="bj-action bj-action-more" onClick={() => setDrawer(drawer === "menu" ? null : "menu")}>
          ⋯ More
        </button>
      </div>

      {/* DRAWER (settings / history / manual / tipsi review) */}
      {drawer && (
        <div className="bj-drawer-bg" onClick={() => setDrawer(null)}>
          <div className="bj-drawer" onClick={e => e.stopPropagation()}>
            <button className="bj-drawer-close" onClick={() => setDrawer(null)}>×</button>
            {drawer === "menu" && (
              <>
                <h3>More</h3>
                <button className="bj-list-btn" onClick={async () => {
                  // Flip top/bottom — useful if phone orientation has dealer/player reversed
                  try {
                    await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/roi`, {
                      method: "POST",
                      body: JSON.stringify({
                        // Swap them: dealer now at bottom, player at top
                        player_roi: DEFAULT_ROIS.dealer,
                        dealer_roi: DEFAULT_ROIS.player,
                        auto_deal: true,
                      }),
                    });
                    setError("Zones swapped — dealer at bottom now");
                    setDrawer(null);
                  } catch (exc) { setError(`Swap failed: ${exc.message}`); }
                }}>🔄 Swap Dealer/Player zones</button>
                <button className="bj-list-btn" onClick={async () => {
                  // Reset to defaults
                  try {
                    await apiFetch(`/api/blackjack/games/${encodeURIComponent(gameId)}/roi`, {
                      method: "POST",
                      body: JSON.stringify({
                        player_roi: DEFAULT_ROIS.player,
                        dealer_roi: DEFAULT_ROIS.dealer,
                        auto_deal: true,
                      }),
                    });
                    setError("ROIs reset to default (dealer top, player bottom)");
                    setDrawer(null);
                  } catch (exc) { setError(`Reset failed: ${exc.message}`); }
                }}>↺ Reset zones to default</button>
                <button className="bj-list-btn" onClick={() => setDrawer("history")}>📜 History ({history.length})</button>
                <button className="bj-list-btn" onClick={() => setDrawer("manual")}>✋ Manual controls</button>
                <button className="bj-list-btn" onClick={() => setDrawer("settings")}>⚙ Settings</button>
                <button className="bj-list-btn" onClick={() => setDrawer("about")}>ℹ About</button>
              </>
            )}
            {drawer === "tipsi" && (
              <>
                <h3>Tipsi review</h3>
                <div className="bj-tipsi-text">{tipsi}</div>
              </>
            )}
            {drawer === "history" && (
              <>
                <h3>Hand history</h3>
                {history.length === 0 ? (
                  <div className="bj-hint">No hands played yet</div>
                ) : (
                  <ul className="bj-history">
                    {history.slice().reverse().map((h, i) => (
                      <li key={i}>
                        <span className="bj-h-cards">
                          {(h.player_cards||[]).join("")} ({h.player_total}) <small>vs</small>{" "}
                          {(h.dealer_cards||[]).join("")} ({h.dealer_total})
                        </span>
                        <span className={`bj-h-out ${outcomeColor(h.outcome)}`}>
                          {outcomePretty(h.outcome)}
                          {typeof h.payout === "number" && h.payout !== 0 && (
                            <span> ({h.payout > 0 ? "+" : ""}{h.payout.toFixed(1)})</span>
                          )}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            )}
            {drawer === "manual" && (
              <ManualPanel
                onDeal={manualDeal} onHit={manualHit} onDealerDraw={manualDealerDraw}
                hasGame={!!bjState} hasCards={playerCards.length > 0} />
            )}
            {drawer === "settings" && (
              <>
                <h3>Settings</h3>
                <label className="bj-setting">
                  Game ID
                  <input type="text" value={gameId} onChange={e => setGameId(e.target.value)} />
                </label>
                <p className="bj-hint">Frames stream at {fps} FPS · Camera ID auto-managed</p>
              </>
            )}
            {drawer === "about" && (
              <>
                <h3>About</h3>
                <p style={{lineHeight:1.5}}>
                  CardSense Blackjack HUD — distributed real-time card detection &
                  recommendation engine. Built for CS332 (PDC).
                </p>
                <p style={{opacity:.7,fontSize:13}}>
                  Hi-Lo card counting · Basic strategy + Illustrious 18 deviations ·
                  YOLOv8 card detection · Redis Streams worker pool ·
                  Claude Sonnet 4.6 (OpenRouter) for narration & post-hand coaching.
                </p>
              </>
            )}
          </div>
        </div>
      )}

      {error && <div className="bj-toast">{error} <button onClick={() => setError("")}>×</button></div>}
    </div>
  );
}

// Manual controls in the drawer ----------------------------------------------
function ManualPanel({ onDeal, onHit, onDealerDraw, hasGame, hasCards }) {
  const [p1, setP1] = useState(""); const [p2, setP2] = useState("");
  const [d1, setD1] = useState(""); const [d2, setD2] = useState("");
  const [hit, setHit] = useState(""); const [dealer, setDealer] = useState("");
  return (
    <>
      <h3>Manual controls</h3>
      <p className="bj-hint">Use these if camera detection isn't working</p>
      <div className="bj-mp-section">
        <div className="bj-mp-title">Deal initial 4 cards</div>
        <div className="bj-mp-row">
          <input placeholder="Player 1 (e.g. AS)" value={p1} onChange={e => setP1(e.target.value)} />
          <input placeholder="Player 2" value={p2} onChange={e => setP2(e.target.value)} />
        </div>
        <div className="bj-mp-row">
          <input placeholder="Dealer up" value={d1} onChange={e => setD1(e.target.value)} />
          <input placeholder="Dealer hole" value={d2} onChange={e => setD2(e.target.value)} />
        </div>
        <button className="bj-mp-btn" disabled={!hasGame || !(p1&&p2&&d1&&d2)}
          onClick={() => onDeal(p1,p2,d1,d2)}>Deal</button>
      </div>
      <div className="bj-mp-section">
        <div className="bj-mp-title">Player hit</div>
        <div className="bj-mp-row">
          <input placeholder="Card drawn" value={hit} onChange={e => setHit(e.target.value)} />
          <button className="bj-mp-btn" disabled={!hasCards || !hit} onClick={() => { onHit(hit); setHit(""); }}>Hit</button>
        </div>
      </div>
      <div className="bj-mp-section">
        <div className="bj-mp-title">Dealer draw</div>
        <div className="bj-mp-row">
          <input placeholder="Card drawn" value={dealer} onChange={e => setDealer(e.target.value)} />
          <button className="bj-mp-btn" disabled={!dealer} onClick={() => { onDealerDraw(dealer); setDealer(""); }}>Draw</button>
        </div>
      </div>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<BlackjackApp />);
