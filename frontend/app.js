const registerForm = document.getElementById('register-form');
const cameraIdInput = document.getElementById('camera-id');
const cameraSourceInput = document.getElementById('camera-source');
const registerStatus = document.getElementById('register-status');
const eventsEl = document.getElementById('events');
const handCardsEl = document.getElementById('hand-cards');
const handSummaryEl = document.getElementById('hand-summary');
const adviceSummaryEl = document.getElementById('advice-summary');
const adviceTipsEl = document.getElementById('advice-tips');
const startStreamBtn = document.getElementById('start-stream');
const stopStreamBtn = document.getElementById('stop-stream');
const streamFpsInput = document.getElementById('stream-fps');
const streamBatchInput = document.getElementById('stream-batch');
const streamStatusEl = document.getElementById('stream-status');
const localPreview = document.getElementById('local-preview');

const THULLA_HAND_SIZE = 13;
let mediaStream = null;
let streamTimer = null;
let frameBatch = [];
const captureCanvas = document.createElement('canvas');

function updateStreamStatus(text) {
  streamStatusEl.textContent = text;
}

function frameToBase64(videoEl) {
  const width = videoEl.videoWidth;
  const height = videoEl.videoHeight;
  if (!width || !height) return null;

  captureCanvas.width = width;
  captureCanvas.height = height;
  const ctx = captureCanvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, width, height);
  const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.72);
  return dataUrl.split(',')[1] || null;
}

async function flushBatch() {
  if (!frameBatch.length) return;

  const payload = frameBatch.length === 1 ? frameBatch[0] : { frames: frameBatch };
  const endpoint = frameBatch.length === 1 ? '/api/frames' : '/api/frames/batch';

  try {
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const details = await res.text();
      throw new Error(details || 'upload failed');
    }
  } finally {
    frameBatch = [];
  }
}

async function startWebcamStream() {
  const cameraId = cameraIdInput.value.trim();
  if (!cameraId) {
    updateStreamStatus('Set camera id first.');
    return;
  }

  if (mediaStream) {
    updateStreamStatus('Stream already running.');
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    localPreview.srcObject = mediaStream;
    await localPreview.play();

    startStreamBtn.disabled = true;
    stopStreamBtn.disabled = false;

    const fps = Math.max(1, Math.min(30, Number(streamFpsInput.value) || 10));
    const batchSize = Math.max(1, Math.min(16, Number(streamBatchInput.value) || 4));
    const intervalMs = Math.round(1000 / fps);
    updateStreamStatus(`Streaming ${fps} FPS, batch=${batchSize}`);

    streamTimer = setInterval(async () => {
      if (!mediaStream) return;
      const frame_b64 = frameToBase64(localPreview);
      if (!frame_b64) return;

      frameBatch.push({
        camera_id: cameraId,
        frame_b64,
        ts_ms: Date.now(),
      });

      if (frameBatch.length >= batchSize) {
        try {
          await flushBatch();
        } catch (err) {
          updateStreamStatus(`Streaming error: ${err}`);
        }
      }
    }, intervalMs);
  } catch (err) {
    updateStreamStatus(`Cannot access webcam: ${err}`);
    mediaStream = null;
  }
}

async function stopWebcamStream() {
  if (streamTimer) {
    clearInterval(streamTimer);
    streamTimer = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  localPreview.srcObject = null;
  startStreamBtn.disabled = false;
  stopStreamBtn.disabled = true;

  try {
    await flushBatch();
  } catch (_) {}

  updateStreamStatus('Stream stopped.');
}

function renderAdvice(advice) {
  if (!advice) {
    adviceSummaryEl.textContent = 'No advice yet.';
    adviceTipsEl.innerHTML = '';
    return;
  }

  const discard = (advice.discard_candidates || []).join(', ') || 'N/A';
  adviceSummaryEl.textContent = `${advice.summary} | discard candidates: ${discard}`;

  adviceTipsEl.innerHTML = '';
  for (const tip of advice.tips || []) {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${tip.title}:</strong> ${tip.detail}`;
    adviceTipsEl.appendChild(li);
  }
}

function renderHand(handState, fallbackCameraId) {
  const cards = handState?.cards || [];
  const cameraId = handState?.camera_id || fallbackCameraId || '-';
  const updated = handState?.updated_ts_ms ? new Date(handState.updated_ts_ms).toLocaleTimeString() : '-';

  handSummaryEl.textContent = `camera=${cameraId} cards=${cards.length}/${THULLA_HAND_SIZE} updated=${updated}`;
  handCardsEl.innerHTML = '';

  for (let i = 0; i < THULLA_HAND_SIZE; i++) {
    const card = cards[i];
    const slot = document.createElement('div');
    slot.className = `hand-slot ${card ? 'filled' : 'empty'}`;

    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = card ? card.label : '—';

    const conf = document.createElement('div');
    conf.className = 'conf';
    conf.textContent = card ? `${(card.confidence * 100).toFixed(1)}%` : '';

    slot.appendChild(label);
    slot.appendChild(conf);
    handCardsEl.appendChild(slot);
  }
}

function addEvent(event) {
  const wrapper = document.createElement('div');
  wrapper.className = 'event';

  const title = document.createElement('div');
  title.textContent = `Camera ${event.camera_id} | ${event.model_name} | worker=${event.worker_name ?? '-'} | ${event.detections.length} detections`;

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = `ts=${event.ts_ms} latency=${event.latency_ms ?? '-'}ms`;

  const details = document.createElement('pre');
  details.textContent = JSON.stringify(event.detections, null, 2);

  wrapper.appendChild(title);
  wrapper.appendChild(meta);
  wrapper.appendChild(details);

  eventsEl.prepend(wrapper);

  while (eventsEl.children.length > 40) {
    eventsEl.removeChild(eventsEl.lastChild);
  }

  if (event.hand_state) {
    renderHand(event.hand_state, event.camera_id);
  }
  if (event.advice) {
    renderAdvice(event.advice);
  }
}

registerForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const payload = {
    camera_id: cameraIdInput.value.trim(),
    source: cameraSourceInput.value.trim(),
  };

  try {
    const res = await fetch('/api/cameras/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(await res.text());
    }

    registerStatus.textContent = `Registered ${payload.camera_id}`;
  } catch (err) {
    registerStatus.textContent = `Registration failed: ${err}`;
  }
});

startStreamBtn.addEventListener('click', startWebcamStream);
stopStreamBtn.addEventListener('click', stopWebcamStream);

async function loadLatest() {
  try {
    const [eventsRes, handsRes] = await Promise.all([
      fetch('/api/detections/latest'),
      fetch('/api/hands/latest'),
    ]);

    if (eventsRes.ok) {
      const data = await eventsRes.json();
      for (const event of data.events || []) {
        addEvent(event);
      }
    }

    if (handsRes.ok) {
      const handData = await handsRes.json();
      const cameraId = cameraIdInput.value.trim() || 'cam-1';
      const selected = (handData.hands || []).find((hand) => hand.camera_id === cameraId) || handData.hands?.[0];
      if (selected) {
        renderHand(selected, selected.camera_id);

        const adviceRes = await fetch(`/api/hands/${selected.camera_id}/advice`);
        if (adviceRes.ok) {
          const adviceData = await adviceRes.json();
          renderAdvice(adviceData.advice);
        }
      }
    }
  } catch (_) {}
}

function connectWS() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${protocol}://${location.host}/ws/detections`);

  ws.onmessage = (msg) => {
    try {
      addEvent(JSON.parse(msg.data));
    } catch (_) {}
  };

  ws.onclose = () => {
    setTimeout(connectWS, 1000);
  };
}

loadLatest();
connectWS();
updateStreamStatus('Register camera then start webcam stream.');
