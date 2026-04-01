const registerForm = document.getElementById('register-form');
const cameraIdInput = document.getElementById('camera-id');
const cameraSourceInput = document.getElementById('camera-source');
const registerStatus = document.getElementById('register-status');
const eventsEl = document.getElementById('events');

function addEvent(event) {
  const wrapper = document.createElement('div');
  wrapper.className = 'event';

  const title = document.createElement('div');
  title.textContent = `Camera ${event.camera_id} | ${event.model_name} | ${event.detections.length} detections`;

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

async function loadLatest() {
  try {
    const res = await fetch('/api/detections/latest');
    if (!res.ok) return;
    const data = await res.json();
    for (const event of data.events || []) {
      addEvent(event);
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
