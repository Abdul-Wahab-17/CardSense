"""Cheap frame-diff change detector.

The agent only calls Claude vision when the frame is meaningfully different
from the last classified one. Otherwise we'd burn API calls on a static scene.

Approach: downsample to 32×32 grayscale, store the byte signature per game,
mean-absolute-difference against the last signature, threshold.

Adds Pillow as a backend dependency (~200KB) — used only here.
"""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("cardsense.blackjack.change_detector")


# Downsample resolution. Lower = faster + cheaper but less sensitive. 32 is
# enough to detect a card landing on the table without overreacting to noise.
_DOWNSAMPLE = 32
# Threshold: mean absolute difference (per-pixel grayscale 0-255 range) above
# which we consider the frame "changed". 8 is a reasonable starting point —
# tolerates lighting flicker and minor camera shake but catches a card flip.
_DIFF_THRESHOLD = 8.0
# Forced refresh: if no change has fired for this long, send the frame anyway.
# Catches slow scene changes (someone slowly slides a card in) the diff missed.
_FORCE_REFRESH_MS = 12_000
# Per-game state.
_lock = threading.Lock()
_last_signature: Dict[str, bytes] = {}
_last_classified_ts: Dict[str, int] = {}
# Ring buffer holds the current frame + 1 prior. get_recent_frames() returns
# everything EXCEPT the most-recently-pushed (which is the current request),
# so maxlen=2 → we deliver 1 previous frame to the agent. Combined with the
# current frame the LLM sees 2 images total (down from 4 for ~2x speedup).
_RECENT_FRAMES_KEEP = 2
_recent_frames: Dict[str, Deque[Tuple[int, str]]] = {}


def _signature(image_b64: str) -> Optional[bytes]:
    """Return a downsampled grayscale signature of the JPEG. None on decode error."""
    try:
        from PIL import Image  # local import — avoid hard dep when unused
    except ImportError:
        # Pillow not installed → degrade gracefully: never skip frames.
        return None
    try:
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw)).convert("L").resize(
            (_DOWNSAMPLE, _DOWNSAMPLE), Image.BILINEAR
        )
        return img.tobytes()
    except Exception as exc:  # noqa: BLE001
        logger.warning("frame signature failed: %s", exc)
        return None


def _mae(a: bytes, b: bytes) -> float:
    if len(a) != len(b):
        return float("inf")
    total = 0
    for x, y in zip(a, b):
        d = x - y
        total += d if d >= 0 else -d
    return total / len(a)


def frame_changed(game_id: str, image_b64: str, now_ms: Optional[int] = None) -> bool:
    """True if this frame is different enough from the last classified frame
    for this game (or if the forced-refresh timer has elapsed). When True,
    the frame is also cached into the recent-frames ring buffer for temporal
    context in the next agent call."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    sig = _signature(image_b64)
    if sig is None:
        # Couldn't compute (Pillow missing or decode error) — fail open: classify.
        with _lock:
            _last_classified_ts[game_id] = now_ms
            _push_recent(game_id, now_ms, image_b64)
        return True

    with _lock:
        prev_sig = _last_signature.get(game_id)
        prev_ts = _last_classified_ts.get(game_id, 0)

    if prev_sig is None:
        # First frame for this game → always classify
        with _lock:
            _last_signature[game_id] = sig
            _last_classified_ts[game_id] = now_ms
            _push_recent(game_id, now_ms, image_b64)
        return True

    diff = _mae(sig, prev_sig)
    forced = (now_ms - prev_ts) >= _FORCE_REFRESH_MS

    if diff >= _DIFF_THRESHOLD or forced:
        with _lock:
            _last_signature[game_id] = sig
            _last_classified_ts[game_id] = now_ms
            _push_recent(game_id, now_ms, image_b64)
        return True
    return False


def _push_recent(game_id: str, ts_ms: int, image_b64: str) -> None:
    """Append a (ts, b64) to the ring buffer. Caller holds _lock."""
    buf = _recent_frames.get(game_id)
    if buf is None:
        buf = deque(maxlen=_RECENT_FRAMES_KEEP)
        _recent_frames[game_id] = buf
    buf.append((ts_ms, image_b64))


def get_recent_frames(game_id: str) -> List[Tuple[int, str]]:
    """Return list of (ts_ms, b64) for the last classified frames, oldest first.
    Excludes the very latest one (it's the current request — caller passes that
    separately as the 'current' frame). Up to N-1 entries returned."""
    with _lock:
        buf = _recent_frames.get(game_id)
        if not buf:
            return []
        # The last entry IS the current frame (just pushed). Skip it.
        items = list(buf)[:-1]
        return items


def reset(game_id: str) -> None:
    """Drop cached signature + recent frames for a game (on new_round / start_game)."""
    with _lock:
        _last_signature.pop(game_id, None)
        _last_classified_ts.pop(game_id, None)
        _recent_frames.pop(game_id, None)
