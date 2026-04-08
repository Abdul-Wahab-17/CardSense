import argparse
import base64
import time
import uuid

import cv2
import requests


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def register(base_url: str, camera_id: str, source: str, token: str = "") -> None:
    requests.post(
        f"{base_url}/api/cameras/register",
        json={"camera_id": camera_id, "source": source},
        headers=_auth_headers(token),
        timeout=10,
    ).raise_for_status()


def flush_batch(base_url: str, batch: list[dict], token: str = "") -> None:
    if not batch:
        return
    if len(batch) == 1:
        requests.post(f"{base_url}/api/frames", json=batch[0], headers=_auth_headers(token), timeout=10).raise_for_status()
        return
    requests.post(
        f"{base_url}/api/frames/batch",
        json={"frames": batch},
        headers=_auth_headers(token),
        timeout=10,
    ).raise_for_status()


def send_frames(
    base_url: str,
    game_id: str,
    camera_id: str,
    camera_type: str,
    source: str,
    fps: float,
    batch_size: int,
    target_model_type: str,
    device_id: str,
    session_id: str,
    auth_token: str,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    frame_interval = 1.0 / max(fps, 1)
    last_send = 0.0
    batch: list[dict] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        if now - last_send < frame_interval:
            continue
        last_send = now

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            continue

        frame_b64 = base64.b64encode(jpg.tobytes()).decode("utf-8")
        payload = {
            "game_id": game_id,
            "camera_id": camera_id,
            "camera_type": camera_type,
            "target_model_type": target_model_type,
            "device_id": device_id,
            "session_id": session_id,
            "frame_id": f"{camera_id}:{int(now * 1000)}:{uuid.uuid4().hex[:8]}",
            "frame_b64": frame_b64,
            "ts_ms": int(now * 1000),
        }
        batch.append(payload)

        if len(batch) >= max(batch_size, 1):
            try:
                flush_batch(base_url, batch, token=auth_token)
                batch.clear()
            except Exception as exc:
                print(f"Failed sending frame batch: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--game-id", default="default-game")
    parser.add_argument("--camera-id", default="cam-1")
    parser.add_argument("--camera-type", default="hand", choices=["hand", "table"])
    parser.add_argument("--source", default="0", help="OpenCV source index or RTSP/file URL")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-model-type", default="", choices=["", "hand", "table"])
    parser.add_argument("--device-id", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--auth-token", default="")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    target_model_type = args.target_model_type or args.camera_type
    device_id = args.device_id or args.camera_id
    session_id = args.session_id or f"sess-{uuid.uuid4().hex[:10]}"
    register(args.base_url, args.camera_id, str(source), token=args.auth_token)
    print(f"Registered {args.camera_id}. Streaming frames...")
    send_frames(
        args.base_url,
        args.game_id,
        args.camera_id,
        args.camera_type,
        source,
        args.fps,
        args.batch_size,
        target_model_type,
        device_id,
        session_id,
        args.auth_token,
    )


if __name__ == "__main__":
    main()
