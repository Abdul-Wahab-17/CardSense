import argparse
import base64
import time

import cv2
import requests


def register(base_url: str, camera_id: str, source: str) -> None:
    requests.post(
        f"{base_url}/api/cameras/register",
        json={"camera_id": camera_id, "source": source},
        timeout=10,
    ).raise_for_status()


def send_frames(base_url: str, camera_id: str, source: str, fps: float) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")

    frame_interval = 1.0 / max(fps, 1)
    last_send = 0.0

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
            "camera_id": camera_id,
            "frame_b64": frame_b64,
            "ts_ms": int(now * 1000),
        }

        try:
            requests.post(f"{base_url}/api/frames", json=payload, timeout=10).raise_for_status()
        except Exception as exc:
            print(f"Failed sending frame: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--camera-id", default="cam-1")
    parser.add_argument("--source", default="0", help="OpenCV source index or RTSP/file URL")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    register(args.base_url, args.camera_id, str(source))
    print(f"Registered {args.camera_id}. Streaming frames...")
    send_frames(args.base_url, args.camera_id, source, args.fps)


if __name__ == "__main__":
    main()
