from __future__ import annotations

from .config import settings


def compute_desired_workers(queue_depth_frames: int) -> dict:
    target = max(1, int(settings.target_queue_depth_per_worker))
    min_w = max(1, int(settings.worker_autoscale_min))
    max_w = max(min_w, int(settings.worker_autoscale_max))
    desired = max(min_w, min(max_w, int((queue_depth_frames + target - 1) / target)))
    return {
        "queue_depth_frames": int(queue_depth_frames),
        "target_per_worker": target,
        "desired_workers": desired,
        "min": min_w,
        "max": max_w,
    }
