from backend.app.autoscaler import compute_desired_workers


def test_load_autoscaler_scales_with_queue_depth():
    low = compute_desired_workers(0)
    high = compute_desired_workers(10_000)
    assert low["desired_workers"] >= low["min"]
    assert high["desired_workers"] <= high["max"]
    assert high["desired_workers"] >= low["desired_workers"]
