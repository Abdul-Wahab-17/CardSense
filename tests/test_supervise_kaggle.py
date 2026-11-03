import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "kaggle"))
import supervise_kaggle as sk  # noqa: E402


class Fake:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0

    def __call__(self, cmd, **kw):
        self.calls += 1
        rc, out = self.outputs.pop(0)
        return subprocess.CompletedProcess(cmd, rc, out, "")


def test_rate_limit_is_retried_with_backoff(monkeypatch):
    sleeps = []
    fake = Fake([(1, "429 Too Many Requests"), (0, "rate limited, slow down"), (0, "ok")])
    monkeypatch.setattr(sk.subprocess, "run", fake)
    monkeypatch.setattr(sk.time, "sleep", sleeps.append)
    assert sk.kaggle("kernels", "status", "x") == "ok"
    assert sleeps == [30, 60] and fake.calls == 3


def test_permanent_error_is_not_retried(monkeypatch):
    fake = Fake([(1, "403 Forbidden: phone verification required")])
    monkeypatch.setattr(sk.subprocess, "run", fake)
    monkeypatch.setattr(sk.time, "sleep", lambda s: pytest.fail("must not sleep"))
    with pytest.raises(RuntimeError, match="403"):
        sk.kaggle("kernels", "push")
    assert fake.calls == 1


def test_gives_up_after_the_retry_budget(monkeypatch):
    fake = Fake([(1, "Read timed out")] * 3)
    monkeypatch.setattr(sk.subprocess, "run", fake)
    monkeypatch.setattr(sk.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        sk.kaggle("datasets", "status", "x", retries=3)
    assert fake.calls == 3


@pytest.mark.parametrize("text,transient", [
    ("CUDA error: an illegal memory access", True), ("DataLoader worker killed", True),
    ("SyntaxError: invalid syntax", False), ("ModuleNotFoundError: No module named 'foo'", False),
])
def test_failure_classification(text, transient):
    assert bool(sk.TRANSIENT.search(text)) is transient
