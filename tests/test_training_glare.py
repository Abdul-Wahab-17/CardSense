import pytest

torch = pytest.importorskip('torch')           # the training stack is optional for the rest of the suite
pytest.importorskip('ultralytics')
from train_yolo import add_glare  # noqa: E402  (training/ is on the pytest path, see pytest.ini)


def test_glare_only_brightens_and_keeps_shape():
    torch.manual_seed(0)
    imgs = torch.zeros(8, 3, 64, 64)
    out = add_glare(imgs.clone(), p=1.0)
    assert out.shape == imgs.shape
    assert out.min() >= 0 and out.max() <= 1
    assert out.max() > 0  # a bright ellipse landed somewhere


def test_glare_probability_zero_is_a_no_op():
    imgs = torch.rand(4, 3, 32, 32)
    assert torch.equal(add_glare(imgs.clone(), p=0.0), imgs)


def test_trainer_applies_glare_in_preprocess_batch(monkeypatch):
    """The augmentation must hang off the method ultralytics actually calls for each training batch."""
    import train_yolo
    from ultralytics.models.yolo.detect import DetectionTrainer

    assert train_yolo.GlareTrainer.preprocess_batch is not DetectionTrainer.preprocess_batch
    calls = []
    monkeypatch.setattr(train_yolo, "add_glare", lambda x, *a, **k: (calls.append(1), x)[1])
    trainer = object.__new__(train_yolo.GlareTrainer)
    trainer.device = torch.device("cpu")
    trainer.args = type("A", (), {"multi_scale": 0.0})()
    out = trainer.preprocess_batch({"img": torch.randint(0, 255, (2, 3, 32, 32), dtype=torch.uint8)})
    assert calls == [1]
    assert out["img"].dtype == torch.float32 and out["img"].max() <= 1.0


def test_glare_is_not_registered_as_a_callback():
    # "on_preprocess_batch" is not an ultralytics event; registering it silently does nothing.
    from ultralytics.utils.callbacks.base import default_callbacks
    assert "on_preprocess_batch" not in default_callbacks
