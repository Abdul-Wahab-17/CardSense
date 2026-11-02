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
