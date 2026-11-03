import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
yaml = pytest.importorskip("yaml")
import prepare_dataset as pd  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("10C", "10C"), ("10 of clubs", "10C"), ("ten_of_hearts", "10H"), ("Qs", "QS"), ("ace of spades", "AS"),
    ("king-diamonds", "KD"), ("9♥", "9H"), ("jack of clubs", "JC"), ("2d", "2D"),
])
def test_canonical_name(raw, expected):
    assert pd.canonical_name(raw) == expected


@pytest.mark.parametrize("raw", ["joker", "Joker", "back", "11C", "1H", ""])
def test_non_cards_are_rejected(raw):
    assert pd.canonical_name(raw) is None


def test_all_52_names_round_trip():
    assert len(pd.CLASSES) == len(set(pd.CLASSES)) == 52
    assert all(pd.canonical_name(c) == c for c in pd.CLASSES)


def _make_source(root: Path, names, splits=("train", "valid", "test"), per_class=1, bad_label=None):
    n = 0
    for split in splits:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for cid in range(len(names)):
            for k in range(per_class):
                n += 1
                (root / split / "images" / f"{n}.jpg").write_bytes(f"img-{n}".encode())
                (root / split / "labels" / f"{n}.txt").write_text(f"{cid} 0.5 0.5 0.2 0.3\n")
    if bad_label:
        (root / "train" / "labels" / "1.txt").write_text(bad_label)
    (root / "data.yaml").write_text("names: " + repr(list(names)) + "\n")


def _source_names():
    return [f"{r} of {s}" for s in ("clubs", "diamonds", "hearts", "spades")
            for r in ("ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king")] + ["Joker"]


def test_convert_remaps_ids_and_drops_joker(tmp_path):
    names = _source_names()[::-1]  # reversed, so ids must be remapped
    _make_source(tmp_path / "src", names)
    summary = pd.convert(tmp_path / "src", tmp_path / "out")
    assert summary["images"] == {"train": 52, "val": 52, "test": 52}  # joker images have no card boxes: dropped
    assert summary["ignored_source_classes"] == ["Joker"]
    cfg = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
    assert cfg["nc"] == 52 and cfg["names"][0] == "AC" and cfg["names"][51] == "KS"
    # source id 52 was "ace of clubs" after reversing -> canonical id 0 ... check one label file
    ids = {int(f.read_text().split()[0]) for f in (tmp_path / "out" / "labels" / "train").iterdir()}
    assert ids == set(range(52))


def test_missing_card_is_an_error(tmp_path):
    _make_source(tmp_path / "src", _source_names()[:40])
    with pytest.raises(SystemExit, match="no class for"):
        pd.convert(tmp_path / "src", tmp_path / "out")


def test_malformed_label_is_an_error(tmp_path):
    _make_source(tmp_path / "src", _source_names(), bad_label="0 0.5 0.5 1.4 0.2\n")
    with pytest.raises(SystemExit, match="outside the image"):
        pd.convert(tmp_path / "src", tmp_path / "out")


def test_test_split_is_carved_from_val_when_absent(tmp_path):
    _make_source(tmp_path / "src", _source_names(), splits=("train", "valid"), per_class=2)
    summary = pd.convert(tmp_path / "src", tmp_path / "out")
    assert summary["images"]["test"] > 0 and summary["images"]["val"] > 0


def test_identical_image_is_kept_in_one_split_only(tmp_path):
    _make_source(tmp_path / "src", _source_names())
    (tmp_path / "src" / "train" / "images" / "1.jpg").write_bytes(
        (tmp_path / "src" / "test" / "images" / "107.jpg").read_bytes())
    summary = pd.convert(tmp_path / "src", tmp_path / "out")
    assert summary["duplicate_images_skipped"] >= 1
    stems = [p.stem for s in ("train", "val", "test") for p in (tmp_path / "out" / "images" / s).iterdir()]
    assert len(stems) == len(set(stems))
