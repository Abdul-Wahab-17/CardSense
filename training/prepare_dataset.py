"""Download (or take) a playing-card detection dataset and turn it into a checked, 52-class YOLO dataset.

Whatever the source calls its classes ("10 of clubs", "10C", "ten_of_clubs", "Qs" ...), every label is
rewritten to the canonical class list below (rank + suit letter, "AS", "10H", "KD" ...), which the backend
already parses. Class ids in the label files are remapped to match. The result is checked before training:
all 52 cards present, every label file well formed, no image in two splits.

Sources (pick one):
    --kaggle OWNER/SLUG     needs ~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY
    --roboflow WS/PROJECT/VERSION   needs ROBOFLOW_API_KEY   (the report's family of datasets:
                                    augmented-startups/playing-cards-ow27d/4)
    --dir PATH              an already-extracted YOLO dataset (or a .zip of one)

    python training/prepare_dataset.py --kaggle andy8744/playing-cards-object-detection-dataset --out data/cards
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["C", "D", "H", "S"]
CLASSES = [f"{r}{s}" for s in SUITS for r in RANKS]  # 52 names; index = class id

_RANK_WORDS = {
    "ace": "A", "one": "A", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "jack": "J", "queen": "Q", "king": "K",
}
_SUIT_WORDS = {
    "club": "C", "clubs": "C", "diamond": "D", "diamonds": "D", "heart": "H", "hearts": "H",
    "spade": "S", "spades": "S", "c": "C", "d": "D", "h": "H", "s": "S",
}
_SPLITS = {"train": "train", "training": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def canonical_name(raw: str) -> str | None:
    """Map a source class name to one of CLASSES, or None if it is not one of the 52 cards (e.g. 'joker')."""
    s = str(raw).strip().lower().replace("♠", "s").replace("♥", "h").replace("♦", "d").replace("♣", "c")
    s = s.replace("_", " ").replace("-", " ")
    parts = [p for p in re.split(r"\s+", s) if p and p != "of"]
    if len(parts) == 2 and parts[1] in _SUIT_WORDS:  # "10 clubs", "queen spades"
        rank = _RANK_WORDS.get(parts[0], parts[0].upper())
        suit = _SUIT_WORDS[parts[1]]
    else:
        token = "".join(parts)
        m = re.fullmatch(r"(10|[2-9]|[ajqk]|ace|jack|queen|king|ten)([cdhs]|clubs?|diamonds?|hearts?|spades?)", token)
        if not m:
            return None
        rank = _RANK_WORDS.get(m.group(1), m.group(1).upper())
        suit = _SUIT_WORDS[m.group(2)]
    name = f"{rank}{suit}"
    return name if name in CLASSES else None


def _read_names(yaml_path: Path) -> list[str]:
    import yaml

    data = yaml.safe_load(yaml_path.read_text())
    names = data["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names)


def _find_root(path: Path) -> Path:
    """The directory holding data.yaml (it may be one level down inside an archive)."""
    for candidate in [path, *sorted(p for p in path.rglob("*") if p.is_dir())]:
        for name in ("data.yaml", "dataset.yaml", "data.yml"):
            if (candidate / name).is_file():
                return candidate
    raise SystemExit(f"no data.yaml found under {path}")


def _split_dirs(root: Path) -> dict[str, tuple[Path, Path]]:
    """{split: (images_dir, labels_dir)} for both layouts: <split>/images and images/<split>."""
    found: dict[str, tuple[Path, Path]] = {}
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if d.name.lower() in _SPLITS and (d / "images").is_dir():
            found[_SPLITS[d.name.lower()]] = (d / "images", d / "labels")
    images, labels = root / "images", root / "labels"
    if images.is_dir():
        for d in images.iterdir():
            if d.is_dir() and d.name.lower() in _SPLITS:
                found.setdefault(_SPLITS[d.name.lower()], (d, labels / d.name))
    return found


def convert(src: Path, out: Path) -> dict:
    """Write the canonical dataset to `out` and return a summary. Raises SystemExit on a bad dataset."""
    root = _find_root(src)
    names = _read_names(next(root / n for n in ("data.yaml", "dataset.yaml", "data.yml") if (root / n).is_file()))
    id_map: dict[int, int] = {}
    ignored: list[str] = []
    for i, raw in enumerate(names):
        canon = canonical_name(raw)
        if canon is None:
            ignored.append(str(raw))
        else:
            id_map[i] = CLASSES.index(canon)
    missing = [c for c in CLASSES if CLASSES.index(c) not in id_map.values()]
    if missing:
        raise SystemExit(f"source has no class for: {', '.join(missing)} (source names: {names})")
    if len(set(id_map.values())) != len(id_map):
        raise SystemExit("two source classes map to the same card; check the dataset's class names")

    splits = _split_dirs(root)
    if not {"train", "val"} <= splits.keys():
        raise SystemExit(f"need train and val splits, found {sorted(splits)}")
    if "test" not in splits:  # held-out test is required for an honest score; carve one from val
        print("no test split in the source: using the second half of val (by file hash) as test", file=sys.stderr)

    if out.exists():
        shutil.rmtree(out)
    counts = {s: 0 for s in ("train", "val", "test")}
    boxes = {c: 0 for c in CLASSES}
    dropped_boxes = 0
    seen: dict[str, str] = {}  # image content hash -> split, to catch leakage across splits
    duplicates = 0

    def emit(split: str, img: Path, lbl_dir: Path):
        nonlocal dropped_boxes, duplicates
        digest = hashlib.sha1(img.read_bytes()).hexdigest()
        if digest in seen:  # identical image already placed (same or other split): keep the first only
            duplicates += 1
            return
        seen[digest] = split
        lbl = lbl_dir / (img.stem + ".txt")
        lines_out = []
        if lbl.is_file():
            for ln, line in enumerate(lbl.read_text().splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    raise SystemExit(f"{lbl}:{ln}: expected 'class x y w h', got {len(parts)} fields (segmentation labels are not supported)")
                cid = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise SystemExit(f"{lbl}:{ln}: box outside the image: {line}")
                if cid not in id_map:
                    dropped_boxes += 1
                    continue
                boxes[CLASSES[id_map[cid]]] += 1
                lines_out.append(f"{id_map[cid]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        if not lines_out:  # an image with no card boxes is not useful training signal
            return
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        stem = f"{digest[:16]}"
        shutil.copyfile(img, out / "images" / split / (stem + img.suffix.lower()))
        (out / "labels" / split / (stem + ".txt")).write_text("\n".join(lines_out) + "\n")
        counts[split] += 1

    for split, (img_dir, lbl_dir) in sorted(splits.items()):
        for img in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in _IMG_EXT):
            emit(split, img, lbl_dir)

    if counts["test"] == 0:  # move half of val (deterministic, by name) to test
        val_imgs = sorted((out / "images" / "val").iterdir())
        (out / "images" / "test").mkdir(parents=True, exist_ok=True)
        (out / "labels" / "test").mkdir(parents=True, exist_ok=True)
        for img in val_imgs[len(val_imgs) // 2:]:
            shutil.move(str(img), out / "images" / "test" / img.name)
            shutil.move(str(out / "labels" / "val" / (img.stem + ".txt")), out / "labels" / "test" / (img.stem + ".txt"))
            counts["val"] -= 1
            counts["test"] += 1

    empty = [c for c, n in boxes.items() if n == 0]
    if empty:
        raise SystemExit(f"no boxes at all for: {', '.join(empty)}")
    if min(counts.values()) == 0:
        raise SystemExit(f"an empty split: {counts}")

    yaml_text = (
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    )
    (out / "data.yaml").write_text(yaml_text)
    summary = {
        "images": counts, "boxes_per_class_min": min(boxes.values()), "boxes_per_class_max": max(boxes.values()),
        "boxes_total": sum(boxes.values()), "ignored_source_classes": ignored, "dropped_boxes_of_ignored": dropped_boxes,
        "duplicate_images_skipped": duplicates,
    }
    return summary


def fetch_kaggle(slug: str, dest: Path) -> Path:
    import kagglehub  # reads ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY

    return Path(kagglehub.dataset_download(slug))


def fetch_roboflow(spec: str, dest: Path) -> Path:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("set ROBOFLOW_API_KEY (free at roboflow.com -> Settings -> API Keys)")
    workspace, project, version = spec.split("/")
    from roboflow import Roboflow

    ds = Roboflow(api_key=key).workspace(workspace).project(project).version(int(version)).download("yolov8", location=str(dest))
    return Path(ds.location)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--kaggle")
    src.add_argument("--roboflow")
    src.add_argument("--dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/cards"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw_cards"), help="where --roboflow/--zip are unpacked")
    args = ap.parse_args(argv)

    if args.kaggle:
        raw = fetch_kaggle(args.kaggle, args.raw)
    elif args.roboflow:
        raw = fetch_roboflow(args.roboflow, args.raw)
    else:
        raw = args.dir
        if raw.suffix == ".zip":
            args.raw.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(raw) as z:
                z.extractall(args.raw)
            raw = args.raw
    summary = convert(raw, args.out)
    print(f"dataset ready at {args.out}/data.yaml")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
