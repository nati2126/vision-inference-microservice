"""The coco128 dataset: download, and read images with their ground truth.

coco128 is the first 128 images of COCO train2017 with YOLO-format labels.
It is small enough to evaluate every backend in seconds, which is why it is
used here, and also why its mAP is only a relative comparison between
backends, not a substitute for a full COCO val2017 evaluation.
"""

import io
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "datasets"


@dataclass(frozen=True)
class Sample:
    """One image and its ground-truth boxes in pixel xyxy."""

    path: Path
    image: npt.NDArray[np.uint8]
    boxes: npt.NDArray[np.float32]  # (N, 4) xyxy, original-image pixels
    classes: npt.NDArray[np.int64]  # (N,)


def ensure_coco128(root: Path = DEFAULT_ROOT) -> Path:
    """Download and extract coco128 under ``root`` if needed; return its directory."""
    dataset_dir = root / "coco128"
    if (dataset_dir / "images" / "train2017").is_dir():
        return dataset_dir

    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading coco128 from {COCO128_URL} ...")
    with urllib.request.urlopen(COCO128_URL, timeout=120) as response:
        payload = response.read()

    resolved_root = root.resolve()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        # Refuse any member that would land outside ``root`` ("zip slip").
        for member in archive.namelist():
            target = (root / member).resolve()
            if not target.is_relative_to(resolved_root):
                raise RuntimeError(f"Unsafe path in coco128 archive: {member!r}")
        archive.extractall(root)
    return dataset_dir


def image_paths(dataset_dir: Path, limit: int | None = None) -> list[Path]:
    """Sorted image paths, so every run sees the same images in the same order."""
    paths = sorted((dataset_dir / "images" / "train2017").glob("*.jpg"))
    return paths[:limit] if limit is not None else paths


def label_path_for(dataset_dir: Path, image_path: Path) -> Path:
    """``images/train2017/x.jpg`` -> ``labels/train2017/x.txt``."""
    return dataset_dir / "labels" / "train2017" / f"{image_path.stem}.txt"


def load_labels(
    label_path: Path, width: int, height: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Read a YOLO label file and convert it to pixel xyxy boxes.

    Each line is ``class cx cy w h`` with coordinates normalised to [0, 1].
    Images without a label file are background images with no objects.
    """
    if not label_path.exists():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)

    rows = np.loadtxt(label_path, ndmin=2, dtype=np.float32)
    if rows.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)

    classes = rows[:, 0].astype(np.int64)
    cx, cy, w, h = rows[:, 1] * width, rows[:, 2] * height, rows[:, 3] * width, rows[:, 4] * height
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes.astype(np.float32), classes


def iter_samples(dataset_dir: Path, limit: int | None = None) -> Iterator[Sample]:
    """Yield decoded images with their ground truth."""
    for path in image_paths(dataset_dir, limit):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        boxes, classes = load_labels(
            label_path_for(dataset_dir, path), image.shape[1], image.shape[0]
        )
        yield Sample(path, np.asarray(image, dtype=np.uint8), boxes, classes)
