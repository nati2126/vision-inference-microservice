"""Send images to a running service and draw the returned detections into one image."""

import argparse
import colorsys
import hashlib
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import numpy.typing as npt

PANEL_HEIGHT = 540
CAPTION_HEIGHT = 44


def _colour(label: str) -> tuple[int, int, int]:
    hue = int(hashlib.md5(label.encode()).hexdigest()[:4], 16) / 0xFFFF
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _draw(image: npt.NDArray[np.uint8], detections: list[dict[str, Any]]) -> None:
    thickness = max(2, round(min(image.shape[:2]) / 300))
    scale = thickness / 3
    for det in detections:
        box = det["bbox"]
        p1 = (round(box["x_min"]), round(box["y_min"]))
        p2 = (round(box["x_max"]), round(box["y_max"]))
        colour = _colour(det["label"])
        cv2.rectangle(image, p1, p2, colour, thickness, cv2.LINE_AA)

        text = f"{det['label']} {det['confidence']:.2f}"
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness // 2 + 1)
        top = max(p1[1] - h - base - 4, 0)
        cv2.rectangle(image, (p1[0], top), (p1[0] + w + 6, top + h + base + 4), colour, -1)
        cv2.putText(
            image,
            text,
            (p1[0] + 3, top + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness // 2 + 1,
            cv2.LINE_AA,
        )


def _detect(client: httpx.Client, path: Path) -> dict[str, Any]:
    response = client.post(
        "/api/v1/detect", files={"file": (path.name, path.read_bytes(), "image/jpeg")}
    )
    response.raise_for_status()
    return dict(response.json())


def _panel(path: Path, client: httpx.Client, caption_prefix: str) -> npt.NDArray[np.uint8]:
    body = _detect(client, path)

    raw = cv2.imread(str(path))
    if raw is None:
        raise SystemExit(f"Could not read {path}")
    image = np.asarray(raw, dtype=np.uint8)
    _draw(image, body["detections"])

    width = round(image.shape[1] * PANEL_HEIGHT / image.shape[0])
    image = np.asarray(
        cv2.resize(image, (width, PANEL_HEIGHT), interpolation=cv2.INTER_AREA), dtype=np.uint8
    )
    caption = np.full((CAPTION_HEIGHT, width, 3), 255, dtype=np.uint8)
    meta = body["metadata"]
    text = f"{caption_prefix} | {meta['detections_count']} objects"
    cv2.putText(
        caption, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA
    )
    print(f"{path.name}: {text}")
    return np.vstack([image, caption])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, default=Path("docs/demo.jpg"))
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=60) as client:
        health = client.get("/api/v1/health").json()
        prefix = f"{health['backend']} {health['precision']} on {health['device']}"
        panels = [_panel(path, client, prefix) for path in args.images]

    gap = np.full((PANEL_HEIGHT + CAPTION_HEIGHT, 12, 3), 255, dtype=np.uint8)
    row = panels[0]
    for panel in panels[1:]:
        row = np.hstack([row, gap, panel])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), row, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
