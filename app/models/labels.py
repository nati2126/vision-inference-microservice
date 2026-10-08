"""Class names for the exported backends.

Ultralytics stores the class names in the ONNX metadata as the ``repr`` of a
``{id: name}`` dict. Artifacts that lose that metadata (a quantised or
converted copy) fall back to the 80 COCO classes yolov8n was trained on.
"""

import ast

COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]  # fmt: skip


def parse_names(raw: str | None) -> list[str]:
    """Parse ultralytics' ``names`` metadata, falling back to COCO classes.

    ``ast.literal_eval`` only accepts Python literals, so a tampered
    metadata string cannot execute code.
    """
    if not raw:
        return list(COCO_CLASSES)
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return list(COCO_CLASSES)
    if not isinstance(parsed, dict):
        return list(COCO_CLASSES)
    return [str(parsed[i]) for i in sorted(parsed)]
