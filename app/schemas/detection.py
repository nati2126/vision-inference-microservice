"""Detection request / response schemas.

Pydantic models define the API contract. FastAPI uses them for:
- Automatic request validation
- OpenAPI (Swagger) documentation generation
- Response serialisation
"""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in xyxy (pixel) format."""

    x_min: float = Field(..., description="Left edge (px)")
    y_min: float = Field(..., description="Top edge (px)")
    x_max: float = Field(..., description="Right edge (px)")
    y_max: float = Field(..., description="Bottom edge (px)")


class Detection(BaseModel):
    """A single detected object."""

    label: str = Field(..., description="Class label predicted by the model")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")


class DetectionMetadata(BaseModel):
    """Metadata about the inference run."""

    image_width: int = Field(..., description="Width of the input image (px)")
    image_height: int = Field(..., description="Height of the input image (px)")
    inference_time_ms: float = Field(..., description="Model inference wall-clock time (ms)")
    detections_count: int = Field(..., ge=0, description="Number of objects detected")


class DetectionResponse(BaseModel):
    """Response payload for ``POST /detect``."""

    detections: list[Detection] = Field(
        default_factory=list, description="List of detected objects"
    )
    metadata: DetectionMetadata = Field(..., description="Inference run metadata")


class HealthResponse(BaseModel):
    """Response payload for ``GET /health``."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Application version")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded and ready")
    backend: str = Field(
        ..., description="Inference runtime: pytorch, onnxruntime, openvino or tensorrt"
    )
    precision: str = Field(
        ..., description="Numeric precision of the loaded model: fp32, fp16 or int8"
    )
    device: str = Field(..., description="Compute device the backend runs on")


class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx / 5xx responses."""

    error: str = Field(..., description="Human-readable error message")
    # ``default=`` must be named: with a positional default, mypy's
    # dataclass_transform handling does not see the field as optional and
    # reports every construction as missing the argument.
    detail: str | None = Field(default=None, description="Additional diagnostic information")
