from pydantic import BaseModel, Field
from typing import List, Optional


class FrameIn(BaseModel):
    camera_id: str = Field(..., min_length=1)
    frame_b64: str = Field(..., min_length=10, description="Base64 JPEG/PNG frame")
    ts_ms: int = Field(..., ge=0)


class CameraRegister(BaseModel):
    camera_id: str = Field(..., min_length=1)
    source: Optional[str] = None


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    cls_id: int
    cls_name: str
    conf: float
    bbox: BBox


class DetectionEvent(BaseModel):
    camera_id: str
    ts_ms: int
    model_name: str
    detections: List[Detection]
