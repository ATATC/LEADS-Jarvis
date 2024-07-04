from typing import override as _override, Any as _Any, Literal as _Literal

from numpy import ndarray as _ndarray
from ultralytics import YOLO as _YOLO

from leads_jarvis.network.detection.prototype import Detection
from leads_jarvis.utils import _CHECKPOINTS_PATH, to_opencv


class PretrainedYOLOWrapper(Detection):
    def __init__(self, yolo: _YOLO) -> None:
        self._yolo: _YOLO = yolo

    @_override
    def detect(self, image: _ndarray) -> list[dict[str, _Any]]:
        return self._yolo(to_opencv(image))[0].summary()


class PretrainedYOLO(Detection):
    @_override
    def __new__(cls, variant: _Literal["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"] = "yolov8n") -> Detection:
        return PretrainedYOLOWrapper(_YOLO(f"{_CHECKPOINTS_PATH}/{variant}.pt"))
