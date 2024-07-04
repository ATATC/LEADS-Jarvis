from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod
from typing import Any as _Any

from cv2 import rectangle as _rectangle, putText as _put_text, FONT_HERSHEY_COMPLEX_SMALL as _FONT
from numpy import ndarray as _ndarray

from leads_jarvis.utils import to_opencv, from_opencv


class Detection(object, metaclass=_ABCMeta):
    @_abstractmethod
    def detect(self, image: _ndarray) -> list[dict[str, _Any]]:
        raise NotImplementedError

    def __call__(self, image: _ndarray) -> list[dict[str, _Any]]:
        return self.detect(image)

    def mark(self, image: _ndarray, data: list[dict[str, _Any]] | None = None,
             filter_type: tuple[str, ...] | None = None) -> _ndarray:
        if data is None:
            data = self(image)
        image = to_opencv(image)
        for item in data:
            name = item["name"]
            if filter_type is not None and name not in filter_type:
                continue
            box = item["box"]
            x1, y1, x2, y2 = round(box["x1"]), round(box["y1"]), round(box["x2"]), round(box["y2"])
            image = _rectangle(image, (x1, y1), (x2, y2), (255, 255, 255))
            image = _put_text(image, name, (x1, y1 - 4), _FONT, 1, (255, 255, 255))
        return from_opencv(image)
