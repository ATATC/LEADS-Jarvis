from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod
from typing import Any as _Any

from numpy import ndarray as _ndarray


class Segmentation(object, metaclass=_ABCMeta):
    @_abstractmethod
    def segment(self, image: _ndarray) -> list[dict[str, _Any]]:
        raise NotImplementedError

    def __call__(self, image: _ndarray) -> list[dict[str, _Any]]:
        return self.segment(image)
