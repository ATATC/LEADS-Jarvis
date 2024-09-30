from typing import Self as _Self

from leads_jarvis.types import Device as _Device


class Prototype(object):
    def __init__(self, device: _Device) -> None:
        self._device: _Device = device

    def to(self, device: _Device) -> _Self:
        self._device = device
        return self
