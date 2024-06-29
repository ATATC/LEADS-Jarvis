from json import loads as _loads, JSONDecodeError as _JSONDecodeError
from time import sleep as _sleep
from typing import override as _override, Generator as _Generator, Any as _Any

from leads import dlat2meters as _dlat2meters, dlon2meters as _dlon2meters
from leads.comm import Client as _Client, create_client as _create_client, Callback as _Callback, Service as _Service, \
    start_client as _start_client
from leads.data_persistence import CSVDataset as _CSVDataset
from leads.data_persistence.analyzer import DynamicProcessor as _DynamicProcessor
from numpy import arctan as _arctan
from torch import Tensor as _Tensor, tensor as _tensor, float as _float, stack as _stack
from torch.nn.functional import pad as _pad
from torchvision.transforms.functional import resize as _resize


def _delta_theta(a: dict[str, _Any], b: dict[str, _Any], c: dict[str, _Any]) -> float:
    lat_a, lon_a, lat_b, lon_b, lat_c, lon_c = (a["latitude"], a["longitude"], b["latitude"], b["longitude"],
                                                c["latitude"], c["longitude"])
    theta_i = _dlon2meters(lon_b - lon_a, lat_a) / _dlat2meters(lat_b - lat_a)
    theta_j = _dlon2meters(lon_c - lon_b, lat_b) / _dlat2meters(lat_c - lat_b)
    return _arctan(theta_j) - _arctan(theta_i)


def calculate_padding(width: int, height: int) -> tuple[int, int, int, int]:
    target = max(width, height)
    left = (target - width) // 2
    right = target - width - left
    top = (target - height) // 2
    bottom = target - height - top
    return left, right, top, bottom


def transform_batch(x: _Tensor, img_size: int = 224) -> _Tensor:
    transformed_tensors = []
    for img in x:
        img = _pad(img, list(calculate_padding(img.shape[-1], img.shape[-2])))
        img = _resize(img, [img_size, img_size])
        transformed_tensors.append(img)
    return _stack(transformed_tensors)


class BatchDataset(_CSVDataset):
    def __init__(self, file: str, batch_size: int, channels: tuple[str, ...]) -> None:
        super().__init__(file, batch_size)
        self._channels: tuple[str, ...] = channels

    @_override
    def __iter__(self) -> _Generator[tuple[_Tensor, _Tensor], None, None]:
        batch = []
        for i in super().__iter__():
            if len(batch) >= self._chunk_size:
                yield (_tensor(_DynamicProcessor(batch).to_tensor(self._channels), dtype=_float),
                       _tensor((i["throttle"], i["brake"], _delta_theta(*batch[-2:], i)), dtype=_float))
                batch.clear()
            batch.append(i)


class OnlineDataset(BatchDataset, _Callback):
    def __init__(self, server_address: str, server_port: int, batch_size: int, channels: tuple[str, ...]) -> None:
        BatchDataset.__init__(self, server_address, batch_size, channels)
        _Callback.__init__(self)
        self._address: str = server_address
        self._port: int = server_port
        self._client: _Client | None = None
        self._loaded: bool = False
        self._batch: list[dict[str, _Any]] = []

    @_override
    def on_receive(self, service: _Service, msg: bytes) -> None:
        self.super(service=service, msg=msg)
        try:
            self._batch.append(_loads(msg.decode()))
        except _JSONDecodeError:
            pass

    @_override
    def require_loaded(self) -> None:
        if not self._client:
            self.load()

    @_override
    def load(self) -> None:
        self._client = _start_client(self._address, _create_client(self._port, self), True)

    @_override
    def __iter__(self) -> _Generator[tuple[_Tensor, _Tensor], None, None]:
        self.require_loaded()
        while True:
            b = ()
            while len(b) <= self._chunk_size:
                _sleep(.05)
                b = self._batch.copy()
            n = b[self._chunk_size]
            b = b[:self._chunk_size]
            yield (_tensor(_DynamicProcessor(b).to_tensor(self._channels), dtype=_float),
                   _tensor((n["throttle"], n["brake"], _delta_theta(*b[-2:], n)), dtype=_float))
            self._batch.clear()
