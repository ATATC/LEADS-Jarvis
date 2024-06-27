from typing import Literal as _Literal

from leads import L as _L
from torch import save as _save
from torch.nn import Module as _Module
from torch.optim import Optimizer as _Optimizer

from leads_jarvis.dataset import BatchDataset


class Trainer(object):
    def __init__(self, dataset: BatchDataset, network: _Module, criterion: _Module, optimizer: _Optimizer,
                 weights_file: str, device: _Literal["cpu", "cuda"] = "cpu") -> None:
        self._dataset: BatchDataset = dataset
        self._network: _Module = network
        self._criterion: _Module = criterion
        self._optimizer: _Optimizer = optimizer
        self._weights_file: str = weights_file
        self._device: _Literal["cpu", "cuda"] = device

    def initialize(self) -> None:
        self._dataset.load()
        self._network = self._network.to(self._device)

    def train(self, num_epochs: int) -> None:
        self._network.train()
        i = 0
        for data, target in self._dataset:
            if i >= num_epochs:
                return
            data = data.to(self._device)
            target = target.to(self._device)
            output = self._network(data)
            _L.debug(f"Network output: {output}")
            _L.debug(f"Target: {target}")
            loss = self._criterion(output, target)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            _L.info(f"Epoch {i}: Loss: {loss.item():.4f}")
            _save(self._network.state_dict(), self._weights_file)
            i += 1
