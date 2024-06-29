from typing import override as _override

from leads.data_persistence.analyzer import JarvisBackend as _JarvisBackend
from numpy import ndarray as _ndarray
from torch import Tensor as _Tensor, zeros as _zeros, arange as _arange, exp as _exp, log as _log, tensor as _tensor, \
    sin as _sin, cos as _cos, mean as _mean, cat as _cat
from torch.nn import Module as _Module, Dropout as _Dropout, Linear as _Linear, \
    TransformerEncoderLayer as _TransformerEncoderLayer, TransformerEncoder as _TransformerEncoder

from leads_jarvis.network.vit import PretrainedVisionTransformer


class PositionalEncoding(_Module):
    def __init__(self, num_dim: int, dropout_rate: float = .1, max_length: int = 5000) -> None:
        super().__init__()
        self.dropout: _Dropout = _Dropout(p=dropout_rate)

        pe = _zeros(max_length, num_dim)
        position = _arange(0, max_length).unsqueeze(1)
        div_term = _exp(_arange(0, num_dim, 2).float() * (-_log(_tensor(10000.0)) / num_dim))
        pe[:, 0::2] = _sin(position * div_term)
        pe[:, 1::2] = _cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: _Tensor) -> _Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FridayNetwork(_Module):
    def __init__(self, visual_dim: int, input_dim: int, num_dim: int, num_head: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout_rate: float, output_dim: int = 3) -> None:
        super().__init__()
        input_dim += visual_dim
        self.vit: _Module = PretrainedVisionTransformer("vit_base_patch16_224", visual_dim)
        self.embedding: _Module = _Linear(input_dim, num_dim)
        self.positional_encoding: _Module = PositionalEncoding(num_dim, dropout_rate)
        self.transformer_encoder: _Module = _TransformerEncoder(
            _TransformerEncoderLayer(num_dim, num_head, dim_feedforward, dropout_rate), num_encoder_layers)
        self.fc_out: _Module = _Linear(num_dim, output_dim)

    def forward(self, x: _Tensor, visual: _Tensor) -> _Tensor:
        v = self.vit(visual)
        x = _cat((x, v), dim=1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = _mean(x, dim=1)
        x = self.fc_out(x)
        return x[-1]


class Friday(_JarvisBackend):
    def __init__(self, network: _Module) -> None:
        self._network: _Module = network

    @_override
    def predict(self, x: _ndarray, visual: _ndarray) -> tuple[float, float, float]:
        self._network(x, visual)
        return 0, 0, 0
