from typing import override as _override, Literal as _Literal

from timm import create_model as _create_model
from torch import Tensor as _Tensor
from torch.nn import Module as _Module, Linear as _Linear


class PretrainedVisionTransformerWrapper(_Module):
    def __init__(self, base: _Module, output_dim: int, frozen: bool = True) -> None:
        super().__init__()
        base.head = _Linear(base.head.in_features, output_dim)
        self.base: _Module = base.eval() if frozen else base

    def forward(self, x: _Tensor) -> _Tensor:
        return self.base(x)


class PretrainedVisionTransformer(_Module):
    @_override
    def __new__(cls, variant: _Literal["vit_base_patch16_224"], output_dim: int, pretrained: bool = True) -> _Module:
        return PretrainedVisionTransformerWrapper(_create_model(variant, pretrained), output_dim, pretrained)
