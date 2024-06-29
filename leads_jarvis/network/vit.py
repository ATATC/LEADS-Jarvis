from typing import override as _override, Literal as _Literal

from timm import create_model as _create_model
from torch import Tensor as _Tensor, stack as _stack
from torch.nn import Module as _Module, Linear as _Linear
from torchvision.transforms.functional import pad as _pad, resize as _resize, normalize as _normalize


def calculate_padding(height: int, width: int) -> tuple[int, int, int, int]:
    padding_top = (max(height, width) - height) // 2
    padding_bottom = max(height, width) - height - padding_top
    padding_left = (max(height, width) - width) // 2
    padding_right = max(height, width) - width - padding_left
    return padding_left, padding_right, padding_top, padding_bottom


def transform_batch(x: _Tensor) -> _Tensor:
    transformed_tensors = []
    for img in x:
        padding_left, padding_right, padding_top, padding_bottom = calculate_padding(img.shape[1], img.shape[2])
        img_padded = _pad(img, [padding_left, padding_right, padding_top, padding_bottom])
        img_resized = _resize(img_padded, [224, 224])
        img_normalized = _normalize(img_resized, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transformed_tensors.append(img_normalized)
    return _stack(transformed_tensors)


class PretrainedVisionTransformerWrapper(_Module):
    def __init__(self, base: _Module, output_dim: int, frozen: bool = True) -> None:
        super().__init__()
        base.head = _Linear(base.head.in_features, output_dim)
        self.base: _Module = base.eval() if frozen else base

    def forward(self, x: _Tensor) -> _Tensor:
        transform_batch(x)
        return self.base(x)


class PretrainedVisionTransformer(_Module):
    @_override
    def __new__(cls, variant: _Literal["vit_base_patch16_224"], output_dim: int, pretrained: bool = True) -> _Module:
        return PretrainedVisionTransformerWrapper(_create_model(variant, pretrained), output_dim, pretrained)
