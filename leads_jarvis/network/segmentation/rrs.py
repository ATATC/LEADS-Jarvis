from torch import Tensor as _Tensor, cat as _cat, zeros_like as _zeros_like
from torch.nn import Module as _Module, LazyConv2d as _LazyConv2d, LazyBatchNorm2d as _LazyBatchNorm2d, ReLU as _ReLU, \
    PReLU as _PReLU, MaxPool2d as _MaxPool2d
from torch.nn.functional import interpolate as _interpolate


class ResidualLayer(_Module):
    def __init__(self, num_channels: int, stride: int = 1, use_side_conv: bool = False) -> None:
        super().__init__()
        self.conv1: _Module = _LazyConv2d(num_channels, 3, stride, 1)
        self.bn1: _Module = _LazyBatchNorm2d()
        self.relu1: _Module = _ReLU()
        self.conv2: _Module = _LazyConv2d(num_channels, 3, 1, 1)
        self.bn2: _Module = _LazyBatchNorm2d()
        self.side_conv: _Module | None = _LazyConv2d(num_channels, 1, stride) if use_side_conv else None
        self.relu2: _Module = _ReLU()

    def forward(self, x: _Tensor) -> _Tensor:
        gx = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))
        if self.side_conv:
            x = self.side_conv(x)
        return self.relu2(gx + x)


class SemanticLayer(_Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.res1: _Module = ResidualLayer(in_channels)
        self.res2: _Module = ResidualLayer(out_channels, 2, True)
        self.res3: _Module = ResidualLayer(out_channels)

    def forward(self, x: _Tensor) -> _Tensor:
        return self.res3(self.res2(self.res1(x)))


class DownSamplingBottleneck(_Module):
    def __init__(self, in_channels: int, out_channels: int, internal_ratio: int = 4) -> None:
        super().__init__()
        internal_channels = in_channels // internal_ratio
        self.conv1: _Module = _LazyConv2d(internal_channels, 2, 2)
        self.bn1: _Module = _LazyBatchNorm2d()
        self.relu1: _Module = _PReLU()
        self.conv2: _Module = _LazyConv2d(internal_channels, 3, 1, 1)
        self.bn2: _Module = _LazyBatchNorm2d()
        self.relu2: _Module = _PReLU()
        self.conv3: _Module = _LazyConv2d(out_channels, 1)
        self.bn3: _Module = _LazyBatchNorm2d()
        self.relu3: _Module = _PReLU()
        self.pool: _Module = _MaxPool2d(2, 2)

    def forward(self, x: _Tensor) -> _Tensor:
        main = self.pool(x)
        main = _cat((main, _zeros_like(main)), 1)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu3(out + main)


class RRSNetwork(_Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, semantic_channels: int = 32) -> None:
        super().__init__()
        self.semantic: _Module = SemanticLayer(in_channels, semantic_channels)
        self.down1: _Module = DownSamplingBottleneck(semantic_channels, semantic_channels * 2)
        self.down2: _Module = DownSamplingBottleneck(semantic_channels * 2, semantic_channels * 4)
        self.conv: _Module = _LazyConv2d(out_channels, 1)

    def forward(self, x: _Tensor) -> _Tensor:
        x = self.semantic(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.conv(x)
        return _interpolate(x, scale_factor=8, mode="bilinear", align_corners=True)
