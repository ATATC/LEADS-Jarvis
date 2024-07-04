from typing import Literal as _Literal, override as _override, Any as _Any

from numpy import ndarray as _ndarray
from segment_anything import sam_model_registry as _sam_model_registry, \
    SamAutomaticMaskGenerator as _SamAutomaticMaskGenerator

from leads_jarvis.network.segmentation.prototype import Segmentation
from leads_jarvis.utils import download_checkpoint, to_opencv


class PretrainedSAMWrapper(Segmentation):
    def __init__(self, mask_generator: _SamAutomaticMaskGenerator) -> None:
        super().__init__()
        self._mask_generator: _SamAutomaticMaskGenerator = mask_generator

    @_override
    def segment(self, image: _ndarray) -> list[dict[str, _Any]]:
        return self._mask_generator.generate(to_opencv(image))


class PretrainedSAM(Segmentation):
    @_override
    def __new__(cls, variant: _Literal["vit_h", "vit_l", "vit_b"] = "vit_h",
                url: str = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                device: _Literal["cpu", "cuda"] = "cpu") -> Segmentation:
        return PretrainedSAMWrapper(_SamAutomaticMaskGenerator(_sam_model_registry[variant](download_checkpoint(
            url, f"sam_{variant}.pth"))))
