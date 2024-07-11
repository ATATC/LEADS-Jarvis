from os import listdir
from typing import Any, override

from PIL.Image import open
from torch import Tensor
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, image_path: str, mask_path: str, transform: Any | None = None) -> None:
        self.image_dir: str = image_path
        self.mask_dir: str = mask_path
        self.transform: Any | None = transform

    def __len__(self) -> int:
        return len(listdir(self.image_dir))

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image = open(f"{self.image_dir}/{idx}.png").convert("RGB")
        mask = open(f"{self.mask_dir}/{idx}.png").convert("RGB")
        return self.transform(image) if self.transform else image, self.transform(mask) if self.transform else mask
