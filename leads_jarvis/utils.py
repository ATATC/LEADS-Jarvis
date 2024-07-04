from os.path import abspath as _abspath, exists as _exists

from leads import L as _L
from numpy import ndarray as _ndarray
from requests import head as _head, get as _get
from rich.progress import Progress as _Progress

_CHECKPOINTS_PATH: str = f"{_abspath(__file__)[:-8]}checkpoints"


def from_opencv(image: _ndarray) -> _ndarray:
    return image.transpose(2, 0, 1)


def to_opencv(image: _ndarray) -> _ndarray:
    return image.transpose(1, 2, 0)


def download_checkpoint(url: str, to: str, overwrite: bool = False) -> str:
    if _exists(to := f"{_CHECKPOINTS_PATH}/{to}") and not overwrite:
        return to
    _L.info(f"Downloading checkpoint {url}...")
    response = _head(url)
    file_size = int(response.headers.get("content-length", 0))
    response = _get(url, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download checkpoint: {response.status_code}")
    with _Progress() as progress:
        task = progress.add_task("[white]Downloading checkpoint...", total=file_size)
        progress.update(task)
        with open(to, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress.update(task, advance=len(chunk))
    return to
