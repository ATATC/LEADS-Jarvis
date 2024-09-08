from typing import Any as _Any

from leads_vec import Config as _Config


class Config(_Config):
    def __init__(self, base: dict[str, _Any]) -> None:
        self.comm_addr: str = "127.0.0.1"
        super().__init__(base)
