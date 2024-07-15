from io import BytesIO as _BytesIO
from typing import override as _override

from PIL.Image import open as _open
from leads import require_config, FRONT_VIEW_CAMERA, LEFT_VIEW_CAMERA, RIGHT_VIEW_CAMERA, REAR_VIEW_CAMERA, L
from leads.comm import start_client, create_client, Callback, Service, ConnectionBase
from leads_gui import Window, ContextManager, RuntimeData, Photo, ImageVariable


def main() -> int:
    cfg = require_config()
    w = Window(cfg.width, cfg.height, cfg.refresh_rate, RuntimeData(), title="LEADS Jarvis", fullscreen=cfg.fullscreen,
               no_title_bar=cfg.no_title_bar, theme_mode=cfg.theme_mode)
    front = ImageVariable(w.root(), None)
    left = ImageVariable(w.root(), None)
    right = ImageVariable(w.root(), None)
    rear = ImageVariable(w.root(), None)

    class StreamCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self._map: dict[str, ImageVariable] = {
                FRONT_VIEW_CAMERA: front,
                LEFT_VIEW_CAMERA: left,
                RIGHT_VIEW_CAMERA: right,
                REAR_VIEW_CAMERA: rear
            }

        @_override
        def on_connect(self, service: Service, connection: ConnectionBase) -> None:
            self.super(service=service, connection=connection)
            L.info("Connected")

        @_override
        def on_disconnect(self, service: Service, connection: ConnectionBase) -> None:
            self.super(service=service, connection=connection)
            L.info("Disconnected")
            uim.kill()

        @_override
        def on_receive(self, service: Service, msg: bytes) -> None:
            self.super(service=service, msg=msg)
            split = msg.find(b":")
            if split < 1:
                return
            try:
                self._map[msg[:split].decode()].set(_open(_BytesIO(msg[split + 1:])))
            except (UnicodeDecodeError, KeyError):
                return

        @_override
        def on_fail(self, service: Service, error: Exception) -> None:
            self.super(service=service, error=error)
            L.error(f"Comm stream client error: {repr(error)}")
            uim.kill()

    start_client(cfg.comm_addr, create_client(cfg.comm_stream_port, StreamCallback(), b"end;"), True)
    height = w.height() // 2
    uim = ContextManager(w)
    uim[FRONT_VIEW_CAMERA] = Photo(w.root(), height=height, variable=front)
    uim[LEFT_VIEW_CAMERA] = Photo(w.root(), height=height, variable=left)
    uim[RIGHT_VIEW_CAMERA] = Photo(w.root(), height=height, variable=right)
    uim[REAR_VIEW_CAMERA] = Photo(w.root(), height=height, variable=rear)
    uim.layout([[FRONT_VIEW_CAMERA, REAR_VIEW_CAMERA],
                [LEFT_VIEW_CAMERA, RIGHT_VIEW_CAMERA]])
    uim.show()
    return 0
