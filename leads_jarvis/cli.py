from io import BytesIO as _BytesIO
from threading import Thread as _Thread
from time import sleep as _sleep
from typing import override as _override, Callable as _Callable

from PIL.Image import open as _open, fromarray as _fromarray
from leads import require_config, FRONT_VIEW_CAMERA, LEFT_VIEW_CAMERA, RIGHT_VIEW_CAMERA, REAR_VIEW_CAMERA, L
from leads.comm import start_client, create_client, Callback, Service, ConnectionBase
from leads_gui import Window, ContextManager, RuntimeData, Photo, ImageVariable
from numpy import array as _array

from leads_jarvis.network import Detection, PretrainedYOLO
from leads_jarvis.utils import from_opencv, to_opencv


class StreamCallback(Callback):
    def __init__(self, context_manager: ContextManager, variables: dict[str, ImageVariable]) -> None:
        super().__init__()
        self.uim: ContextManager = context_manager
        self.variables: dict[str, ImageVariable] = variables

    @_override
    def on_connect(self, service: Service, connection: ConnectionBase) -> None:
        self.super(service=service, connection=connection)
        L.info("Connected")

    @_override
    def on_disconnect(self, service: Service, connection: ConnectionBase) -> None:
        self.super(service=service, connection=connection)
        L.info("Disconnected")
        self.uim.kill()

    @_override
    def on_receive(self, service: Service, msg: bytes) -> None:
        self.super(service=service, msg=msg)
        split = msg.find(b":")
        if split < 1:
            return
        try:
            self.variables[msg[:split].decode()].set(_open(_BytesIO(msg[split + 1:])))
        except (UnicodeDecodeError, KeyError):
            return

    @_override
    def on_fail(self, service: Service, error: Exception) -> None:
        self.super(service=service, error=error)
        L.error(f"Comm stream client error: {repr(error)}")
        self.uim.kill()


def make_processor(detection_model: Detection, target_variable: ImageVariable,
                   output_variable: ImageVariable) -> _Callable[[], None]:
    def _() -> None:
        while True:
            if frame := target_variable.get():
                output_variable.set(_fromarray(to_opencv(detection_model.mark(from_opencv(_array(frame)),
                                                                              accepted_classes=("car", "person")))))
            else:
                _sleep(.01)

    return _


def main() -> int:
    cfg = require_config()
    w = Window(cfg.width, cfg.height, cfg.refresh_rate, RuntimeData(), title="LEADS Jarvis", fullscreen=cfg.fullscreen,
               no_title_bar=cfg.no_title_bar, theme_mode=cfg.theme_mode)
    front = ImageVariable(w.root(), None)
    left = ImageVariable(w.root(), None)
    right = ImageVariable(w.root(), None)
    rear = ImageVariable(w.root(), None)
    front_ai = ImageVariable(w.root(), None)
    left_ai = ImageVariable(w.root(), None)
    right_ai = ImageVariable(w.root(), None)
    rear_ai = ImageVariable(w.root(), None)
    uim = ContextManager(w)
    uim["front"] = Photo(w.root(), variable=front).lock_ratio(cfg.m_ratio)
    uim["left"] = Photo(w.root(), variable=left).lock_ratio(cfg.m_ratio)
    uim["right"] = Photo(w.root(), variable=right).lock_ratio(cfg.m_ratio)
    uim["rear"] = Photo(w.root(), variable=rear).lock_ratio(cfg.m_ratio)
    uim["front_ai"] = Photo(w.root(), variable=front_ai).lock_ratio(cfg.m_ratio)
    uim["left_ai"] = Photo(w.root(), variable=left_ai).lock_ratio(cfg.m_ratio)
    uim["right_ai"] = Photo(w.root(), variable=right_ai).lock_ratio(cfg.m_ratio)
    uim["rear_ai"] = Photo(w.root(), variable=rear_ai).lock_ratio(cfg.m_ratio)
    uim.layout([["left", "front", "rear", "right"],
                ["left_ai", "front_ai", "rear_ai", "right_ai"]])
    yolo = PretrainedYOLO("yolov8s")
    _Thread(name="front_processor", target=make_processor(yolo, front, front_ai), daemon=True).start()
    _Thread(name="left_processor", target=make_processor(yolo, left, left_ai), daemon=True).start()
    _Thread(name="right_processor", target=make_processor(yolo, right, right_ai), daemon=True).start()
    _Thread(name="rear_processor", target=make_processor(yolo, rear, rear_ai), daemon=True).start()
    start_client(cfg.comm_addr, create_client(cfg.comm_stream_port, StreamCallback(uim, variables={
        FRONT_VIEW_CAMERA: front,
        LEFT_VIEW_CAMERA: left,
        RIGHT_VIEW_CAMERA: right,
        REAR_VIEW_CAMERA: rear
    }), b"end;"), True)
    uim.show()
    return 0
