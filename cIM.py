# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

import ctypes
from typing import Collection, IO
from dataclasses import dataclass
from contextlib import contextmanager
from collections.abc import Sequence, Generator

libc = ctypes.CDLL("libc.so.6")
libc.free.argtypes = (ctypes.c_void_p,)
libc.free.restype = None

_IM = ctypes.CDLL("./IM.so")

_IM.im_init.argtypes = ()
_IM.im_init.restype = None

_IM.im_quit.argtypes = ()
_IM.im_quit.restype = None

_IM.im_set_context.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
_IM.im_set_context.restype = None

_IM.im_mouse_pos.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
_IM.im_mouse_pos.restype = None

_IM.im_load_texture.argtypes = (ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
_IM.im_load_texture.restype = ctypes.c_void_p

_IM.im_draw_texture.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
_IM.im_draw_texture.restype = None

_IM.im_rotated_rectangle_extents.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
_IM.im_rotated_rectangle_extents.restype = None

_IM.im_begin_frame.argtypes = ()
_IM.im_begin_frame.restype = None

_IM.im_draw_thought_bubble.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
_IM.im_draw_thought_bubble.restype = None

_IM.im_running.argtypes = ()
_IM.im_running.restype = ctypes.c_bool

_IM.im_stop.argtypes = ()
_IM.im_stop.restype = None

_IM.im_end_frame.argtypes = ()
_IM.im_end_frame.restype = None

_IM.im_load_mask.argtypes = (ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
_IM.im_load_mask.restype = ctypes.POINTER(ctypes.c_uint64)

_IM.im_fast_overlap.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
_IM.im_fast_overlap.restype = ctypes.c_int

class im_key_event(ctypes.Structure):
    _fields_ = (
        ("keyname", ctypes.c_char_p),
        ("held", ctypes.c_int)
    )

_IM.im_get_key_events.argtypes = (ctypes.POINTER(ctypes.c_size_t),)
_IM.im_get_key_events.restype = ctypes.POINTER(im_key_event)

_IM.im_free_texture.argtypes = (ctypes.c_void_p,)
_IM.im_free_texture.restype = None

_IM.im_rotozoom.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
_IM.im_rotozoom.restype = ctypes.c_void_p

@dataclass(slots=True, frozen=True)
class KeyEvent:
    key: str
    held: bool

@dataclass
class FrameInfo:
    key_events: list[KeyEvent]

@dataclass
class Texture:
    handle: ctypes.c_void_p
    w: int
    h: int

    def __del__(self):
        _IM.im_free_texture(self.handle)

@dataclass
class Mask:
    handle: ctypes.POINTER(ctypes.c_uint64)
    w: int
    h: int

    def __del__(self):
        libc.free(self.handle)

@dataclass(slots=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def __iter__(self):
        return iter((self.x, self.y, self.width, self.height))

    def __getitem__(self, idx):
        return (self.x, self.y, self.width, self.height)[idx]

    def __setitem__(self, idx, value):
        setattr(self, "xywh"[idx].name, value)

def rotozoom(tex: Texture, angle: float, scale: float):
    raise RuntimeError("don't use this pls")
    w = ctypes.c_int()
    h = ctypes.c_int()
    handle = _IM.im_rotozoom(tex.handle, angle, scale, ctypes.byref(w), ctypes.byref(h))
    return Texture(ctypes.c_void_p(handle), w.value, h.value)

def load_mask(file: IO[bytes]) -> Mask:
    data = file.read()
    w = ctypes.c_int()
    h = ctypes.c_int()
    handle = _IM.im_load_mask(data, len(data), ctypes.byref(w), ctypes.byref(h))
    return Mask(handle, w.value, h.value)

def overlap_masks(mask1: Mask, mask2: Mask, offs: tuple[int, int]) -> tuple[int, int] | None:
    x = ctypes.c_int()
    y = ctypes.c_int()
    if _IM.im_fast_overlap(mask1.handle, mask1.w, mask1.h, mask2.handle, mask2.w, mask2.h, int(offs[0]), int(offs[1]), ctypes.byref(x), ctypes.byref(y)):
        return (x.value, y.value)
    return None

def init() -> None:
    _IM.im_init()

window_width: int = 0
window_height: int = 0
def set_context(width: int, height: int, fps: int=30) -> None:
    global window_width, window_height  # pylint: disable=global-statement
    window_width = width
    window_height = height
    _IM.im_set_context(width, height, fps)

def load_texture(file: IO[bytes]) -> Texture:
    data = file.read()
    w = ctypes.c_int()
    h = ctypes.c_int()
    handle = _IM.im_load_texture(data, len(data), ctypes.byref(w), ctypes.byref(h))
    return Texture(ctypes.c_void_p(handle), w.value, h.value)

def draw_texture(texture: Texture, pos: Collection[int], size: float, angle: float, anchor: Collection[float] = (.5, .5)) -> None:
    _IM.im_draw_texture(texture.handle, pos[0], pos[1], size, angle, anchor[0], anchor[1])

def rotated_rectangle_extents(width: float, height: float, angle: float) -> Rect:
    w = ctypes.c_int()
    h = ctypes.c_int()
    _IM.im_rotated_rectangle_extents(width, height, angle, ctypes.byref(w), ctypes.byref(h))
    return Rect(0, 0, w.value, h.value)

key_states: dict[str, bool] = {}
def key_state(name: str) -> bool:
    return key_states.get(name, False)

def mouse_pos() -> tuple[int, int]:
    x = ctypes.c_int()
    y = ctypes.c_int()
    _IM.im_mouse_pos(ctypes.byref(x), ctypes.byref(y))
    return (x.value, y.value)

@contextmanager
def frame() -> Generator[FrameInfo, None, None]:
    _IM.im_begin_frame()
    n = ctypes.c_size_t()
    ke_raw = _IM.im_get_key_events(ctypes.byref(n))
    key_events = [KeyEvent(i.keyname.decode("ascii"), bool(i.held)) for i in ke_raw[:n.value]]
    for event in key_events:
        key_states[event.key] = event.held
    try:
        yield FrameInfo(key_events)
    finally:
        _IM.im_end_frame()

def stop() -> None:
    _IM.im_stop()

def end() -> None:
    _IM.im_quit()

def mainloop() -> Generator[FrameInfo, None, None]:
    while True:
        with frame() as info:
            yield info
        if not _IM.im_running():
            break

# Scratch stuff
def draw_thought_bubble(text: str, pos: Sequence[float]) -> None:
    _IM.im_draw_thought_bubble(text.encode("utf8"), int(pos[0]), int(pos[1]))
