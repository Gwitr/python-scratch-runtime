# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Collection, IO, Self
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

_IM.im_tex_to_mask.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
_IM.im_tex_to_mask.restype = ctypes.POINTER(ctypes.c_uint64)

_IM.im_empty_mask.argtypes = (ctypes.c_int, ctypes.c_int)
_IM.im_empty_mask.restype = ctypes.POINTER(ctypes.c_uint64)

class timespec(ctypes.Structure):
    _fields_ = (
        ("tv_sec", ctypes.c_time_t),
        ("tv_nsec", ctypes.c_long)
    )

class im_text_input_state(ctypes.Structure):
    _fields_ = (
        ("text", ctypes.c_char_p),
        ("cursor_location", ctypes.c_int),
        ("edit_start", timespec),
        ("closed", ctypes.c_int)
    )

class TextInputState:
    struct: ctypes.POINTER(im_text_input_state)

    def __init__(self, struct):
        self.struct = struct

    def __getattr__(self, key):
        return getattr(self.struct.contents, key)

    def __del__(self):
        libc.free(self.struct)

_IM.im_draw_question_box.argtypes = (ctypes.POINTER(im_text_input_state),)
_IM.im_draw_question_box.restype = ctypes.POINTER(im_text_input_state)

@dataclass(slots=True, frozen=True)
class KeyEvent:
    key: str
    held: bool

@dataclass
class FrameInfo:
    key_events: list[KeyEvent]

@dataclass(slots=True, frozen=True)
class Texture:
    handle: ctypes.c_void_p
    w: int
    h: int

    @classmethod
    def from_file(cls, file: IO[bytes]) -> Self:
        data = file.read()
        w = ctypes.c_int()
        h = ctypes.c_int()
        handle = _IM.im_load_texture(data, len(data), ctypes.byref(w), ctypes.byref(h))
        return cls(ctypes.c_void_p(handle), w.value, h.value)

    def draw(self, pos: Collection[int], size: float, angle: float, anchor: Collection[float] = (.5, .5)) -> None:
        _IM.im_draw_texture(self.handle, pos[0], pos[1], size, angle, anchor[0], anchor[1])

    def __del__(self):
        _IM.im_free_texture(self.handle)

@dataclass(slots=True, frozen=True)
class Mask:
    handle: ctypes.POINTER(ctypes.c_uint64)
    w: int
    h: int

    def __del__(self):
        libc.free(self.handle)

    @classmethod
    def empty(cls, width: int, height: int, default: bool) -> Self:
        return cls(_IM.im_empty_mask(width, height, default), width, height)

    @classmethod
    def from_file(cls, file: IO[bytes]) -> Self:
        data = file.read()
        w = ctypes.c_int()
        h = ctypes.c_int()
        handle = _IM.im_load_mask(data, len(data), ctypes.byref(w), ctypes.byref(h))
        return cls(handle, w.value, h.value)

    @classmethod
    def from_texture(cls, tex: Texture, angle: float, scale: float) -> Self:
        w = ctypes.c_int()
        h = ctypes.c_int()
        handle = _IM.im_tex_to_mask(tex.handle, angle, scale, ctypes.byref(w), ctypes.byref(h))
        return cls(handle, w.value, h.value)

    def overlap(self, mask2: Mask, offs: tuple[int, int]) -> tuple[int, int] | None:
        x = ctypes.c_int()
        y = ctypes.c_int()
        if _IM.im_fast_overlap(self.handle, self.w, self.h, mask2.handle, mask2.w, mask2.h, int(offs[0]), int(offs[1]), ctypes.byref(x), ctypes.byref(y)):
            return (x.value, y.value)
        return None

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

def init() -> None:
    _IM.im_init()

window_width: int = 0
window_height: int = 0
window_fps: int = 0
def set_context(width: int, height: int, fps: int = 30) -> None:
    global window_width, window_height, window_fps  # pylint: disable=global-statement
    window_width = width
    window_height = height
    window_fps = fps
    _IM.im_set_context(width, height, fps)

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
    key_events = [KeyEvent(i.keyname.decode("ascii"), bool(i.held)) for i in ke_raw[:n.value] if i.keyname is not None]
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

def draw_question_box(state: TextInputState | None) -> TextInputState:
    if state is None:
        return TextInputState(_IM.im_draw_question_box(None))
    _IM.im_draw_question_box(state.struct)
    return state
