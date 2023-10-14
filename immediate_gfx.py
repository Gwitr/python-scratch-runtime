# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

from __future__ import annotations

import copy
import time
from collections.abc import Sequence, Generator
from typing import Collection, Any, TypeVar, cast, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

import pygame
import numpy as np
import numpy.typing as npt

T = TypeVar("T")

def non_optional(x: T | None) -> T:
    assert x is not None
    return x

def namespace(cls: type[T]) -> T:
    return dataclass(cls)()

@namespace
class Context:
    display: pygame.Surface | None = None
    clock: pygame.time.Clock | None = None
    fps: int | None = None
    bg: pygame.Surface | None = None
    font_cache: dict[tuple[str, int], pygame.font.Font] = field(default_factory=lambda: {})

def preload_font(name: str, size: int, system_font: bool=False) -> None:
    if system_font:
        Context.font_cache["sys:" * system_font + name, size] = pygame.font.SysFont(name, size)
    else:
        Context.font_cache["sys:" * system_font + name, size] = pygame.font.Font(name, size)

def get_font(name: str, size: int, system_font: bool=False) -> pygame.font.Font:
    if ("sys:" * system_font + name, size) not in Context.font_cache:
        preload_font(name, size, system_font)
    return Context.font_cache["sys:" * system_font + name, size]

def set_context(width: int, height: int, fps: int=30) -> None:
    display = pygame.display.set_mode((width, height))
    Context.display = display
    Context.clock = pygame.time.Clock()
    Context.fps = fps
    Context.bg = pygame.Surface((width, height), pygame.SRCALPHA).convert_alpha()
    Context.font_cache = {}

def bg_texture() -> pygame.Surface:
    return cast(pygame.Surface, Context.bg)

def draw_texture(texture: pygame.Surface, pos: Collection[int], size: float, angle: float,
                 anchor: Collection[float] = (.5, .5)) -> None:
    surf = pygame.transform.rotozoom(texture, angle, size)
    final_pos: npt.NDArray[np.float_] = -np.array(anchor) * surf.get_size() + pos * np.array((1, -1)) + np.array(non_optional(Context.display).get_size()) / 2
    non_optional(Context.display).blit(surf, cast(Sequence[int], final_pos.astype(np.int32)))

def rotated_rectangle_extents(width: float, height: float, angle: float) -> pygame.Rect:
    corners = [
        (-width // 2, -height // 2),
        (-width // 2, +height // 2),
        (+width // 2, +height // 2),
        (+width // 2, -height // 2),
    ]
    angle = np.radians(90.0 - angle)
    for idx, corner in enumerate(corners):
        corners[idx] = (
            corner[0] * np.cos(angle) - corner[1] * np.sin(angle),
            corner[0] * np.sin(angle) + corner[1] * np.cos(angle)
        )

    minx, maxx = min(i[0] for i in corners), max(i[0] for i in corners)
    miny, maxy = min(i[1] for i in corners), max(i[1] for i in corners)
    return pygame.Rect(0, 0, maxx-minx, maxy-miny)

@contextmanager
def frame() -> Generator[None, None, None]:
    non_optional(Context.clock).tick(non_optional(Context.fps))
    non_optional(Context.display).fill((255, 255, 255))
    if Context.bg:
        non_optional(Context.display).blit(Context.bg, (0, 0))
    yield
    pygame.display.flip()


# Scratch stuff
def draw_thought_bubble(text: str, pos: Sequence[float]) -> None:
    font = get_font("Arial", 14, system_font=True)

    split_text = [i + " " for i in str(text).split() if i != ""]
    lines = [""]
    for word in split_text:
        text_width = font.size(word)[0]
        line_width = font.size(lines[-1])[0]
        if line_width + text_width > 108:
            lines.append(word)
        else:
            lines[-1] += word
    width = max(32 + max(font.size(i)[0] for i in lines), 55)
    height = 41 + len(lines) * 22

    bubble = pygame.Surface((width, height), pygame.SRCALPHA).convert_alpha()
    pygame.draw.rect(bubble, (255, 255, 255), (4, 4, width - 4, height - 24), border_radius=12)
    pygame.draw.rect(bubble, (220, 220, 220), (4, 4, width - 4, height - 24), width=2,
                     border_radius=12)
    pygame.draw.circle(bubble, (255, 255, 255), (width//2 + 2, height - 20), 8)
    pygame.draw.circle(bubble, (220, 220, 220), (width//2 + 2, height - 22), 8, width=2,
                       draw_bottom_left=True, draw_bottom_right=True, draw_top_left=False,
                       draw_top_right=False)
    pygame.draw.circle(bubble, (255, 255, 255), (width//2 + 9, height - 13), 4)
    pygame.draw.circle(bubble, (220, 220, 220), (width//2 + 9, height - 13), 4, width=2)
    pygame.draw.circle(bubble, (255, 255, 255), (width//2 + 16, height - 10), 3)
    pygame.draw.circle(bubble, (220, 220, 220), (width//2 + 16, height - 10), 3, width=2)

    for lineno, line in enumerate(lines):
        bubble.blit(font.render(line, True, (20, 20, 20)), (16, 16 + lineno * 22))

    non_optional(Context.display).blit(bubble, (
        pos[0] + 240 - bubble.get_width() * 3 // 4,
        -pos[1] + 180 - bubble.get_height() + 7,
    ))

@dataclass
class TextInputState:
    text: str
    cursor_location: int | None
    edit_start: float
    closed: bool

def draw_question_box(state: TextInputState | None = None, key_events: list[pygame.event.Event] | None = None) -> TextInputState:
    font = get_font("calibril.ttf", 12)

    if state is None:
        state = TextInputState("", None, time.perf_counter(), False)
    else:
        state = copy.copy(state)
    if key_events is None:
        key_events = []

    edit_bounds = pygame.Rect(27, 301, 428, 32)
    if pygame.mouse.get_pressed()[0] and edit_bounds.collidepoint(pygame.mouse.get_pos()):
        state.cursor_location = 0
        state.edit_start = time.perf_counter()

    if state.cursor_location is not None:
        for event in key_events:
            if event.type == pygame.TEXTINPUT:
                state.edit_start = time.perf_counter()
                state.text = state.text[:state.cursor_location] + event.text + state.text[state.cursor_location:]
                state.cursor_location += len(event.text)

            elif event.type == pygame.KEYDOWN:
                state.edit_start = time.perf_counter()
                if event.key == pygame.K_BACKSPACE:
                    state.text = (
                        state.text[:max(state.cursor_location-1,0)] + state.text[state.cursor_location:]
                    )
                    if state.cursor_location > 0:
                        state.cursor_location -= 1
                elif event.key == pygame.K_DELETE:
                    state.text = (
                        state.text[:state.cursor_location] + state.text[state.cursor_location+1:]
                    )
                elif event.key == pygame.K_LEFT:
                    if state.cursor_location > 0:
                        state.cursor_location -= 1
                elif event.key == pygame.K_RIGHT:
                    if state.cursor_location < len(state.text):
                        state.cursor_location += 1
                elif event.key == pygame.K_RETURN:
                    state.closed = True

    pygame.draw.rect(non_optional(Context.display), (255, 255, 255), (7, 284, 468, 66), border_radius=10)
    pygame.draw.rect(non_optional(Context.display), (210, 210, 210), (7, 284, 468, 66), width=2,
                     border_radius=10)

    pygame.draw.rect(non_optional(Context.display), (255, 255, 255), (27, 301, 428, 32), border_radius=15)
    pygame.draw.rect(non_optional(Context.display), (210, 210, 210), (27, 301, 428, 32), width=1, border_radius=15)

    pygame.draw.circle(non_optional(Context.display), (135, 80, 195), (438, 317), 13)

    pygame.draw.line(non_optional(Context.display), (255, 255, 255), (432, 322), (429, 318), 5)
    pygame.draw.line(non_optional(Context.display), (255, 255, 255), (433, 322), (441, 314), 5)
    pygame.draw.circle(non_optional(Context.display), (255, 255, 255), (429, 318), 2)
    pygame.draw.circle(non_optional(Context.display), (255, 255, 255), (433, 323), 2)

    pygame.draw.circle(non_optional(Context.display), (255, 255, 255), (442, 314), 2)

    non_optional(Context.display).blit(font.render(state.text, True, (50, 50, 50)), (38, 311))
    if state.cursor_location is not None and (time.perf_counter() - state.edit_start) % 2 <= 1:
        width = font.size(state.text[:state.cursor_location])[0]
        pygame.draw.line(non_optional(Context.display), (0, 0, 0), (38 + width, 311), (38 + width, 321))

    return state
