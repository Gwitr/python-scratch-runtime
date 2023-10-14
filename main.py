# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

from __future__ import annotations

import inspect
import re
import io
import abc
import math
import json
import enum
import time
import types
import zipfile
from collections.abc import Callable, Sequence, Generator
from typing import cast, TypeVar, Any, IO, ClassVar
from dataclasses import dataclass, field
from xml.etree.ElementTree import ElementTree

import pygame
import typing_inspect
from defusedxml.ElementTree import parse
from cairosvg import svg2png
import cairosvg.helpers

import immediate_gfx as IM

HAT_BLOCKS = ["event_whenflagclicked", "event_whenbroadcastreceived", "event_whenkeypressed", "control_start_as_clone"]

SCRATCH_KEY_TO_PYGAME = {
    "q": pygame.K_q, "w": pygame.K_w, "e": pygame.K_e, "r": pygame.K_r,
    "t": pygame.K_t, "y": pygame.K_y, "u": pygame.K_u, "i": pygame.K_i,
    "o": pygame.K_o, "p": pygame.K_p, "a": pygame.K_a, "s": pygame.K_s,
    "d": pygame.K_d, "f": pygame.K_f, "g": pygame.K_g, "h": pygame.K_h,
    "j": pygame.K_j, "k": pygame.K_k, "l": pygame.K_l, "z": pygame.K_z,
    "x": pygame.K_x, "c": pygame.K_c, "v": pygame.K_v, "b": pygame.K_b,
    "n": pygame.K_n, "m": pygame.K_m, "1": pygame.K_1, "2": pygame.K_2,
    "3": pygame.K_3, "4": pygame.K_4, "5": pygame.K_5, "6": pygame.K_6,
    "7": pygame.K_7, "8": pygame.K_8, "9": pygame.K_9, "0": pygame.K_0,
    "space": pygame.K_SPACE, "left arrow": pygame.K_LEFT,
    "right arrow": pygame.K_RIGHT, "up arrow": pygame.K_UP,
    "down arrow": pygame.K_DOWN
}
PYGAME_KEY_TO_SCRATCH = {v: k for k, v in SCRATCH_KEY_TO_PYGAME.items()}

EVENT_CLONE_SCRIPT_INSTANCES: int | None = None
EVENT_BROADCAST: int | None = None

T = TypeVar("T")
ScratchValue = float | str

def non_optional(x: T | None) -> T:
    assert x is not None
    return x

def as_float(value: ScratchValue) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0

def as_string(value: ScratchValue) -> str:
    return str(value)

def as_bool(value: ScratchValue | bool | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value != 0.0
    return value.lower() not in {"false", ""}

class Stop(Exception):
    pass

class StopAll(Stop):
    pass

class StopThisScript(Stop):
    pass

@dataclass
class Variable:
    name: str
    value: ScratchValue

    @classmethod
    def load(cls, data: list[Any]) -> Variable:
        return cls(*data)

@dataclass
class Costume(abc.ABC):
    _SUBCLASSES = None
    FMT: ClassVar[str | None] = None

    name: str
    md5: str | None
    origin: tuple[float, float]

    @property
    @abc.abstractmethod
    def width(self) -> int: ...

    @property
    @abc.abstractmethod
    def height(self) -> int: ...

    @abc.abstractmethod
    def render(self, *, angle: float=90.0, scale: float=1.0, pixel: float=0.0,
               mosaic: float=0.0, ghost: float=0.0) -> pygame.Surface: ...

    def draw(self, position: Sequence[int], *, angle: float=90.0, scale: float=1.0, pixel: float=0.0,
             mosaic: float=0.0, ghost: float=0.0) -> None:
        surf = self.render(angle=angle, scale=scale, pixel=pixel, mosaic=mosaic, ghost=ghost)
        IM.draw_texture(surf, position, 1.0, 0.0)

    @classmethod
    @abc.abstractmethod
    def _load(cls, file: IO[bytes] | None, name: str, md5: str, origin: Sequence[float]) -> Costume: ...

    @classmethod
    def _get_subclasses(cls) -> set[type[Costume]]:
        subclasses = cls.__subclasses__()
        all_subclasses = set()
        for i in subclasses:
            all_subclasses.add(i)
            all_subclasses.update(i._get_subclasses())  # pylint: disable=protected-access
        return all_subclasses

    @classmethod
    def load(cls, sb3: zipfile.PyZipFile, data: dict[Any, Any]) -> Costume:
        if cls._SUBCLASSES is None:
            cls._SUBCLASSES = {scls.FMT: scls for scls in cls._get_subclasses()}

        fmt = data["dataFormat"]
        try:
            costume_cls = cls._SUBCLASSES[fmt]
        except KeyError:
            raise ValueError(f"Unknown format: {fmt}") from None
        if "md5ext" in data:
            with sb3.open(data["md5ext"]) as file:
                return costume_cls._load(  # pylint: disable=protected-access
                    file, data["name"], data["assetId"],
                    (data["rotationCenterX"], data["rotationCenterY"])
                )
        else:
            return costume_cls._load(  # pylint: disable=protected-access
                None, data["name"], data["assetId"],
                (data["rotationCenterX"], data["rotationCenterY"])
            )

DPI = 96  # pixels per inch

@dataclass
class SvgCostume(Costume):
    FMT = "svg"

    SVG_SCALES = 0.25, 0.5, 1.0, 2.0, 4.0
    NEUTRAL_SCALE = 2

    scaled_surfaces: list[pygame.Surface]

    @property
    def width(self) -> int:
        return self.scaled_surfaces[self.NEUTRAL_SCALE].get_width()

    @property
    def height(self) -> int:
        return self.scaled_surfaces[self.NEUTRAL_SCALE].get_height()

    def render(self, *, angle: float=90.0, scale: float=1.0, pixel: float=0.0, mosaic: float=0.0, ghost: float=0.0) -> pygame.Surface:
        if pixel or mosaic or ghost:
            raise NotImplementedError("pixel/mosaic/ghost effects")
        surface_idx, base_scale = min(
            enumerate(self.SVG_SCALES),
            key=lambda x: float("inf") if self.scaled_surfaces[x[0]] is None else abs(scale - x[1])
        )
        extra_scale = round(scale / base_scale, 3)

        base_surf = self.scaled_surfaces[surface_idx]
        # TODO: Make use of the rotation origin
        surf = pygame.transform.rotozoom(base_surf, 90.0 - angle, extra_scale)
        # width, height = base_surf.get_size()
        # scaled_surf = pygame.transform.scale(base_surf, (int(width * extra_scale), int(height * extra_scale)))
        # surf = pygame.transform.rotate(scaled_surf, 90.0 - angle)
        return surf

    @classmethod
    def _load(cls, file: IO[bytes] | None, name: str, md5: str, origin: Sequence[float]) -> SvgCostume:
        # CairoSVG has trouble figuring out the width/height sometimes, so we have to help it
        if file is None:
            return cls(name, None, cast(tuple[float, float], origin), [
                pygame.Surface((1, 1), pygame.SRCALPHA).convert_alpha() for _ in cls.SVG_SCALES
            ])
        svg_attrib = cast(ElementTree, parse(file)).getroot().attrib

        dims_regex = re.compile(
            r"\s*((?:0|[1-9]\d*)(?:\.\d*)?)\s*(?:(ex|px|pt|pc|cm|mm|in)\s*)?", re.IGNORECASE
        )
        try:
            dims_raw: list[str | float] = [svg_attrib["width"], svg_attrib["height"]]
        except KeyError:
            width = height = None
        else:
            for idx, dim_raw in enumerate(dims_raw):
                if (match := dims_regex.match(cast(str, dim_raw))) is None:
                    raise ValueError(f"Invalid SVG: cannot understand what a length of '{dim_raw}' is")
                length, unit = match.groups()
                length = float(length)
                if unit not in {"px", None}:
                    length *= cairosvg.helpers.UNITS[unit] * DPI
                dims_raw[idx] = length
            width, height = cast(list[float], dims_raw)

        file.seek(0)
        scaled_surfaces: list[pygame.Surface | None] = []
        data = file.read()
        for scale in cls.SVG_SCALES:
            if width is None or height is None:
                pngdata = svg2png(bytestring=data, scale=scale)
            else:
                if width * scale < 1 or height * scale < 1:
                    scaled_surfaces.append(None)
                    continue
                pngdata = svg2png(bytestring=data, scale=scale,
                                  output_width=int(math.ceil(width * scale)),
                                  output_height=int(math.ceil(height * scale)))

            scaled_surfaces.append(pygame.image.load(io.BytesIO(non_optional(pngdata)), "png").convert_alpha())
        return cls(name, md5, cast(tuple[float, float], origin), cast(list[pygame.Surface], scaled_surfaces))

@dataclass
class Sound:
    name: str
    md5: str
    clip: pygame.mixer.Sound

    volume: float = field(init=False)
    pitch: float = field(init=False)
    pan: float = field(init=False)
    channel: pygame.mixer.Channel | None = field(init=False)

    def __post_init__(self) -> None:
        self.volume = 1.0
        self.pitch = 0.0
        self.pan = 0.0
        self.channel = None

    def _check_playing(self) -> None:
        if self.channel is not None and not self.channel.get_busy():
            self.channel = None

    def update_effects(self, *, pitch: float=0.0, pan: float=0.0, volume: float=0.0) -> None:
        self._check_playing()

        self.volume = volume
        self.pitch = pitch
        self.pan = pan
        if self.channel is not None:
            self.update_channel_settings()

    def update_channel_settings(self) -> None:
        left_volume = self.volume * min(2 - (self.pan + 1), 1)
        right_volume = self.volume * min(self.pan + 1, 1)
        non_optional(self.channel).set_volume(left_volume, right_volume)

    def play(self) -> None:
        self._check_playing()

        if self.channel:
            # peculiar Scratch behavior; the same sound can't be played multiple times simultanously
            self.channel.stop()
        self.channel = pygame.mixer.find_channel(force=True)
        self.update_channel_settings()
        self.channel.play(self.clip)

    @classmethod
    def load(cls, sb3: zipfile.PyZipFile, data: dict[Any, Any]) -> Sound:
        with sb3.open(data["md5ext"]) as file:
            return cls(data["name"], data["assetId"], pygame.mixer.Sound(file))

@dataclass(repr=False)
class VariableReference:
    target: Target
    name: str

    def set_target(self, target: Target) -> None:
        self.target = target

    def evaluate(self, lvalue: bool=False) -> Generator[list[Any], ScratchValue, ScratchValue | Variable]:
        try:
            var = [i for i in self.target.variables if i.name == self.name][0]
        except IndexError:
            var = [i for i in self.target.project.stage.variables if i.name == self.name][0]
        if lvalue:
            return var
        return var.value
        yield  # All evaluate functions are generators for now

    def __repr__(self) -> str:
        return self.name

@dataclass
class Block:
    shadow: bool

    target: Target | None = field(repr=False)
    opcode: str
    arguments: dict[str, ScratchValue | VariableReference | Block | BlockList]
    fields: dict[str, tuple[str, ScratchValue | None]]

    Unevaluated = Callable[[], Generator[ScratchValue, ScratchValue, T]]
    EvalFunctionReturnType = Generator[Any, Any, Any] | ScratchValue | bool | None
    EvalFunctionType = Callable[..., EvalFunctionReturnType]

    next_block: dict[Any, Any] | None = field(init=False, repr=False)
    parent_block: dict[Any, Any] | None = field(init=False, repr=False)

    EVAL_FUNCTIONS: ClassVar[dict[str, EvalFunctionType]] = {}

    def __post_init__(self) -> None:
        # Attributes only to help with parsing the JSON data
        self.next_block = None
        self.parent_block = None

    def set_target(self, target: Target) -> None:
        self.target = target
        for value in self.arguments.values():
            if isinstance(value, (Block, BlockList, VariableReference)):
                value.set_target(target)

    def evaluate_argument(self, argname: str, lvalue: bool=False) -> Generator[Any, Any, Any]:
        arg = self.arguments[argname]
        if isinstance(arg, (Block, BlockList)):
            return (yield from cast(Generator[Any, Any, Any], arg.evaluate()))
        if isinstance(arg, VariableReference):
            return (yield from arg.evaluate(lvalue))
        return arg

    @classmethod
    def register_evaluator(cls, opcode_name: str) -> Callable[[EvalFunctionType], EvalFunctionType]:
        def decorator(f: Block.EvalFunctionType) -> Block.EvalFunctionType:
            cls.EVAL_FUNCTIONS[opcode_name] = f
            return f
        return decorator

    def evaluate(self) -> EvalFunctionReturnType:
        if self.opcode in HAT_BLOCKS:
            return None
        try:
            func = self.EVAL_FUNCTIONS[self.opcode]
        except KeyError:
            print(f"Note: executing unknown block {self!r}")
            return None
        spec = inspect.getfullargspec(func)
        args: list[ScratchValue | Callable[..., Any] | int] = []
        for argname in spec.args[1:]:
            match eval(spec.annotations[argname]):
                case x if x == float:
                    args.append(as_float((yield from self.evaluate_argument(argname))))
                case x if x == int:
                    args.append(int(as_float((yield from self.evaluate_argument(argname)))))
                case x if x == str:
                    args.append(as_string((yield from self.evaluate_argument(argname))))
                case x if x == bool:
                    args.append(as_bool((yield from self.evaluate_argument(argname))))
                case x if x == ScratchValue:
                    args.append((yield from self.evaluate_argument(argname)))
                case x if cast(Any, typing_inspect.get_origin(x)) == cast(Any, Callable) and typing_inspect.get_args(x, True)[0] == [] and typing_inspect.is_generic_type(typing_inspect.get_args(x, True)[1]) and typing_inspect.get_args(typing_inspect.get_args(x, True)[1], True)[:2] == (ScratchValue, ScratchValue):
                    args.append(lambda *, x=argname: self.evaluate_argument(x))
                case x:
                    raise ValueError("can't cast ScratchValue to %r" % x.__qualname__)
        
        kwargs: dict[str, VariableReference | ScratchValue | None] = {}
        for kwargname in spec.kwonlyargs:
            match eval(spec.annotations.get(kwargname, "None")):
                case x if x == VariableReference:
                    kwargs[kwargname] = VariableReference(non_optional(self.target), self.fields[kwargname][0])
                case _:
                    kwargs[kwargname] = self.fields[kwargname][0]
        
        if spec.annotations[spec.args[0]] is not Target:
            if not isinstance(self.target, eval(spec.annotations[spec.args[0]])):
                return

        result = func(self.target, *args, **kwargs)
        if isinstance(result, types.GeneratorType):
            return (yield from result)
        return result

    # def __repr__(self) -> str:
        # args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        # return f"{self.opcode}({args})"

@Block.register_evaluator("motion_movesteps")
def op_motion_movesteps(sprite: Sprite, STEPS: float) -> None:
    angle = math.radians(90.0 - sprite.angle)
    sprite.xpos += STEPS * math.cos(angle)
    sprite.ypos += STEPS * math.sin(angle)

@Block.register_evaluator("motion_goto")
def op_motion_goto(sprite: Sprite, TO: str) -> None:
    if len(targets := [i for i in sprite.project.targets if isinstance(i, Sprite) and i.name == TO and not i.is_clone]) == 0:
        print(f"motion_goto: no such target {TO}")
        return None
    sprite.xpos = targets[0].xpos
    sprite.ypos = targets[0].ypos
    return None

@Block.register_evaluator("motion_goto_menu")
def op_motion_goto_menu(sprite: Sprite, *, TO: str) -> str:
    return TO

@Block.register_evaluator("motion_gotoxy")
def op_motion_gotoxy(sprite: Sprite, X: float, Y: float) -> None:
    sprite.xpos = X
    sprite.ypos = Y

@Block.register_evaluator("motion_changexby")
def op_motion_changexby(sprite: Sprite, DX: float) -> None:
    sprite.xpos += DX

@Block.register_evaluator("motion_changeyby")
def op_motion_changeyby(sprite: Sprite, DY: float) -> None:
    sprite.ypos += DY

@Block.register_evaluator("motion_pointindirection")
def op_motion_pointindirection(sprite: Sprite, DIRECTION: float) -> None:
    sprite.angle = DIRECTION

@Block.register_evaluator("motion_turnright")
def op_motion_turnright(sprite: Sprite, DEGREES: float) -> None:
    sprite.angle += DEGREES

@Block.register_evaluator("motion_turnleft")
def op_motion_turnleft(sprite: Sprite, DEGREES: float) -> None:
    sprite.angle -= DEGREES

@Block.register_evaluator("motion_xposition")
def op_motion_xposition(sprite: Sprite) -> float:
    return sprite.xpos

@Block.register_evaluator("motion_yposition")
def op_motion_yposition(sprite: Sprite) -> float:
    return sprite.ypos

@Block.register_evaluator("looks_show")
def op_looks_show(sprite: Sprite) -> None:
    sprite.visible = True

@Block.register_evaluator("looks_hide")
def op_looks_hide(sprite: Sprite) -> None:
    sprite.visible = False

@Block.register_evaluator("looks_thinkforsecs")
def op_looks_thinkforsecs(sprite: Sprite, MESSAGE: str, SECS: float) -> Generator[Any, Any, None]:
    # TODO: Don't hard-code FPS; also use an event instead of this
    sprite.bubble = MESSAGE
    for _ in range(int(30 * SECS)):
        yield []
    sprite.bubble = None

@Block.register_evaluator("looks_sayforsecs")
def op_looks_sayforsecs(sprite: Sprite, MESSAGE: str, SECS: float) -> Generator[Any, Any, None]:
    # TODO: Use another type of bubble; also don't hardcode FPS
    sprite.bubble = MESSAGE
    for _ in range(int(30 * SECS)):
        yield []
    sprite.bubble = None

@Block.register_evaluator("control_delete_this_clone")
def op_control_delete_this_clone(sprite: Sprite) -> None:
    if sprite.is_clone and sprite in sprite.project.targets:
        sprite.project.targets.remove(sprite)

@Block.register_evaluator("control_stop")
def op_control_stop(_target: Target, *, STOP_OPTION: str) -> None:
    if STOP_OPTION == "all":
        raise StopAll
    elif STOP_OPTION == "this script":
        raise StopThisScript
    print(f"op_control_stop: unknown stop option {STOP_OPTION}")

@Block.register_evaluator("control_forever")
def op_control_forever(_target: Target, SUBSTACK: Block.Unevaluated[None]) -> Generator[Any, Any, None]:
    while True:
        yield from SUBSTACK()
        yield []

@Block.register_evaluator("control_if")
def op_control_if(_target: Target, CONDITION: bool, SUBSTACK: Block.Unevaluated[None]) -> Generator[Any, Any, None]:
    if CONDITION:
        yield from SUBSTACK()

@Block.register_evaluator("control_repeat_until")
def op_control_repeat_until(_target: Target, CONDITION: Block.Unevaluated[bool], SUBSTACK: Block.Unevaluated[None]) -> Generator[Any, Any, None]:
    while not bool((yield from CONDITION())):
        yield from SUBSTACK()
        yield []

@Block.register_evaluator("control_wait")
def op_control_wait(_target: Target, DURATION: float) -> Generator[Any, Any, None]:
    # TODO: Don't hard-code FPS; also use an event instead of this
    for _ in range(int(30 * DURATION)):
        yield []

@Block.register_evaluator("control_create_clone_of")
def op_control_create_clone_of(target: Target, CLONE_OPTION: str) -> None:
    if len(targets := [i for i in target.project.targets if isinstance(i, Sprite) and i.name == CLONE_OPTION and not i.is_clone]) == 0:
        return None
    to_clone = targets[0]

    def copy_block(j: Block) -> Block:
        return Block(j.shadow, None, j.opcode, cast(Any, {k: copy_blocklist(v) if isinstance(v, BlockList) else (VariableReference(None, v.name) if isinstance(v, VariableReference) else v) for k, v in j.arguments.items()}), j.fields)

    def copy_blocklist(i: BlockList) -> BlockList:
        return BlockList(None, [copy_block(j) for j in i.blocks])

    blocks = [copy_blocklist(i) for i in to_clone.blocks]
    # breakpoint()
    to_clone.project.targets.append(Sprite(
        to_clone.project, to_clone.name, [Variable(i.name, i.value) for i in to_clone.variables], None,
        to_clone.broadcasts, blocks, to_clone.comments, to_clone.costumes,
        to_clone.sounds, to_clone.costume, to_clone.volume, to_clone.visible,
        to_clone.xpos, to_clone.ypos, to_clone.scale, to_clone.angle,
        to_clone.draggable, to_clone.rotation_style, to_clone.bubble
    ))
    to_clone.project.targets[-1].is_clone = True
    for blocklist in to_clone.project.targets[-1].blocks:
        blocklist.set_target(to_clone.project.targets[-1])
    pygame.event.post(pygame.event.Event(non_optional(EVENT_CLONE_SCRIPT_INSTANCES), target=to_clone.project.targets[-1]))
    return None

@Block.register_evaluator("control_create_clone_of_menu")
def op_control_create_clone_of_menu(_target: Target, *, CLONE_OPTION: str) -> str:
    return CLONE_OPTION

@Block.register_evaluator("sensing_touchingobject")
def op_sensing_touchingobject(sprite: Sprite, TOUCHINGOBJECTMENU: str) -> bool:
    # TODO: Don't hard-code screen size

    if not sprite.visible:
        return False

    if TOUCHINGOBJECTMENU == "_mouse_":
        mousex, mousey = pygame.mouse.get_pos()
        # bbox = sprite.bounding_box()
        bbox = IM.rotated_rectangle_extents(
            sprite.width * sprite.scale / 100,
            sprite.height * sprite.scale / 100,
            sprite.angle
        )
        result = sprite.mask.overlap(
            pygame.mask.from_surface(pygame.Surface((1, 1))),
            (int(mousex - 240 - sprite.xpos + bbox[2] / 2),
                int(non_optional(IM.Context.display).get_height() - mousey - 180 - sprite.ypos + bbox[3] / 2))
        ) is not None
        return result

    elif TOUCHINGOBJECTMENU == "_edge_":
        bbox = IM.rotated_rectangle_extents(
            sprite.width * sprite.scale / 100,
            sprite.height * sprite.scale / 100,
            sprite.angle
        )
        return (sprite.xpos + bbox[2] / 2 >= 240 or sprite.xpos - bbox[2] / 2 <= -240 or
                sprite.ypos + bbox[3] / 2 >= 180 or sprite.ypos - bbox[3] / 2 <= -180)

    elif len(targets := [i for i in sprite.project.targets if i.name == TOUCHINGOBJECTMENU and isinstance(i, Sprite)]) > 0:
        return any(target.visible and sprite.mask.overlap(target.mask, (target.xpos - sprite.xpos, target.ypos - sprite.ypos)) is not None for target in targets)

    print(f"Note: unknown touching object setting {TOUCHINGOBJECTMENU}")
    return False

@Block.register_evaluator("sensing_keypressed")
def op_sensing_keypressed(_target: Target, KEY_OPTION: str) -> bool:
    key_to_check = SCRATCH_KEY_TO_PYGAME[KEY_OPTION]
    # print(KEY_OPTION, "=>", pygame.key.get_pressed()[key_to_check])
    return pygame.key.get_pressed()[key_to_check]

# Operator blocks
@Block.register_evaluator("operator_add")
def op_operator_add(_target: Target, NUM1: float, NUM2: float) -> float:
    return NUM1 + NUM2

@Block.register_evaluator("operator_subtract")
def op_operator_subtract(_target: Target, NUM1: float, NUM2: float) -> float:
    return NUM1 - NUM2

@Block.register_evaluator("operator_multiply")
def op_operator_multiply(_target: Target, NUM1: float, NUM2: float) -> float:
    return NUM1 * NUM2

@Block.register_evaluator("operator_divide")
def op_operator_divide(_target: Target, NUM1: float, NUM2: float) -> ScratchValue:
    if NUM2 == 0:
        if NUM1 == 0:
            return "NaN"
        return ["-Infinity", "", "Infinity"][round(NUM2 / abs(NUM2) + 1)]
    return NUM1 / NUM2

@Block.register_evaluator("operator_mathop")
def op_operator_mathop(_target: Target, NUM: float, *, OPERATOR: str) -> float:
    match OPERATOR:
        case "abs":
            return abs(NUM)
    print(f"operator_mathop: unknown operator {OPERATOR!r}")
    return 0.0

@Block.register_evaluator("operator_equals")
def op_operator_equals(_target: Target, OPERAND1: ScratchValue, OPERAND2: ScratchValue) -> bool:
    try:
        return float(OPERAND1) == float(OPERAND2)
    except ValueError:
        pass
    return str(OPERAND1) == str(OPERAND2)

@Block.register_evaluator("data_setvariableto")
def op_data_setvariableto(_target: Target, VALUE: ScratchValue, *, VARIABLE: VariableReference) -> Generator[Any, Any, None]:
    cast(Variable, (yield from VARIABLE.evaluate(True))).value = VALUE

@Block.register_evaluator("data_changevariableby")
def op_data_changevariableby(_target: Target, VALUE: float, *, VARIABLE: VariableReference) -> Generator[Any, Any, None]:
    var = cast(Variable, (yield from VARIABLE.evaluate(True)))
    var.value = as_float(var.value) + VALUE

@Block.register_evaluator("sensing_touchingobjectmenu")
def op_sensing_touchingobjectmenu(_target: Target, *, TOUCHINGOBJECTMENU: T) -> T:
    return TOUCHINGOBJECTMENU

@Block.register_evaluator("sensing_keyoptions")
def op_sensing_keyoptions(_target: Target, *, KEY_OPTION: T) -> T:
    return KEY_OPTION

@Block.register_evaluator("sensing_answer")
def op_sensing_answer(target: Target) -> str:
    return target.project.sensing_answer

@Block.register_evaluator("sensing_timer")
def op_sensing_timer(target: Target) -> float:
    return round(time.perf_counter() - target.project.timer_start_time, 3)

@Block.register_evaluator("sensing_of")
def sop_ensing_of(target: Target, OBJECT: str, *, PROPERTY: str) -> ScratchValue:
    if len(targets := [i for i in target.project.targets if i.name == OBJECT and isinstance(i, Sprite) and not i.is_clone]) == 0:
        print(f"sensing_of: no object with the name of {OBJECT}")
        return 0.0
    other = targets[0]
    match PROPERTY:
        case "x position":
            return other.xpos
        case "y position":
            return other.ypos
    print(f"sensing of: unknown property {PROPERTY!r}")
    return 0.0

@Block.register_evaluator("sensing_askandwait")
def op_sensing_askandwait(target: Target, QUESTION: str) -> Generator[Any, Any, None]:
    # TODO: Add text to question box when target is hidden or not a sprite
    if isinstance(target, Sprite):
        target.bubble = QUESTION
    target.project.question = None
    target.project.show_question = True
    while target.project.show_question:
        yield []
    if isinstance(target, Sprite):
        target.bubble = None

@Block.register_evaluator("sensing_of_object_menu")
def op_sensing_of_object_menu(_target: Target, *, OBJECT: str) -> str:
    return OBJECT

@Block.register_evaluator("event_broadcast")
def op_event_broadcast(_target: Target, BROADCAST_INPUT: str) -> None:
    pygame.event.post(pygame.event.Event(non_optional(EVENT_BROADCAST), id=BROADCAST_INPUT))

@dataclass(frozen=True)
class BlockEvent:
    pass

@dataclass(frozen=True)
class FlagClickEvent(BlockEvent):
    pass

@dataclass(frozen=True, eq=False)
class CloneCreatedEvent(BlockEvent):
    target: Target

    def __hash__(self) -> int:
        return hash((id(self), id(self.target)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CloneCreatedEvent):
            return False
        return id(self.target) == id(other.target)

@dataclass(frozen=True)
class KeyPressedEvent(BlockEvent):
    key: str

@dataclass(frozen=True)
class BroadcastEvent(BlockEvent):
    id: str

@dataclass(repr=False)
class BlockList:
    target: Target | None
    blocks: list[Block]

    is_in_progress: bool = field(init=False, default=False)

    def __repr__(self) -> str:
        return f"BlockList({self.blocks})"

    def set_target(self, target: Target) -> None:
        self.target = target
        for block in self.blocks:
            block.set_target(target)

    @property
    def launch_event(self) -> BlockEvent | None:
        match self.blocks[0].opcode:
            case opcode if not opcode in HAT_BLOCKS:
                return None
            case "control_start_as_clone":
                return CloneCreatedEvent(non_optional(self.target))
            case "event_whenbroadcastreceived":
                return BroadcastEvent(cast(str, self.blocks[0].fields["BROADCAST_OPTION"][1]))
            case "event_whenflagclicked":
                return FlagClickEvent()
            case "event_whenkeypressed":
                return KeyPressedEvent(self.blocks[0].fields["KEY_OPTION"][0])
        raise NotImplementedError(self.blocks[0].opcode)

    def evaluate(self) -> Generator[Any, Any, Any]:
        if self.is_in_progress:
            return None
        self.is_in_progress = True
        last = None
        try:
            for block in self.blocks:
                last = yield from cast(Generator[Any, Any, Any], block.evaluate())
        finally:
            self.is_in_progress = False
        return last

    @classmethod
    def load_lists(cls, raw_block_data: dict[Any, Any]) -> list[BlockList]:
        # Pass 1: Create all the blocks you can, leaving the block references as they are
        block_id_map = {}
        to_replace: list[tuple[Block, str]] = []
        for block_id, raw_block in raw_block_data.items():
            args: dict[str, ScratchValue] = {}
            local_to_replace = []
            for arg_name, raw_arg in raw_block["inputs"].items():
                # shadow argument
                if raw_arg[0] == 3:
                    raw_arg = raw_arg[1]

                match raw_arg:
                    case (1, (unknown, value)):
                        print("arg parse: 1 encountered", unknown, value)
                        args[arg_name] = value

                    # TODO: Figure out what's the difference between the three
                    case (1, str(shadow)):
                        args[arg_name] = shadow
                        local_to_replace.append(arg_name)
                    case (2, str(shadow)):
                        args[arg_name] = shadow
                        local_to_replace.append(arg_name)
                    case str(shadow):
                        args[arg_name] = shadow
                        local_to_replace.append(arg_name)

                    case (1, (11, str(broadcast_name), str(broadcast_id))):  # ???
                        args[arg_name] = broadcast_id

                    case (12, str(varname), str(_)):
                        args[arg_name] = VariableReference(None, varname)
                    case (unknown_arg_type, *rest):
                        print("arg parse: unknown encountered", unknown_arg_type, rest)
                        args[arg_name] = f"<unknown {unknown_arg_type}:{rest}>"
                    case unknown_arg:
                        print("arg parse: full unknown encountered", unknown_arg)
                        args[arg_name] = f"<unknown {unknown_arg}>"

            block_id_map[block_id] = Block(raw_block["shadow"], None,
                                           raw_block["opcode"], cast(dict[str, Any], args), raw_block["fields"])
            block_id_map[block_id].next_block = raw_block["next"]
            block_id_map[block_id].parent_block = raw_block["parent"]
            to_replace.extend((block_id_map[block_id], i) for i in local_to_replace)

        # Pass 2: Construct all the BlockLists out of the Blocks
        blocklist_id_map = {}
        blocks_to_remove = []
        for block_id, block in block_id_map.items():
            if block.shadow:
                continue

            current = block
            current_id = block_id
            while current.parent_block is not None and block_id_map[current.parent_block].next_block == current_id:
                current_id = current.parent_block
                current = block_id_map[current_id]

            if current.shadow:
                continue

            blocks = [current]
            start_id = current_id
            blocks_to_remove.append(current_id)
            while current.next_block is not None:
                current_id = current.next_block
                current = block_id_map[current_id]
                blocks.append(current)
                blocks_to_remove.append(current_id)
            blocklist_id_map[start_id] = cls(None, blocks)

        # Pass 3: Resolve references to other blocks
        for block, arg_name in to_replace:
            target_block_id = block.arguments[arg_name]
            target_block: Block | BlockList
            try:
                target_block = blocklist_id_map[target_block_id]
            except KeyError:
                try:
                    target_block = block_id_map[target_block_id]
                except KeyError:
                    # pylint: disable=line-too-long
                    raise ValueError("Reference to block that is in the middle of a block list") from None
            block.arguments[arg_name] = target_block
            if isinstance(target_block, Block):
                print(target_block.shadow)        

        final_lists = [i for i in blocklist_id_map.values() if i.blocks and i.blocks[0].opcode in HAT_BLOCKS]
        return final_lists

@dataclass(eq=False)
class Target(abc.ABC):
    project: Project = field(repr=False)

    name: str
    variables: list[Variable]
    lists: None
    broadcasts: list[str]
    blocks: list[BlockList]
    comments: None
    costumes: list[Costume]
    sounds: list[Sound]

    costume: Costume
    volume: float

    is_clone: bool = field(init=False, default=False)

    @property
    @abc.abstractmethod
    def mask(self) -> pygame.mask.Mask:
        ...

    @abc.abstractmethod
    def draw(self) -> None:
        ...

    @staticmethod
    def load(project: Project, sb3: zipfile.PyZipFile, data: dict[Any, Any], sounds_cache: dict[str, Sound],
             costumes_cache: dict[str, Costume], variables_cache: dict[str, Variable]) -> Target:
        variable_id_map = {k: Variable.load(v) for k, v in data["variables"].items()}
        variables_cache.update(variable_id_map)

        blocks = BlockList.load_lists(data["blocks"])

        sounds = [sounds_cache[i["assetId"]] if i["assetId"] in sounds_cache else Sound.load(sb3, i)
                  for i in data["sounds"]]
        sounds_cache.update({i.name: i for i in sounds})

        costumes = [costumes_cache[i["assetId"]]
                    if i["assetId"] in costumes_cache else
                    Costume.load(sb3, i)
                    for i in data["costumes"]]
        costumes_cache.update({i.name: i for i in costumes})

        broadcasts = list(data["broadcasts"].keys())

        target: Target
        if data["isStage"]:
            target = Stage(project,
                           data["name"], list(variable_id_map.values()),
                           None, broadcasts, blocks, None, costumes, sounds,
                           costumes[data["currentCostume"]], data["volume"])
        else:
            target = Sprite(project,
                            data["name"], list(variable_id_map.values()),
                            None, broadcasts, blocks, None, costumes, sounds,
                            costumes[data["currentCostume"]], data["volume"],
                            data["visible"], data["x"], data["y"], data["size"],
                            data["direction"], data["draggable"],
                            {"all around": RotationStyle.ALL_AROUND}[data["rotationStyle"]])
        for block_list in blocks:
            block_list.set_target(target)
        return target

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)

@dataclass(eq=False)
class Stage(Target):

    @property
    def mask(self) -> pygame.mask.Mask:
        # TODO: Don't hard-code screen size
        return pygame.mask.from_surface(pygame.Surface((480, 360)))

    def draw(self) -> None:
        self.costume.draw((0, 0))

class RotationStyle(enum.IntEnum):
    ALL_AROUND = enum.auto()

@dataclass(eq=False)
class Sprite(Target):
    visible: bool
    xpos: float
    ypos: float
    scale: float
    angle: float
    draggable: bool
    rotation_style: ScratchValue | int

    bubble: str | None = None

    @property
    def width(self) -> int:
        return int(self.costume.width * self.scale / 100)

    @property
    def height(self) -> int:
        return int(self.costume.height * self.scale / 100)

    @property
    def mask(self) -> pygame.mask.Mask:
        return pygame.mask.from_surface(
            self.costume.render(angle=self.angle, scale=self.scale / 100)
        )

    def draw(self) -> None:
        if not self.visible or self.xpos != self.xpos or abs(self.xpos) == float("inf") or self.ypos != self.ypos or abs(self.ypos) == float("inf"):
            return

        pos = int(self.xpos), int(self.ypos)
        self.costume.draw(pos, angle=self.angle, scale=self.scale / 100)  # self.angle
        if not self.bubble:
            return

        box = IM.rotated_rectangle_extents(
            self.width * self.scale / 100, self.height * self.scale / 100, self.angle
        )
        box.topleft = int(self.xpos - box.width / 2), int(self.ypos + box.height / 2)
        IM.draw_thought_bubble(self.bubble, (self.xpos, box[1]))

@dataclass
class Project:
    targets: list[Target]
    sensing_answer: str
    timer_start_time: float

    question: str | None = ""
    show_question: bool = False

    @property
    def stage(self) -> Stage:
        return [i for i in self.targets if isinstance(i, Stage)][0]

    @classmethod
    def load(cls, sb3: zipfile.PyZipFile) -> Project:
        with sb3.open("project.json") as file:
            project = json.load(file)

        res_project = cls([], "", time.perf_counter())
        sounds: dict[str, Sound] = {}
        costumes: dict[str, Costume] = {}
        variables: dict[str, Variable] = {}
        for target_raw in project["targets"]:
            res_project.targets.append(
                Target.load(res_project, sb3, target_raw, sounds, costumes, variables)
            )

        return res_project





def main() -> None:
    global EVENT_CLONE_SCRIPT_INSTANCES, EVENT_BROADCAST

    pygame.init()
    pygame.mixer.set_num_channels(20)
    pygame.key.set_repeat(500, 50)

    IM.set_context(480, 360, 30)

    EVENT_CLONE_SCRIPT_INSTANCES = pygame.event.custom_type()
    EVENT_BROADCAST = pygame.event.custom_type()

    with zipfile.PyZipFile("Project(5).sb3") as sb3:  # test.sb3
        project = Project.load(sb3)

    scripts: dict[type[BlockEvent], list[tuple[BlockEvent, BlockList]]] = {
        FlagClickEvent: [], KeyPressedEvent: [], BroadcastEvent: [], CloneCreatedEvent: []
    }
    for target in project.targets:
        for block_list in target.blocks:
            event = non_optional(block_list.launch_event)
            scripts[type(event)].append((event, block_list))
    script_queue: list[BlockEvent | Generator[Any, Any, Any]] = [FlagClickEvent()]

    running = True
    qbox_state = None
    try:
        while running:
            with IM.frame():
                key_events = []
                for pevent in pygame.event.get():
                    if pevent.type == pygame.QUIT:
                        running = False
                    
                    elif pevent.type == pygame.TEXTINPUT:
                        key_events.append(pevent)
                    
                    elif pevent.type == pygame.KEYDOWN:
                        key_events.append(pevent)
                        if pevent.key in PYGAME_KEY_TO_SCRATCH:
                            script_queue.append(KeyPressedEvent(PYGAME_KEY_TO_SCRATCH[pevent.key]))

                    elif pevent.type == EVENT_BROADCAST:
                        script_queue.append(BroadcastEvent(pevent.id))

                    elif pevent.type == EVENT_CLONE_SCRIPT_INSTANCES:
                        print("request to clone the running scripts of", pevent.target.name, pevent.target.is_clone)
                        for block_list in pevent.target.blocks:
                            event = non_optional(block_list.launch_event)
                            scripts[type(event)].append((event, block_list))
                        script_queue.append(CloneCreatedEvent(pevent.target))

                next_frame_queue: list[BlockEvent | Generator[Any, Any, Any]] = []
                for script_or_event in script_queue:
                    if isinstance(script_or_event, BlockEvent):
                        results = []
                        for event, script in scripts[type(script_or_event)]:
                            if script_or_event != event:
                                continue
                            try:
                                results.append(script.evaluate())
                            except StopThisScript:
                                pass
                        script_queue.extend(results)
                        continue

                    try:
                        script_queue.extend(next(script_or_event))
                        next_frame_queue.append(script_or_event)
                    except StopIteration:
                        pass
                script_queue = next_frame_queue

                for target in project.targets:
                    target.draw()

                if project.show_question:
                    qbox_state = IM.draw_question_box(qbox_state, key_events)
                    if qbox_state.closed:
                        project.sensing_answer = qbox_state.text
                        project.show_question = False
                        qbox_state = None

    except StopAll:
        pass

    pygame.quit()

main()
