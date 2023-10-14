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
from collections.abc import Callable
from typing import cast, Generator, TypeVar
from dataclasses import dataclass
from xml.etree.ElementTree import ElementTree

import pygame
import typing_inspect
from defusedxml.ElementTree import parse
from cairosvg import svg2png
import cairosvg.helpers

import immediate_gfx as IM

T = TypeVar('T')
ScratchValue = float | str

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

@dataclass
class Variable:
    name: str
    value: ScratchValue

    @classmethod
    def load(cls, data):
        return cls(*data)

@dataclass
class Costume(abc.ABC):
    _SUBCLASSES = None
    FMT = None

    name: str
    md5: str
    origin: tuple[float, float]

    @property
    @abc.abstractmethod
    def width(self):
        ...

    @property
    @abc.abstractmethod
    def height(self):
        ...

    @abc.abstractmethod
    def render(self, *, angle=90.0, scale=1.0, pixel=0.0, mosaic=0.0, ghost=0.0):
        ...

    def draw(self, position, *, angle=90.0, scale=1.0, pixel=0.0, mosaic=0.0, ghost=0.0):
        surf = self.render(angle=angle, scale=scale, pixel=pixel, mosaic=mosaic, ghost=ghost)
        IM.draw_texture(surf, position, 1.0, 0.0)

    @classmethod
    @abc.abstractmethod
    def _load(cls, file, name, md5, origin):
        ...

    @classmethod
    def _get_subclasses(cls):
        subclasses = cls.__subclasses__()
        all_subclasses = set()
        for i in subclasses:
            all_subclasses.add(i)
            all_subclasses.update(i._get_subclasses())  # pylint: disable=protected-access
        return all_subclasses

    @classmethod
    def load(cls, sb3, data):
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
    def width(self):
        return self.scaled_surfaces[self.NEUTRAL_SCALE].get_width()

    @property
    def height(self):
        return self.scaled_surfaces[self.NEUTRAL_SCALE].get_height()

    def render(self, *, angle=90.0, scale=1.0, pixel=0.0, mosaic=0.0, ghost=0.0):
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
    def _load(cls, file, name, md5, origin):
        # CairoSVG has trouble figuring out the width/height sometimes, so we have to help it
        if file is None:
            return cls(name, None, origin, [
                pygame.Surface((1, 1), pygame.SRCALPHA).convert_alpha() for _ in cls.SVG_SCALES
            ])
        svg_attrib = cast(ElementTree, parse(file)).getroot().attrib

        dims_regex = re.compile(
            r"\s*((?:0|[1-9]\d*)(?:\.\d*)?)\s*(?:(ex|px|pt|pc|cm|mm|in)\s*)?", re.IGNORECASE
        )
        try:
            dims_raw = [svg_attrib["width"], svg_attrib["height"]]
        except KeyError:
            width = height = None
        else:
            for idx, dim_raw in enumerate(dims_raw):
                if (match := dims_regex.match(dim_raw)) is None:
                    raise ValueError(
                        f"Invalid SVG: cannot understand what a length of '{dim_raw}' is")
                length, unit = match.groups()
                length = float(length)
                if unit not in {"px", None}:
                    length *= cairosvg.helpers.UNITS[unit] * DPI
                dims_raw[idx] = length
            width, height = dims_raw

        file.seek(0)
        scaled_surfaces = []
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

            scaled_surfaces.append(pygame.image.load(io.BytesIO(pngdata), "png").convert_alpha())
        return cls(name, md5, origin, scaled_surfaces)

@dataclass
class Sound:
    name: str
    md5: str
    clip: pygame.mixer.Sound

    def __post_init__(self):
        self.volume = 1.0
        self.pitch = 0.0
        self.pan = 0.0
        self.channel = None

    def _check_playing(self):
        if self.channel is not None and not self.channel.get_busy():
            self.channel = None

    def update_effects(self, *, pitch=0.0, pan=0.0, volume=0.0):
        self._check_playing()

        self.volume = volume
        self.pitch = pitch
        self.pan = pan
        if self.channel is not None:
            self.update_channel_settings()

    def update_channel_settings(self):
        left_volume = self.volume * min(2 - (self.pan + 1), 1)
        right_volume = self.volume * min(self.pan + 1, 1)
        self.channel.set_volume(left_volume, right_volume)

    def play(self):
        self._check_playing()

        if self.channel:
            # peculiar Scratch behavior; the same sound can't be played multiple times simultanously
            self.channel.stop()
        self.channel = pygame.mixer.find_channel(force=True)
        self.update_channel_settings()
        self.channel.play(self.clip)

    @classmethod
    def load(cls, sb3, data):
        with sb3.open(data["md5ext"]) as file:
            return cls(data["name"], data["assetId"], pygame.mixer.Sound(file))

@dataclass(repr=False)
class VariableReference:
    target: Target
    name: str

    def set_target(self, target):
        self.target = target

    def evaluate(self, lvalue=False):
        try:
            var = [i for i in self.target.variables if i.name == self.name][0]
        except IndexError:
            var = [i for i in self.target.project.stage.variables if i.name == self.name][0]
        if lvalue:
            return var
        return var.value
        yield  # All evaluate functions are generators for now

    def __repr__(self):
        return self.name

class Block:
    shadow: bool

    target: Target
    opcode: str
    arguments: dict[str, ScratchValue | VariableReference | Block | BlockList]
    fields: dict[str, tuple[str, ScratchValue | None]]

    Unevaluated = Callable[[], Generator[ScratchValue, ScratchValue, T]]

    def __init__(self, shadow, target, opcode, arguments, fields):
        self.shadow = shadow
        self.target = target
        self.opcode = opcode
        self.arguments = arguments
        self.fields = fields
        # Attributes only to help with parsing the JSON data
        self.next_block = None
        self.parent_block = None

    def set_target(self, target):
        self.target = target
        for value in self.arguments.values():
            if isinstance(value, (Block, BlockList, VariableReference)):
                value.set_target(target)

    def evaluate_argument(self, argname, lvalue=False):
        arg = self.arguments[argname]
        if isinstance(arg, (Block, BlockList)):
            return (yield from arg.evaluate())
        if isinstance(arg, VariableReference):
            return (yield from arg.evaluate(lvalue))
        return arg

    EVAL_FUNCTIONS: dict[Callable] = {}
    @classmethod
    def register_evaluator(cls, opcode_name):
        def decorator(f):
            cls.EVAL_FUNCTIONS[opcode_name] = f
            return f
        return decorator

    def evaluate(self):
        # pylint: disable=no-member
        # pylint won't SHUT THE FUCK UP and keeps forgetting
        # that I LITERALLY CASTED THE FUCKING VARIABLE
        try:
            func = self.EVAL_FUNCTIONS[self.opcode]
        except KeyError:
            print(f"Note: executing unknown block {self!r}")
            return
        spec = inspect.getfullargspec(func)
        args = []
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
                case x if typing_inspect.get_origin(x) == Callable and typing_inspect.get_args(x, True)[0] == [] and typing_inspect.is_generic_type(typing_inspect.get_args(x, True)[1]) and typing_inspect.get_args(typing_inspect.get_args(x, True)[1], True)[:2] == (ScratchValue, ScratchValue):
                    args.append(lambda *, x=argname: self.evaluate_argument(x))
                case x:
                    raise ValueError("can't cast ScratchValue to %r" % x.__qualname__)
        
        kwargs = {}
        for kwargname in spec.kwonlyargs:
            match eval(spec.annotations.get(kwargname, "None")):
                case x if x == VariableReference:
                    kwargs[kwargname] = VariableReference(self.target, self.fields[kwargname][0])
                case _:
                    kwargs[kwargname] = self.fields[kwargname][0]
        
        if spec.annotations[spec.args[0]] is not Target:
            if not isinstance(self.target, eval(spec.annotations[spec.args[0]])):
                return

        result = func(self.target, *args, **kwargs)
        if isinstance(result, types.GeneratorType):
            return (yield from result)
        return result

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.opcode}({args})"

@Block.register_evaluator("motion_movesteps")
def op_motion_movesteps(sprite: Sprite, STEPS: float):
    angle = math.radians(90.0 - sprite.angle)
    sprite.xpos += STEPS * math.cos(angle)
    sprite.ypos += STEPS * math.sin(angle)

@Block.register_evaluator("motion_gotoxy")
def op_motion_gotoxy(sprite: Sprite, X: float, Y: float):
    sprite.xpos = X
    sprite.ypos = Y

@Block.register_evaluator("motion_pointindirection")
def op_motion_pointindirection(sprite: Sprite, DIRECTION: float):
    sprite.angle = DIRECTION

@Block.register_evaluator("motion_turnright")
def op_motion_turnright(sprite: Sprite, DEGREES: float):
    sprite.angle += DEGREES

@Block.register_evaluator("motion_turnleft")
def op_motion_turnleft(sprite: Sprite, DEGREES: float):
    sprite.angle -= DEGREES

@Block.register_evaluator("looks_thinkforsecs")
def op_looks_thinkforsecs(sprite: Sprite, MESSAGE: str, SECS: float):
    # TODO: Don't hard-code FPS; also use an event instead of this
    sprite.bubble = MESSAGE
    for _ in range(int(30 * SECS)):
        yield []
    sprite.bubble = None

@Block.register_evaluator("looks_sayforsecs")
def op_looks_thinkforsecs(sprite: Sprite, MESSAGE: str, SECS: float):
    # TODO: Use another type of bubble; also don't hardcode FPS
    sprite.bubble = MESSAGE
    for _ in range(int(30 * SECS)):
        yield []
    sprite.bubble = None

@Block.register_evaluator("control_forever")
def op_control_forever(_target: Target, SUBSTACK: Block.Unevaluated[None]):
    while True:
        yield from SUBSTACK()
        yield []

@Block.register_evaluator("control_if")
def op_control_if(_target: Target, CONDITION: bool, SUBSTACK: Block.Unevaluated[None]):
    if CONDITION:
        yield from SUBSTACK()

@Block.register_evaluator("control_wait")
def op_control_wait(_target: Target, DURATION: float):
    # TODO: Don't hard-code FPS; also use an event instead of this
    for _ in range(int(30 * DURATION)):
        yield []

@Block.register_evaluator("sensing_touchingobject")
def op_sensing_touchingobject(sprite: Sprite, TOUCHINGOBJECTMENU: str):
    if TOUCHINGOBJECTMENU == "_mouse_":
        # TODO: Don't hard-code screen size
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
                int(IM.Context.display.get_height() - mousey - 180 - sprite.ypos + bbox[3] / 2))
        ) is not None
        return result

    print(f"Note: unknown touching object setting {TOUCHINGOBJECTMENU}")
    return False

@Block.register_evaluator("sensing_keypressed")
def op_sensing_keypressed(_target: Target, KEY_OPTION: str):
    key_to_check = {
        "q": pygame.K_q, "w": pygame.K_w, "e": pygame.K_e, "r": pygame.K_r,
        "t": pygame.K_t, "y": pygame.K_y, "u": pygame.K_u, "i": pygame.K_i,
        "o": pygame.K_o, "p": pygame.K_p, "a": pygame.K_a, "s": pygame.K_s,
        "d": pygame.K_d, "f": pygame.K_f, "g": pygame.K_g, "h": pygame.K_h,
        "j": pygame.K_j, "k": pygame.K_k, "l": pygame.K_l, "z": pygame.K_z,
        "x": pygame.K_x, "c": pygame.K_c, "v": pygame.K_v, "b": pygame.K_b,
        "n": pygame.K_n, "m": pygame.K_m, "1": pygame.K_1, "2": pygame.K_2,
        "3": pygame.K_3, "4": pygame.K_4, "5": pygame.K_5, "6": pygame.K_6,
        "7": pygame.K_7, "8": pygame.K_8, "9": pygame.K_9, "0": pygame.K_0,
        "space": pygame.K_SPACE,
    }[KEY_OPTION]
    return pygame.key.get_pressed()[key_to_check]

# Operator blocks
@Block.register_evaluator("operator_equals")
def op_operator_equals(_target: Target, OPERAND1: ScratchValue, OPERAND2: ScratchValue):
    try:
        return float(OPERAND1) == float(OPERAND2)
    except ValueError:
        pass
    return str(OPERAND1) == str(OPERAND2)

@Block.register_evaluator("data_setvariableto")
def op_data_setvariableto(_target: Target, VALUE: ScratchValue,*, VARIABLE: VariableReference):
    (yield from VARIABLE.evaluate(True)).value = VALUE

@Block.register_evaluator("data_changevariableby")
def op_data_changevariableby(_target: Target, VALUE: float, *, VARIABLE: VariableReference):
    var = (yield from VARIABLE.evaluate(True))
    var.value = as_float(var.value) + VALUE

@Block.register_evaluator("sensing_touchingobjectmenu")
def op_sensing_touchingobjectmenu(_target: Target, *, TOUCHINGOBJECTMENU):
    return TOUCHINGOBJECTMENU

@Block.register_evaluator("sensing_keyoptions")
def op_sensing_touchingobjectmenu(_target: Target, *, KEY_OPTION):
    return KEY_OPTION

@Block.register_evaluator("sensing_answer")
def op_sensing_answer(target: Target):
    return target.project.sensing_answer

@Block.register_evaluator("sensing_timer")
def op_sensing_timer(target: Target):
    return round(time.perf_counter() - target.project.timer_start_time, 3)

@Block.register_evaluator("sensing_askandwait")
def op_sensing_askandwait(target: Target, QUESTION: str):
    # TODO: Add text to question box when target is hidden or not a sprite
    if isinstance(target, Sprite):
        cast(Sprite, target).bubble = QUESTION
    target.project.question = None
    target.project.show_question = True
    while target.project.show_question:
        yield []
    if isinstance(target, Sprite):
        cast(Sprite, target).bubble = None

@dataclass(frozen=True)
class BlockEvent:
    pass

@dataclass(frozen=True)
class FlagClickEvent(BlockEvent):
    pass

@dataclass(frozen=True)
class KeyPressedEvent(BlockEvent):
    key: str

@dataclass(repr=False)
class BlockList:
    target: Target
    blocks: list[Block]

    def __repr__(self):
        return f"BlockList({self.blocks})"

    def set_target(self, target):
        self.target = target
        for block in self.blocks:
            block.set_target(target)

    @property
    def launch_event(self):
        match self.blocks[0].opcode:
            case opcode if not opcode.startswith("event_"):
                return None
            case "event_whenflagclicked":
                return FlagClickEvent()
            case "event_whenkeypressed":
                return KeyPressedEvent(self.blocks[0].fields["KEY_OPTION"][0])
        raise NotImplementedError(self.blocks[0].opcode)

    def evaluate(self):
        last = None
        for block in self.blocks:
            last = yield from block.evaluate()
        return last

    @classmethod
    def load_lists(cls, raw_block_data) -> list[BlockList]:
        # Pass 1: Create all the blocks you can, leaving the block references as they are
        block_id_map = {}
        to_replace = []
        for block_id, raw_block in raw_block_data.items():
            args = {}
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

                    case (12, str(varname), str(_)):
                        args[arg_name] = VariableReference(None, varname)
                    case (unknown_arg_type, *rest):
                        print("arg parse: unknown encountered", unknown_arg_type, rest)
                        args[arg_name] = 0.0
                    case unknown_arg:
                        print("arg parse: full unknown encountered", unknown_arg)
                        args[arg_name] = 0.0

            block_id_map[block_id] = Block(raw_block["shadow"], None,
                                           raw_block["opcode"], args, raw_block["fields"])
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

        blocks = [i for i in blocklist_id_map.values()
                  if i.blocks and i.blocks[0].opcode.startswith("event")]
        return blocks

@dataclass
class Target(abc.ABC):
    project: Project

    name: str
    variables: list[Variable]
    lists: None
    broadcasts: None
    blocks: list[BlockList]
    comments: None
    costumes: list[Costume]
    sounds: list[Sound]

    costume: Costume
    volume: float

    @property
    @abc.abstractmethod
    def mask(self):
        ...

    @abc.abstractmethod
    def draw(self):
        ...

    @staticmethod
    def load(project, sb3, data, sounds_cache, costumes_cache, variables_cache):
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

        if data["isStage"]:
            target = Stage(project,
                           data["name"], list(variable_id_map.values()),
                           None, None, blocks, None, costumes, sounds,
                           costumes[data["currentCostume"]], data["volume"])
        else:
            target = Sprite(project,
                            data["name"], list(variable_id_map.values()),
                            None, None, blocks, None, costumes, sounds,
                            costumes[data["currentCostume"]], data["volume"],
                            data["visible"], data["x"], data["y"], data["size"],
                            data["direction"], data["draggable"],
                            {"all around": RotationStyle.ALL_AROUND}[data["rotationStyle"]])
        for block_list in blocks:
            block_list.set_target(target)
        return target

@dataclass
class Stage(Target):

    @property
    def mask(self):
        # TODO: Don't hard-code screen size
        return pygame.mask.from_surface(pygame.Surface((480, 360)))

    def draw(self):
        self.costume.draw((0, 0))

class RotationStyle(enum.Enum):
    ALL_AROUND = enum.auto()

@dataclass
class Sprite(Target):
    visible: bool
    xpos: float
    ypos: float
    scale: float
    angle: float
    draggable: bool
    rotation_style: ScratchValue | int

    bubble: None | pygame.Surface = None

    @property
    def width(self):
        return self.costume.width * self.scale / 100

    @property
    def height(self):
        return self.costume.height * self.scale / 100

    @property
    def mask(self):
        return pygame.mask.from_surface(
            self.costume.render(angle=self.angle, scale=self.scale / 100)
        )

    def draw(self):
        if not self.visible:
            return

        pos = int(self.xpos), int(self.ypos)
        self.costume.draw(pos, angle=self.angle, scale=self.scale / 100)  # self.angle
        if not self.bubble:
            return

        box = IM.rotated_rectangle_extents(
            self.width * self.scale / 100, self.height * self.scale / 100, self.angle
        )
        box.topleft = self.xpos - box.width / 2, self.ypos + box.height / 2
        IM.draw_thought_bubble(self.bubble, (self.xpos, box[1]))

@dataclass
class Project:
    targets: list[Target]
    sensing_answer: str
    timer_start_time: float

    question: str = ""
    show_question: bool = False

    @property
    def stage(self):
        return [i for i in self.targets if isinstance(i, Target)][0]

    @classmethod
    def load(cls, sb3):
        with sb3.open("project.json") as file:
            project = json.load(file)

        res_project = cls([], "", time.perf_counter())
        sounds = {}
        costumes = {}
        variables = {}
        for target_raw in project["targets"]:
            res_project.targets.append(
                Target.load(res_project, sb3, target_raw, sounds, costumes, variables)
            )

        return res_project





def main():
    pygame.init()
    pygame.mixer.set_num_channels(20)
    pygame.key.set_repeat(500, 50)

    IM.set_context(480, 360, 30)

    with zipfile.PyZipFile("test.sb3") as sb3:
        project = Project.load(sb3)

    scripts = {FlagClickEvent: [], KeyPressedEvent: []}
    for target in project.targets:
        for block_list in target.blocks:
            event = block_list.launch_event
            scripts[type(event)].append((event, block_list))
    script_queue = [FlagClickEvent()]

    running = True
    qbox_state = None
    while running:
        with IM.frame():
            key_events = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.TEXTINPUT:
                    key_events.append(event)
                elif event.type == pygame.KEYDOWN:
                    key_events.append(event)

            next_frame_queue = []
            for script_or_event in script_queue:
                if isinstance(script_or_event, BlockEvent):
                    script_queue.extend(
                        script.evaluate() for event, script in scripts[type(script_or_event)]
                        if script_or_event == event
                    )
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

    pygame.quit()

main()
