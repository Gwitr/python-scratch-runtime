from __future__ import annotations

import abc
import enum
import zipfile
from dataclasses import dataclass
from collections.abc import Collection, Sequence, Generator
from typing import TypeVar, Callable, Any, ClassVar, IO

import pygame

T = TypeVar("T")

ScratchValue = float | str

EvaluateGenerator = Generator[list[Any], ScratchValue, ScratchValue | None]

@dataclass
class Variable:
    name: str
    value: ScratchValue

    @classmethod
    def load(cls, data: dict[Any, Any]) -> Variable: ...

@dataclass
class Costume(abc.ABC):
    _SUBCLASSES: ClassVar[list[type[Costume]] | None] = None
    FMT: ClassVar[str | None] = None

    name: str
    md5: str
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

    def draw(self, position: Sequence[float], *, angle: float=90.0, scale: float=1.0,
             pixel: float=0.0, mosaic: float=0.0, ghost: float=0.0) -> None: ...

    @classmethod
    @abc.abstractmethod
    def _load(cls, file: IO[bytes] | None, name: str, md5: str, origin: Sequence[float]) -> Costume: ...

    @classmethod
    def _get_subclasses(cls) -> type[Costume]: ...

    @classmethod
    def load(cls, sb3: zipfile.PyZipFile, data: dict[Any, Any]) -> Costume: ...

DPI: int

@dataclass
class SvgCostume(Costume):
    FMT: ClassVar[str | None] = "svg"

    SVG_SCALES: ClassVar = 0.25, 0.5, 1.0, 2.0, 4.0
    NEUTRAL_SCALE: ClassVar = 2

    scaled_surfaces: list[pygame.Surface]

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    def render(self, *, angle: float=90.0, scale: float=1.0, pixel: float=0.0,
               mosaic: float=0.0, ghost: float=0.0) -> pygame.Surface: ...

    @classmethod
    def _load(cls, file: IO[bytes] | None, name: str, md5: str, origin: Sequence[float]) -> SvgCostume: ...

@dataclass
class Sound:
    name: str
    md5: str
    clip: pygame.mixer.Sound

    def __post_init__(self) -> None: ...

    def _check_playing(self) -> None: ...

    def update_effects(self, *, pitch: float=0.0, pan: float=0.0, volume: float=0.0) -> None: ...

    def update_channel_settings(self) -> None: ...

    def play(self) -> None: ...

    @classmethod
    def load(cls, sb3: zipfile.PyZipFile, data: dict[Any, Any]) -> None: ...

@dataclass
class VariableReference:
    target: Target
    name: str

    def set_target(self, target: Target) -> None: ...

    def evaluate(self, lvalue: bool=False) -> EvaluateGenerator: ...

@dataclass
class Block:
    ArgsType = dict[str, ScratchValue | VariableReference | 'Block' | 'BlockList']
    FieldsType = dict[str, tuple[str, ScratchValue | None]]

    shadow: bool

    target: Target
    opcode: str
    arguments: ArgsType
    fields: FieldsType

    Unevaluated = Callable[[], Generator[ScratchValue, ScratchValue, T]]

    def set_target(self, target: Target) -> None: ...

    def evaluate_argument(self, argname: str, lvalue: bool=False) -> EvaluateGenerator: ...
    EVAL_FUNCTIONS: dict[str, Callable[..., ScratchValue]] = {}

    @classmethod
    def register_evaluator(cls, opcode_name: str) -> Callable[[T], T]: ...

    def evaluate(self) -> EvaluateGenerator: ...

@dataclass(frozen=True)
class BlockEvent:
    pass

@dataclass(frozen=True)
class FlagClickEvent(BlockEvent):
    ...

@dataclass(frozen=True)
class KeyPressedEvent(BlockEvent):
    key: str

@dataclass
class BlockList:
    target: Target
    blocks: list[Block]

    def set_target(self, target: Target) -> None: ...

    @property
    def launch_event(self) -> BlockEvent | None: ...

    def evaluate(self) -> ScratchValue | None: ...

    @classmethod
    def load_lists(cls, raw_block_data: dict[Any, Any]) -> list[BlockList]: ...

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
    def mask(self) -> pygame.mask.Mask: ...

    @abc.abstractmethod
    def draw(self) -> None: ...

    @staticmethod
    def load(project: Project, sb3: zipfile.PyZipFile, data: dict[Any, Any], sounds_cache: dict[str, Sound],
             costumes_cache: dict[str, Costume], variables_cache: dict[str, Variable]) -> Target: ...

@dataclass
class Stage(Target):
    @property
    def mask(self) -> pygame.mask.Mask: ...

    def draw(self) -> None: ...

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

    bubble: pygame.Surface | None = None

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...

    @property
    def mask(self) -> pygame.mask.Mask: ...

    def draw(self) -> None: ...

@dataclass
class Project:
    targets: list[Target]
    sensing_answer: str
    timer_start_time: float

    question: str = ""
    show_question: bool = False

    @property
    def stage(self) -> Stage: ...

    @classmethod
    def load(cls, sb3: zipfile.PyZipFile) -> Project: ...
