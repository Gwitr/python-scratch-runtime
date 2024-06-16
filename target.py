from __future__ import annotations

import io
import abc
import enum
import base64
import zipfile
from typing import Any
from dataclasses import dataclass, field

import cIM as IM
from sound import Sound
from block import BlockList
from costume import Costume
from value import Variable, ScratchValue

@dataclass(eq=False)
class Target(abc.ABC):
    project: Any = field(repr=False)

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
    def mask(self) -> IM.Mask:
        ...

    @abc.abstractmethod
    def draw(self) -> None:
        ...

    @staticmethod
    def load(project: Any, sb3: zipfile.PyZipFile, data: dict[Any, Any], sounds_cache: dict[str, Sound],
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
    def mask(self) -> IM.mask.Mask:
        # TODO: Don't hard-code screen size
        png = base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAeAAAAFoAQAAAACnTBWNAAAAIGNIUk0AAHomAACAhAAA+gAAAIDo"
            b"AAB1MAAA6mAAADqYAAAXcJy6UTwAAAACYktHRAAB3YoTpAAAAAd0SU1FB+gGEBEEHmodoFAAAABa"
            b"SURBVHja7csxDQAADAOg+jfditi3wE96EFmWZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmW"
            b"ZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmW5f95NBIQjXdSLDMAAAAldEVYdGRhdGU6Y3Jl"
            b"YXRlADIwMjQtMDYtMTZUMTc6MDQ6MzArMDA6MDB+zhzxAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI0"
            b"LTA2LTE2VDE3OjA0OjMwKzAwOjAwD5OkTQAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNC0wNi0x"
            b"NlQxNzowNDozMCswMDowMFiGhZIAAAAASUVORK5CYII="
        )
        return IM.load_mask(io.BytesIO(png))

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
    def mask(self) -> IM.Mask:
        # TODO: IMPLEMENT!!!!
        # return pygame.mask.from_surface(
            # self.costume.render(angle=self.angle, scale=self.scale / 100)
        # )
        png = base64.b64decode(
            b"iVBORw0KGgoAAAANSUhEUgAAAeAAAAFoAQAAAACnTBWNAAAAIGNIUk0AAHomAACAhAAA+gAAAIDo"
            b"AAB1MAAA6mAAADqYAAAXcJy6UTwAAAACYktHRAAB3YoTpAAAAAd0SU1FB+gGEBEEHmodoFAAAABa"
            b"SURBVHja7csxDQAADAOg+jfditi3wE96EFmWZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmW"
            b"ZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmWZVmW5f95NBIQjXdSLDMAAAAldEVYdGRhdGU6Y3Jl"
            b"YXRlADIwMjQtMDYtMTZUMTc6MDQ6MzArMDA6MDB+zhzxAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI0"
            b"LTA2LTE2VDE3OjA0OjMwKzAwOjAwD5OkTQAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNC0wNi0x"
            b"NlQxNzowNDozMCswMDowMFiGhZIAAAAASUVORK5CYII="
        )
        return IM.load_mask(io.BytesIO(png))

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
        box.x = int(self.xpos - box.width / 2)
        box.y = int(self.ypos + box.height / 2)
        IM.draw_thought_bubble(self.bubble, (self.xpos, box[1]))
