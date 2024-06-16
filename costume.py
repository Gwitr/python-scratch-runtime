from __future__ import annotations

import io
import abc
import base64
import zipfile
from dataclasses import dataclass
from collections.abc import Sequence
from typing import cast, Any, IO, ClassVar, TypeVar

import pyvips

import cIM as IM

T = TypeVar("T")

def non_optional(x: T | None) -> T:
    assert x is not None
    return x

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
    def draw(self, position: Sequence[int], *, angle: float=90.0, scale: float=1.0, pixel: float=0.0,
             mosaic: float=0.0, ghost: float=0.0) -> None:
        pass

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

SCALE = 8  # How many times larger are SVGs rendered for scaling purposes

WHITE_1x1 = base64.b64decode(b"Qk2OAAAAAAAAAIoAAAB8AAAAAQAAAAEAAAABABgAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAD/AAD/AAD/AAAAAAAA/0JHUnOPwvUoUbgeFR6F6wEzMzMTZmZmJmZmZgaZmZkJPQrXAyhcjzIAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAA////AA==")

@dataclass
class SvgCostume(Costume):
    FMT = "svg"

    img: IM.Texture

    @property
    def width(self) -> int:
        return self.img.w

    @property
    def height(self) -> int:
        return self.img.h

    def draw(self, position: Sequence[int], *, angle: float=90.0, scale: float=1.0, pixel: float=0.0, mosaic: float=0.0, ghost: float=0.0) -> IM.Texture:
        if pixel or mosaic or ghost:
            raise NotImplementedError("pixel/mosaic/ghost effects")

        # TODO: Make use of the rotation origin
        IM.draw_texture(self.img, position, scale / 8, angle)

    @classmethod
    def _load(cls, file: IO[bytes] | None, name: str, md5: str, origin: Sequence[float]) -> SvgCostume:
        if file is None:
            return cls(name, None, cast(tuple[float, float], origin), IM.load_texture(io.BytesIO(WHITE_1x1)))

        img = pyvips.Image.new_from_buffer(file.read(), "", dpi=int(72 * SCALE))
        img = IM.load_texture(io.BytesIO(img.write_to_buffer(".png")))
        return cls(name, md5, cast(tuple[float, float], origin), img)
