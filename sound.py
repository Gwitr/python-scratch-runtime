from __future__ import annotations

import zipfile
from typing import TypeVar, Any
from dataclasses import dataclass, field

T = TypeVar("T")

def non_optional(x: T | None) -> T:
    assert x is not None
    return x

@dataclass
class Sound:
    name: str
    md5: str
    # clip: pygame.mixer.Sound
    clip: Any

    volume: float = field(init=False)
    pitch: float = field(init=False)
    pan: float = field(init=False)
    # channel: pygame.mixer.Channel | None = field(init=False)
    channel: Any = field(init=False)

    def __post_init__(self) -> None:
        self.volume = 1.0
        self.pitch = 0.0
        self.pan = 0.0
        self.channel = None

    def _check_playing(self) -> None:
        # if self.channel is not None and not self.channel.get_busy():
        print("TODO: get_busy")
        if self.channel is not None:
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
        print(f"TODO: set_volume({left_volume}, {right_volume})")
        # non_optional(self.channel).set_volume(left_volume, right_volume)

    def play(self) -> None:
        self._check_playing()

        if self.channel:
            # peculiar Scratch behavior; the same sound can't be played multiple times simultanously
            self.channel.stop()
        # self.channel = pygame.mixer.find_channel(force=True)
        print("TODO: find_channel(force=True)")
        self.update_channel_settings()
        print(f"TODO: play({self.clip})")
        # self.channel.play(self.clip)

    @classmethod
    def load(cls, sb3: zipfile.PyZipFile, data: dict[Any, Any]) -> Sound:
        with sb3.open(data["md5ext"]) as file:
            # return cls(data["name"], data["assetId"], pygame.mixer.Sound(file))
            print(f"TODO: Sound({file})")
            return cls(data["name"], data["assetId"], None)
