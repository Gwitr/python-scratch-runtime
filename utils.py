from typing import Any, Generator
from dataclasses import dataclass

@dataclass(frozen=True)
class BlockEvent:
    pass

@dataclass(frozen=True)
class FlagClickEvent(BlockEvent):
    pass

@dataclass(frozen=True, eq=False)
class CloneCreatedEvent(BlockEvent):
    target: Any

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

class Stop(Exception):
    pass

class StopAll(Stop):
    pass

class StopThisScript(Stop):
    pass

@dataclass(frozen=True, slots=True)
class PushScriptsSignal:
    scripts: list[BlockEvent | Generator[Any, Any, Any]]

@dataclass(frozen=True, slots=True)
class WaitFrameSignal:
    pass

@dataclass(frozen=True, slots=True)
class RegisterScriptsSignal:
    target: Any
