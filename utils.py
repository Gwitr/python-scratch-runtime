from typing import Any
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

class ExtraEvents:
    EVENT_CLONE_SCRIPT_INSTANCES = 0
    EVENT_BROADCAST = 1
    queue: list[dict] = []
