from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from collections.abc import Generator

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
    def load(cls, data: list[Any]) -> Variable:
        return cls(*data)

@dataclass(repr=False)
class VariableReference:
    target: Any
    name: str

    def clone(self) -> VariableReference:
        return VariableReference(self.target, self.name)

    def set_target(self, target: Any) -> None:
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
