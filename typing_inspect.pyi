# I wish mypy supported higher-order `TypeVar`s :(

from typing import Any

def get_origin(x: type[Any]) -> type[Any] | None: ...

def get_args(x: type[Any], eval_args: bool) -> tuple[Any, ...]: ...

def is_generic_type(x: type[Any]) -> bool: ...   # Here it'd be especially useful - you could make this return a literal DURING type-checking
