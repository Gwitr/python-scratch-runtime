from __future__ import annotations

import types
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generator, TypeVar, cast, get_origin, get_args

from value import ScratchValue, VariableReference, as_float, as_string, as_bool
from utils import BlockEvent, FlagClickEvent, BroadcastEvent, CloneCreatedEvent, KeyPressedEvent

HAT_BLOCKS = ["event_whenflagclicked", "event_whenbroadcastreceived", "event_whenkeypressed", "control_start_as_clone"]

T = TypeVar("T")

def non_optional(x: T | None) -> T:
    assert x is not None
    return x

@dataclass
class Block:
    shadow: bool

    target: Any | None = field(repr=False)
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

    def clone(self) -> Block:
        return Block(self.shadow, self.target, self.opcode, cast(Any, {k: v.clone() if isinstance(v, (BlockList, VariableReference)) else v for k, v in self.arguments.items()}), self.fields)

    def set_target(self, target: Any) -> None:
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
        # pylint: disable=eval-used
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
            match spec.annotations[argname]:
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
                case x if get_origin(x) == Callable and get_args(x)[0] == [] and get_origin(get_args(x)[1]) is not None and get_args(get_args(x)[1])[:2] == (ScratchValue, ScratchValue):
                    args.append(lambda *, x=argname: self.evaluate_argument(x))
                case x:
                    raise ValueError(f"can't cast ScratchValue to {x!r}")

        kwargs: dict[str, VariableReference | ScratchValue | None] = {}
        for kwargname in spec.kwonlyargs:
            match spec.annotations.get(kwargname, "None"):
                case x if x == VariableReference:
                    kwargs[kwargname] = VariableReference(non_optional(self.target), self.fields[kwargname][0])
                case _:
                    kwargs[kwargname] = self.fields[kwargname][0]

        # if spec.annotations[spec.args[0]] is not Target:
        if not isinstance(self.target, spec.annotations[spec.args[0]]):
            return

        result = func(self.target, *args, **kwargs)
        if isinstance(result, types.GeneratorType):
            return (yield from result)
        return result

    # def __repr__(self) -> str:
        # args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        # return f"{self.opcode}({args})"

@dataclass(repr=False)
class BlockList:
    target: Any | None
    blocks: list[Block]

    is_in_progress: bool = field(init=False, default=False)
    current_block_idx: int = field(init=False, default=-1)
    was_cloned: bool = field(init=False, default=False)

    def clone(self) -> BlockList:
        l = BlockList(None, [block.clone() for block in self.blocks])
        l.current_block_idx = self.current_block_idx
        l.is_in_progress = self.is_in_progress
        l.was_cloned = True
        return l

    def __repr__(self) -> str:
        return f"BlockList({self.blocks})"

    def set_target(self, target: Any) -> None:
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
        if self.is_in_progress and not self.was_cloned:
            return None
        if not self.was_cloned:
            self.current_block_idx = 0
        self.is_in_progress = True
        last = None
        try:
            for block in self.blocks[self.current_block_idx:]:
                last = yield from cast(Generator[Any, Any, Any], block.evaluate())
                self.current_block_idx += 1
        finally:
            self.was_cloned = False
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

                    case (1, (11, str(_broadcast_name), str(broadcast_id))):  # ???
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
