# do NOT do `from __future__ import annotations` here!

import io
import math
import time
import base64
from typing import Any, Generator, cast, TypeVar

import cIM as IM
from target import Target, Sprite
from block import Block, BlockList
from value import VariableReference, Variable, ScratchValue, as_float
from utils import StopAll, StopThisScript, WaitFrameSignal, PushScriptsSignal, BroadcastEvent, RegisterScriptsSignal, CloneCreatedEvent
T = TypeVar("T")

def non_optional(x: T | None) -> T:
    assert x is not None
    return x

WHITE_1x1 = base64.b64decode(b"Qk2OAAAAAAAAAIoAAAB8AAAAAQAAAAEAAAABABgAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAD/AAD/AAD/AAAAAAAA/0JHUnOPwvUoUbgeFR6F6wEzMzMTZmZmJmZmZgaZmZkJPQrXAyhcjzIAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAA////AA==")

@Block.register_evaluator("motion_movesteps")
def op_motion_movesteps(sprite: Sprite, STEPS: float) -> None:
    angle = math.radians(90.0 - sprite.angle)
    sprite.xpos += STEPS * math.cos(angle)
    sprite.ypos += STEPS * math.sin(angle)

@Block.register_evaluator("motion_goto")
def op_motion_goto(sprite: Sprite, TO: str) -> None:
    if len(targets := [i for i in sprite.project.targets if isinstance(i, Sprite) and i.name == TO and not i.is_clone]) == 0:
        print(f"motion_goto: no such target {TO}")
        return None
    sprite.xpos = targets[0].xpos
    sprite.ypos = targets[0].ypos
    return None

@Block.register_evaluator("motion_goto_menu")
def op_motion_goto_menu(_sprite: Sprite, *, TO: str) -> str:
    return TO

@Block.register_evaluator("motion_gotoxy")
def op_motion_gotoxy(sprite: Sprite, X: float, Y: float) -> None:
    sprite.xpos = X
    sprite.ypos = Y

@Block.register_evaluator("motion_changexby")
def op_motion_changexby(sprite: Sprite, DX: float) -> None:
    sprite.xpos += DX

@Block.register_evaluator("motion_changeyby")
def op_motion_changeyby(sprite: Sprite, DY: float) -> None:
    sprite.ypos += DY

@Block.register_evaluator("motion_pointindirection")
def op_motion_pointindirection(sprite: Sprite, DIRECTION: float) -> None:
    sprite.angle = DIRECTION

@Block.register_evaluator("motion_turnright")
def op_motion_turnright(sprite: Sprite, DEGREES: float) -> None:
    sprite.angle += DEGREES

@Block.register_evaluator("motion_turnleft")
def op_motion_turnleft(sprite: Sprite, DEGREES: float) -> None:
    sprite.angle -= DEGREES

@Block.register_evaluator("motion_xposition")
def op_motion_xposition(sprite: Sprite) -> float:
    return sprite.xpos

@Block.register_evaluator("motion_yposition")
def op_motion_yposition(sprite: Sprite) -> float:
    return sprite.ypos

@Block.register_evaluator("looks_show")
def op_looks_show(sprite: Sprite) -> None:
    sprite.visible = True

@Block.register_evaluator("looks_hide")
def op_looks_hide(sprite: Sprite) -> None:
    sprite.visible = False

@Block.register_evaluator("looks_thinkforsecs")
def op_looks_thinkforsecs(sprite: Sprite, MESSAGE: str, SECS: float) -> Generator[Any, Any, None]:
    # TODO: Use an event instead of this
    sprite.bubble = MESSAGE
    for _ in range(int(IM.window_fps * SECS)):
        yield WaitFrameSignal()
    sprite.bubble = None

@Block.register_evaluator("looks_sayforsecs")
def op_looks_sayforsecs(sprite: Sprite, MESSAGE: str, SECS: float) -> Generator[Any, Any, None]:
    # TODO: Use another type of bubble
    sprite.bubble = MESSAGE
    for _ in range(int(IM.window_fps * SECS)):
        yield WaitFrameSignal()
    sprite.bubble = None

@Block.register_evaluator("control_delete_this_clone")
def op_control_delete_this_clone(sprite: Sprite) -> None:
    if sprite.is_clone and sprite in sprite.project.targets:
        sprite.project.targets.remove(sprite)

@Block.register_evaluator("control_stop")
def op_control_stop(_target: Target, *, STOP_OPTION: str) -> None:
    if STOP_OPTION == "all":
        raise StopAll
    if STOP_OPTION == "this script":
        raise StopThisScript
    print(f"op_control_stop: unknown stop option {STOP_OPTION}")

@Block.register_evaluator("control_forever")
def op_control_forever(_target: Target, SUBSTACK: Block.Unevaluated[None]) -> Generator[Any, Any, None]:
    while True:
        yield from SUBSTACK()
        yield WaitFrameSignal()

@Block.register_evaluator("control_if")
def op_control_if(_target: Target, CONDITION: bool, SUBSTACK: Block.Unevaluated[None]) -> Generator[Any, Any, None]:
    if CONDITION:
        yield from SUBSTACK()

@Block.register_evaluator("control_repeat_until")
def op_control_repeat_until(_target: Target, CONDITION: Block.Unevaluated[bool], SUBSTACK: Block.Unevaluated[None]) -> Generator[Any, Any, None]:
    while not bool((yield from CONDITION())):
        yield from SUBSTACK()
        yield WaitFrameSignal()

@Block.register_evaluator("control_wait")
def op_control_wait(_target: Target, DURATION: float) -> Generator[Any, Any, None]:
    # TODO: Use an event instead of this
    for _ in range(int(IM.window_fps * DURATION)):
        yield WaitFrameSignal()

@Block.register_evaluator("control_create_clone_of")
def op_control_create_clone_of(target: Target, CLONE_OPTION: str) -> None:
    if len(targets := [i for i in target.project.targets if isinstance(i, Sprite) and i.name == CLONE_OPTION and not i.is_clone]) == 0:
        return None
    to_clone = targets[0]

    def copy_block(j: Block) -> Block:
        return Block(j.shadow, None, j.opcode, cast(Any, {k: copy_blocklist(v) if isinstance(v, BlockList) else (VariableReference(v.target, v.name) if isinstance(v, VariableReference) else v) for k, v in j.arguments.items()}), j.fields)

    def copy_blocklist(i: BlockList) -> BlockList:
        l = BlockList(None, [copy_block(j) for j in i.blocks])
        l.current_block_idx = i.current_block_idx
        l.is_in_progress = i.is_in_progress
        return l

    blocks = [copy_blocklist(i) for i in to_clone.blocks]
    # breakpoint()
    to_clone.project.targets.append(Sprite(
        to_clone.project, to_clone.name, [Variable(i.name, i.value) for i in to_clone.variables], None,
        to_clone.broadcasts, blocks, to_clone.comments, to_clone.costumes,
        to_clone.sounds, to_clone.costume, to_clone.volume, to_clone.visible,
        to_clone.xpos, to_clone.ypos, to_clone.scale, to_clone.angle,
        to_clone.draggable, to_clone.rotation_style, to_clone.bubble
    ))
    clone = to_clone.project.targets[-1]
    clone.is_clone = True
    for blocklist in clone.blocks:
        blocklist.set_target(clone)
    yield RegisterScriptsSignal(clone)
    yield PushScriptsSignal([CloneCreatedEvent(clone)])
    return None

@Block.register_evaluator("control_create_clone_of_menu")
def op_control_create_clone_of_menu(_target: Target, *, CLONE_OPTION: str) -> str:
    return CLONE_OPTION

@Block.register_evaluator("sensing_touchingobject")
def op_sensing_touchingobject(sprite: Sprite, TOUCHINGOBJECTMENU: str) -> bool:
    if math.isnan(sprite.xpos) or math.isnan(sprite.ypos) or not sprite.visible:
        return False

    if TOUCHINGOBJECTMENU == "_mouse_":
        mousex, mousey = IM.mouse_pos()
        mask = sprite.mask
        result = mask.overlap(
            IM.Mask.from_file(io.BytesIO(WHITE_1x1)),
            (int(mask.w/2 + mousex - IM.window_width / 2 - sprite.xpos), int(mask.h/2-(IM.window_height / 2 - mousey - sprite.ypos)))
        ) is not None
        return result

    if TOUCHINGOBJECTMENU == "_edge_":
        bbox = IM.rotated_rectangle_extents(
            sprite.width * sprite.scale / 100,
            sprite.height * sprite.scale / 100,
            sprite.angle
        )
        return (sprite.xpos + bbox[2] / 2 >= IM.window_width / 2 or sprite.xpos - bbox[2] / 2 <= -IM.window_width / 2 or
                sprite.ypos + bbox[3] / 2 >= IM.window_height / 2 or sprite.ypos - bbox[3] / 2 <= -IM.window_height / 2)

    if len(targets := [i for i in sprite.project.targets if i.name == TOUCHINGOBJECTMENU and isinstance(i, Sprite)]) > 0:
        for target in targets:
            if not target.visible:
                continue
            if math.isnan(target.xpos) or math.isnan(target.ypos):
                continue
            mask1 = sprite.mask
            mask2 = target.mask
            xoffs = target.xpos - sprite.xpos + (mask1.w - mask2.w) / 2
            yoffs = target.ypos - sprite.ypos + (mask1.h - mask2.h) / 2
            if mask1.overlap(mask2, (xoffs, yoffs)) is not None:
                return True
        return False

    print(f"Note: unknown touching object setting {TOUCHINGOBJECTMENU}")
    return False

@Block.register_evaluator("sensing_keypressed")
def op_sensing_keypressed(_target: Target, KEY_OPTION: str) -> bool:
    # print(KEY_OPTION, "=>", pygame.key.get_pressed()[key_to_check])
    return IM.key_state(KEY_OPTION)

# Operator blocks
@Block.register_evaluator("operator_add")
def op_operator_add(_target: Target, NUM1: float, NUM2: float) -> float:
    return NUM1 + NUM2

@Block.register_evaluator("operator_subtract")
def op_operator_subtract(_target: Target, NUM1: float, NUM2: float) -> float:
    return NUM1 - NUM2

@Block.register_evaluator("operator_multiply")
def op_operator_multiply(_target: Target, NUM1: float, NUM2: float) -> float:
    return NUM1 * NUM2

@Block.register_evaluator("operator_divide")
def op_operator_divide(_target: Target, NUM1: float, NUM2: float) -> ScratchValue:
    if NUM2 == 0:
        if NUM1 == 0:
            return "NaN"
        return ["-Infinity", "", "Infinity"][round(NUM2 / abs(NUM2) + 1)]
    return NUM1 / NUM2

@Block.register_evaluator("operator_mathop")
def op_operator_mathop(_target: Target, NUM: float, *, OPERATOR: str) -> float:
    match OPERATOR:
        case "abs":
            return abs(NUM)
    print(f"operator_mathop: unknown operator {OPERATOR!r}")
    return 0.0

@Block.register_evaluator("operator_equals")
def op_operator_equals(_target: Target, OPERAND1: ScratchValue, OPERAND2: ScratchValue) -> bool:
    try:
        return float(OPERAND1) == float(OPERAND2)
    except ValueError:
        pass
    return str(OPERAND1) == str(OPERAND2)

@Block.register_evaluator("data_setvariableto")
def op_data_setvariableto(_target: Target, VALUE: ScratchValue, *, VARIABLE: VariableReference) -> Generator[Any, Any, None]:
    cast(Variable, (yield from VARIABLE.evaluate(True))).value = VALUE

@Block.register_evaluator("data_changevariableby")
def op_data_changevariableby(_target: Target, VALUE: float, *, VARIABLE: VariableReference) -> Generator[Any, Any, None]:
    var = cast(Variable, (yield from VARIABLE.evaluate(True)))
    var.value = as_float(var.value) + VALUE

@Block.register_evaluator("sensing_touchingobjectmenu")
def op_sensing_touchingobjectmenu(_target: Target, *, TOUCHINGOBJECTMENU: T) -> T:
    return TOUCHINGOBJECTMENU

@Block.register_evaluator("sensing_keyoptions")
def op_sensing_keyoptions(_target: Target, *, KEY_OPTION: T) -> T:
    return KEY_OPTION

@Block.register_evaluator("sensing_answer")
def op_sensing_answer(target: Target) -> str:
    return target.project.sensing_answer

@Block.register_evaluator("sensing_timer")
def op_sensing_timer(target: Target) -> float:
    return round(time.perf_counter() - target.project.timer_start_time, 3)

@Block.register_evaluator("sensing_of")
def sop_ensing_of(target: Target, OBJECT: str, *, PROPERTY: str) -> ScratchValue:
    if len(targets := [i for i in target.project.targets if i.name == OBJECT and isinstance(i, Sprite) and not i.is_clone]) == 0:
        print(f"sensing_of: no object with the name of {OBJECT}")
        return 0.0
    other = targets[0]
    match PROPERTY:
        case "x position":
            return other.xpos
        case "y position":
            return other.ypos
    print(f"sensing of: unknown property {PROPERTY!r}")
    return 0.0

@Block.register_evaluator("sensing_askandwait")
def op_sensing_askandwait(target: Target, QUESTION: str) -> Generator[Any, Any, None]:
    # TODO: Add text to question box when target is hidden or not a sprite
    if isinstance(target, Sprite):
        target.bubble = QUESTION
    target.project.question = None
    target.project.show_question = True
    while target.project.show_question:
        yield WaitFrameSignal()
    if isinstance(target, Sprite):
        target.bubble = None

@Block.register_evaluator("sensing_of_object_menu")
def op_sensing_of_object_menu(_target: Target, *, OBJECT: str) -> str:
    return OBJECT

@Block.register_evaluator("event_broadcast")
def op_event_broadcast(_target: Target, BROADCAST_INPUT: str) -> None:
    yield PushScriptsSignal([BroadcastEvent(BROADCAST_INPUT)])
