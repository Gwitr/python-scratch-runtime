# pylint: disable=missing-class-docstring,missing-function-docstring,missing-module-docstring

from __future__ import annotations

import zipfile
from typing import TypeVar, Any
from collections.abc import Generator

import cIM as IM
import blockdefs as _
from project import Project
from block import BlockList
from utils import StopAll, StopThisScript, ExtraEvents, BlockEvent, FlagClickEvent, KeyPressedEvent, CloneCreatedEvent, BroadcastEvent

import faulthandler
faulthandler.enable()

T = TypeVar("T")

def non_optional(x: T | None) -> T:
    assert x is not None
    return x

def main() -> None:
    IM.init()
    IM.set_context(480, 360, 30)

    with zipfile.PyZipFile("test.sb3") as sb3:  # test.sb3
        project = Project.load(sb3)

    scripts: dict[type[BlockEvent], list[tuple[BlockEvent, BlockList]]] = {
        FlagClickEvent: [], KeyPressedEvent: [], BroadcastEvent: [], CloneCreatedEvent: []
    }
    for target in project.targets:
        for block_list in target.blocks:
            event = non_optional(block_list.launch_event)
            scripts[type(event)].append((event, block_list))
    script_queue: list[BlockEvent | Generator[Any, Any, Any]] = [FlagClickEvent()]

    qbox_state = None
    for info in IM.mainloop():
        for pevent in info.key_events:
            script_queue.append(KeyPressedEvent(pevent.key))

        for pevent in ExtraEvents.queue:
            if pevent["type"] == ExtraEvents.EVENT_BROADCAST:
                script_queue.append(BroadcastEvent(pevent["id"]))

            elif pevent["type"] == ExtraEvents.EVENT_CLONE_SCRIPT_INSTANCES:
                print("request to clone the running scripts of", pevent["target"].name, pevent["target"].is_clone)
                for block_list in pevent["target"].blocks:
                    event = non_optional(block_list.launch_event)
                    scripts[type(event)].append((event, block_list))
                script_queue.append(CloneCreatedEvent(pevent["target"]))
        ExtraEvents.queue.clear()

        next_frame_queue: list[BlockEvent | Generator[Any, Any, Any]] = []
        try:
            for script_or_event in script_queue:
                if isinstance(script_or_event, BlockEvent):
                    results = []
                    for event, script in scripts[type(script_or_event)]:
                        if script_or_event != event:
                            continue
                        try:
                            results.append(script.evaluate())
                        except StopThisScript:
                            pass
                    script_queue.extend(results)
                    continue

                try:
                    script_queue.extend(next(script_or_event))
                    next_frame_queue.append(script_or_event)
                except (StopThisScript, StopIteration):
                    pass
        except StopAll:
            IM.stop()
        script_queue = next_frame_queue

        for target in project.targets:
            target.draw()

        if project.show_question:
            qbox_state = IM.draw_question_box(qbox_state, info)
            if qbox_state.closed:
                project.sensing_answer = qbox_state.text
                project.show_question = False
                qbox_state = None

    IM.end()

if __name__ == "__main__":
    main()
