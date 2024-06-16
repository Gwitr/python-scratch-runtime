from __future__ import annotations

import json
import time
import zipfile
from dataclasses import dataclass

from sound import Sound
from value import Variable
from costume import Costume
from target import Target, Stage

@dataclass
class Project:
    targets: list[Target]
    sensing_answer: str
    timer_start_time: float

    question: str | None = ""
    show_question: bool = False

    @property
    def stage(self) -> Stage:
        return [i for i in self.targets if isinstance(i, Stage)][0]

    @classmethod
    def load(cls, sb3: zipfile.PyZipFile) -> Project:
        with sb3.open("project.json") as file:
            project = json.load(file)

        res_project = cls([], "", time.perf_counter())
        sounds: dict[str, Sound] = {}
        costumes: dict[str, Costume] = {}
        variables: dict[str, Variable] = {}
        for target_raw in project["targets"]:
            res_project.targets.append(
                Target.load(res_project, sb3, target_raw, sounds, costumes, variables)
            )

        return res_project
