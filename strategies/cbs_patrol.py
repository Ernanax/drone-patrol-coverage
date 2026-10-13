from __future__ import annotations

from environment import GridEnvironment, Cell
from algorithms import cbs
from .joint_patrol import JointPatrol


class CBSPatrol(JointPatrol):
    name = "Conflict-Based Search"

    def _solve(self, starts: list[Cell], goals: list[Cell]) -> list[list[Cell]] | None:
        return cbs(self.env, starts, goals, heuristic=self.heuristic)
