from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.evaluation.equation_matcher import EquationMatcher


class EpisodeEvaluator:
    """
    Evaluate one episode result with basic exploration metrics
    and final equation matching.
    """

    def __init__(
        self,
        ground_truth_equation: str,
        variable_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        if not isinstance(ground_truth_equation, str) or not ground_truth_equation.strip():
            raise ValueError("ground_truth_equation must be a non-empty string")

        self.ground_truth_equation = ground_truth_equation
        self.matcher = EquationMatcher(variable_mapping=variable_mapping)

    def evaluate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute episode-level evaluation metrics.
        """
        if not isinstance(result, dict):
            raise ValueError("result must be a dictionary")

        if "trajectory" not in result:
            raise ValueError("result must contain 'trajectory'")

        trajectory = result["trajectory"]
        if not isinstance(trajectory, list):
            raise ValueError("'trajectory' must be a list")

        total_steps = len(trajectory)
        finish_reached = bool(result.get("finish_reached", False))
        final_equation = result.get("final_equation")

        unique_states_visited = self._count_unique_states(trajectory)

        if total_steps > 0:
            state_novelty_rate = unique_states_visited / total_steps
            repeated_state_ratio = (total_steps - unique_states_visited) / total_steps
        else:
            state_novelty_rate = 0.0
            repeated_state_ratio = 0.0

        if finish_reached and isinstance(final_equation, str) and final_equation.strip():
            equation_match = self.matcher.match(
                predicted=final_equation,
                ground_truth=self.ground_truth_equation,
            )
        else:
            equation_match = False

        return {
            "total_steps": total_steps,
            "finish_reached": finish_reached,
            "unique_states_visited": unique_states_visited,
            "state_novelty_rate": state_novelty_rate,
            "repeated_state_ratio": repeated_state_ratio,
            "final_equation": final_equation,
            "equation_match": equation_match,
        }

    def _count_unique_states(self, trajectory: list[Dict[str, Any]]) -> int:
        """
        Count unique state_after entries across the trajectory.
        """
        seen = set()

        for step in trajectory:
            if not isinstance(step, dict):
                raise ValueError("each trajectory item must be a dictionary")

            state_after = step.get("state_after")
            if state_after is None:
                continue

            state_key = json.dumps(state_after, sort_keys=True)
            seen.add(state_key)

        return len(seen)