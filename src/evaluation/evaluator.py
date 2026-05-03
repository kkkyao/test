from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.equation_matcher import EquationMatcher


class EpisodeEvaluator:
    """
    Evaluate one episode result with exploration metrics and equation matching.

    Input
    -----
    result : dict returned by EpisodeRunner.run_episode(), which must contain:
        trajectory       : List[dict]
        finish_reached   : bool
        final_equation   : str | None
        finish_step_id   : int | None
        parse_error      : str | None

    Output (evaluate())
    -------------------
    Flat, JSON-friendly dict grouped into three sections:

    ── Outcome ──────────────────────────────────────────────────────────────
    success                    bool   finish AND equation correct
    finish_called              bool   model emitted finish
    valid_finish               bool   finish AND equation was submitted
    equation_correct           bool   submitted equation matches ground truth
    equation_submitted         bool   non-empty final_equation was provided
    steps_to_success           int|None  total steps when succeeded, else None
    termination_reason         str    finish_success | finish_wrong |
                                      max_steps | parse_error
    final_equation             str|None
    error_message              str|None

    ── Exploration ───────────────────────────────────────────────────────────
    total_steps                int
    unique_states              int    distinct state_after values
    unique_actions             int    distinct (type, variable, value) tuples
    repeated_actions           int    total_actions - unique_actions
    loop_count                 int    times a previously-seen action reappears

    ── Scores ────────────────────────────────────────────────────────────────
    state_coverage_ratio       float  unique_states / total_steps
    redundancy_penalty         float  repeated_actions / total_steps
    variable_isolation_score   float  action steps where the action variable
                                      actually changed / total action steps.
                                      In the current env this is always 1.0
                                      for well-formed episodes; useful as a
                                      sanity check and for future noisy envs.
    controlled_experiments_ratio float  alias for variable_isolation_score
    exploration_breadth_score  float  alias for state_coverage_ratio
    discovery_efficiency       float  1 / total_steps if success else 0.0
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

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute all episode-level evaluation metrics."""
        if not isinstance(result, dict):
            raise ValueError("result must be a dictionary")
        if "trajectory" not in result:
            raise ValueError("result must contain 'trajectory'")

        trajectory = result["trajectory"]
        if not isinstance(trajectory, list):
            raise ValueError("'trajectory' must be a list")

        # ── Raw facts from runner ─────────────────────────────────────────────
        total_steps    = len(trajectory)
        finish_reached = bool(result.get("finish_reached", False))
        final_equation = result.get("final_equation")
        finish_step_id = result.get("finish_step_id")
        parse_error    = result.get("parse_error")

        # ── Outcome fields ────────────────────────────────────────────────────
        equation_submitted = isinstance(final_equation, str) and bool(final_equation.strip())

        if finish_reached and equation_submitted:
            equation_correct = self.matcher.match(
                predicted=final_equation,
                ground_truth=self.ground_truth_equation,
            )
        else:
            equation_correct = False

        success      = finish_reached and equation_correct
        valid_finish = finish_reached and equation_submitted

        steps_to_success = (
            (finish_step_id + 1) if (success and finish_step_id is not None) else None
        )

        termination_reason = self._termination_reason(
            parse_error, finish_reached, equation_correct
        )

        # ── Exploration metrics ───────────────────────────────────────────────
        unique_states = self._count_unique_states(trajectory)

        action_metrics           = self._compute_action_metrics(trajectory)
        unique_actions           = action_metrics["unique_actions"]
        repeated_actions         = action_metrics["repeated_actions"]
        loop_count               = action_metrics["loop_count"]
        variable_isolation_score = action_metrics["variable_isolation_score"]

        # ── Scores ────────────────────────────────────────────────────────────
        if total_steps > 0:
            state_coverage_ratio = unique_states / total_steps
            redundancy_penalty   = repeated_actions / total_steps
        else:
            state_coverage_ratio = 0.0
            redundancy_penalty   = 0.0

        discovery_efficiency = (1.0 / total_steps) if (success and total_steps > 0) else 0.0

        return {
            # outcome
            "success":                      success,
            "finish_called":                finish_reached,
            "valid_finish":                 valid_finish,
            "equation_correct":             equation_correct,
            "equation_submitted":           equation_submitted,
            "steps_to_success":             steps_to_success,
            "termination_reason":           termination_reason,
            "final_equation":               final_equation,
            "error_message":                parse_error,
            # exploration counts
            "total_steps":                  total_steps,
            "unique_states":                unique_states,
            "unique_actions":               unique_actions,
            "repeated_actions":             repeated_actions,
            "loop_count":                   loop_count,
            # scores
            "state_coverage_ratio":         round(state_coverage_ratio, 4),
            "redundancy_penalty":           round(redundancy_penalty, 4),
            "variable_isolation_score":     round(variable_isolation_score, 4),
            "controlled_experiments_ratio": round(variable_isolation_score, 4),
            "exploration_breadth_score":    round(state_coverage_ratio, 4),
            "discovery_efficiency":         round(discovery_efficiency, 6),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _termination_reason(
        parse_error: Optional[str],
        finish_reached: bool,
        equation_correct: bool,
    ) -> str:
        # finish_reached always takes priority: a parse_error that occurred
        # mid-episode but was recovered by the forced-finish step should not
        # shadow the final outcome.  parse_error is kept in error_message for
        # diagnostics but does not affect termination_reason when an equation
        # was ultimately submitted.
        if finish_reached and equation_correct:
            return "finish_success"
        if finish_reached and not equation_correct:
            return "finish_wrong"
        if parse_error:
            return "parse_error"
        return "max_steps"

    @staticmethod
    def _count_unique_states(trajectory: List[Dict[str, Any]]) -> int:
        """Count distinct state_after values across the trajectory."""
        seen: set = set()
        for step in trajectory:
            if not isinstance(step, dict):
                raise ValueError("each trajectory item must be a dictionary")
            state_after = step.get("state_after")
            if state_after is None:
                continue
            seen.add(json.dumps(state_after, sort_keys=True))
        return len(seen)

    @staticmethod
    def _action_key(parsed_action: Dict[str, Any]) -> Tuple:
        """Canonical hashable key for an action dict."""
        return (
            parsed_action.get("action_type"),
            parsed_action.get("variable"),
            parsed_action.get("value"),
        )

    @staticmethod
    def _compute_action_metrics(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute all action-level metrics from the trajectory.

        unique_actions
            Number of distinct (action_type, variable, value) tuples.

        repeated_actions
            total_action_steps - unique_actions.

        loop_count
            Number of times a previously-seen action reappears.

        variable_isolation_score
            Fraction of action steps where the action's target variable
            actually changed in state. Checking only the action variable
            (not all state vars) avoids counting equation output cascades
            as additional changes. In the current env this is always 1.0
            for well-formed episodes; useful as a sanity check and for
            future noisy or multi-action environments.
        """
        seen_actions:    set         = set()
        all_action_keys: List[Tuple] = []
        controlled_count   = 0
        total_action_steps = 0

        for step in trajectory:
            if not isinstance(step, dict):
                raise ValueError("each trajectory item must be a dictionary")

            if step.get("step_type") != "action":
                continue

            total_action_steps += 1

            parsed_action = step.get("parsed_action")
            if isinstance(parsed_action, dict):
                key = EpisodeEvaluator._action_key(parsed_action)
                all_action_keys.append(key)
                seen_actions.add(key)

                # Variable isolation: did the action variable actually change?
                action_var   = parsed_action.get("variable")
                state_before = step.get("state_before") or {}
                state_after  = step.get("state_after")  or {}
                if action_var and state_before.get(action_var) != state_after.get(action_var):
                    controlled_count += 1

        unique_actions   = len(seen_actions)
        repeated_actions = total_action_steps - unique_actions

        loop_count    = 0
        seen_so_far: set = set()
        for key in all_action_keys:
            if key in seen_so_far:
                loop_count += 1
            else:
                seen_so_far.add(key)

        variable_isolation_score = (
            controlled_count / total_action_steps if total_action_steps > 0 else 0.0
        )

        return {
            "unique_actions":           unique_actions,
            "repeated_actions":         repeated_actions,
            "loop_count":               loop_count,
            "variable_isolation_score": variable_isolation_score,
        }