from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class EpisodeLogger:
    """
    Save episode outputs into stable JSON files.

    Supported outputs:
    - steps.json
    - trajectory.json
    - interaction_log.json
    - summary.json
    """

    def __init__(
        self,
        output_dir: str,
        save_steps: bool = True,
        save_trajectory: bool = True,
        save_interaction_log: bool = True,
        indent: int = 2,
    ) -> None:
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")

        if not isinstance(indent, int) or indent < 0:
            raise ValueError("indent must be a non-negative integer")

        self.output_dir = Path(output_dir)
        self.save_steps = save_steps
        self.save_trajectory = save_trajectory
        self.save_interaction_log = save_interaction_log
        self.indent = indent

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_episode(
        self,
        result: Dict[str, Any],
        evaluation: Optional[Dict[str, Any]] = None,  # ← new
    ) -> Dict[str, str]:
        """
        Save one episode result to disk.

        Parameters
        ----------
        result:
            Dict returned by EpisodeRunner.run_episode().
        evaluation:
            Optional dict returned by EpisodeEvaluator.evaluate().
            When provided, its fields are merged into summary.json.

        Returns
        -------
        A dictionary mapping artifact names to saved file paths.
        """
        if not isinstance(result, dict):
            raise ValueError("result must be a dictionary")

        if "steps" not in result:
            raise ValueError("result must contain 'steps'")

        if "trajectory" not in result:
            raise ValueError("result must contain 'trajectory'")

        steps = result["steps"]
        trajectory = result["trajectory"]

        if not isinstance(steps, list):
            raise ValueError("'steps' must be a list")

        if not isinstance(trajectory, list):
            raise ValueError("'trajectory' must be a list")

        saved_paths: Dict[str, str] = {}

        if self.save_steps:
            steps_path = self.output_dir / "steps.json"
            self._save_json(steps_path, steps)
            saved_paths["steps"] = str(steps_path)

        if self.save_trajectory:
            trajectory_path = self.output_dir / "trajectory.json"
            self._save_json(trajectory_path, trajectory)
            saved_paths["trajectory"] = str(trajectory_path)

        if self.save_interaction_log:
            interaction_log = self._build_interaction_log(trajectory)
            interaction_path = self.output_dir / "interaction_log.json"
            self._save_json(interaction_path, interaction_log)
            saved_paths["interaction_log"] = str(interaction_path)

        # summary = runner outcome fields + evaluation metrics (if available)
        summary: Dict[str, Any] = {
            "final_equation": result.get("final_equation"),
            "finish_reached": result.get("finish_reached"),
            "finish_step_id": result.get("finish_step_id"),
            "num_steps":      result.get("num_steps"),
            "parse_error":    result.get("parse_error"),
        }
        if evaluation is not None:
            summary["evaluation"] = evaluation   # ← nested under its own key

        summary_path = self.output_dir / "summary.json"
        self._save_json(summary_path, summary)
        saved_paths["summary"] = str(summary_path)

        return saved_paths

    def _save_json(self, path: Path, data: Any) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=self.indent, ensure_ascii=False, sort_keys=False)

    def _build_interaction_log(
        self,
        trajectory: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        interaction_log: List[Dict[str, Any]] = []

        for step in trajectory:
            if not isinstance(step, dict):
                raise ValueError("each trajectory item must be a dictionary")

            interaction_log.append(
                {
                    "step_id":            step.get("step_id"),
                    "step_type":          step.get("step_type"),
                    "prompt":             step.get("prompt"),
                    "raw_model_output":   step.get("raw_model_output"),
                    "reasoning":          step.get("reasoning"),
                    "hypothesis_text":    step.get("hypothesis_text"),
                    "parsed_action":      step.get("parsed_action"),
                    "final_equation":     step.get("final_equation"),
                    "observation_before": step.get("observation_before"),
                    "observation_after":  step.get("observation_after"),
                    "state_before":       step.get("state_before"),
                    "state_after":        step.get("state_after"),
                    "done":               step.get("done"),
                }
            )

        return interaction_log