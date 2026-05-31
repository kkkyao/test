from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class EpisodeLogger:
    """
    Save episode outputs into stable JSON / JSONL files.

    Supported outputs:
    - steps.json
    - trajectory.json
    - interaction_log.json  (successful steps + failed parse attempt entries)
    - summary.json
    - parse_errors.jsonl    (one line per failed parse attempt; only written
                             when the episode has at least one parse failure)
    """
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
        evaluation: Optional[Dict[str, Any]] = None,
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
            Not used for student_simulation tasks (pass None).

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

        failed_parse_attempts: List[Dict[str, Any]] = result.get(
            "parse_error_attempts", []
        ) or []
        image_validation_log: List[Dict[str, Any]] = result.get(
            "image_validation_log", []
        ) or []

        if self.save_interaction_log:
            interaction_log = self._build_interaction_log(
                trajectory, failed_parse_attempts, image_validation_log
            )
            interaction_path = self.output_dir / "interaction_log.json"
            self._save_json(interaction_path, interaction_log)
            saved_paths["interaction_log"] = str(interaction_path)

        # parse_errors.jsonl — one line per failed parse attempt (full raw output)
        if failed_parse_attempts:
            pe_path = self.output_dir / "parse_errors.jsonl"
            with pe_path.open("w", encoding="utf-8") as fh:
                for attempt in failed_parse_attempts:
                    fh.write(json.dumps(attempt, ensure_ascii=False) + "\n")
            saved_paths["parse_errors"] = str(pe_path)

        # summary = runner outcome fields + evaluation metrics (if available)
        summary: Dict[str, Any] = {
            "final_equation":              result.get("final_equation"),
            "finish_reason":               result.get("finish_reason"),
            "finish_reached":              result.get("finish_reached"),
            "finish_step_id":              result.get("finish_step_id"),
            "num_steps":                   result.get("num_steps"),
            "parse_error":                 result.get("parse_error"),
            "forced_finish":               result.get("forced_finish"),
            # Forced-finish diagnostics (None when voluntary finish or not triggered)
            "forced_finish_trigger":       result.get("forced_finish_trigger"),
            "forced_finish_used_images":   result.get("forced_finish_used_images", False),
            "forced_finish_error_type":    result.get("forced_finish_error_type"),
            "forced_finish_error_message": result.get("forced_finish_error_message"),
            "image_paths":                 result.get("image_paths", []),
            "image_error_count":           result.get("image_error_count", 0),
            "has_image_error":             result.get("has_image_error", False),
            "max_parse_retries":           result.get("max_parse_retries"),
        }
        if evaluation is not None:
            summary["evaluation"] = evaluation

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
        failed_parse_attempts: Optional[List[Dict[str, Any]]] = None,
        image_validation_log: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build the interaction log from successful trajectory steps.

        Each step entry is enriched with its image validation record
        (image_paths_passed, image_exists, image_error) when
        image_validation_log is provided.

        failed_parse_attempts entries are appended after all successful
        steps, tagged with entry_type="failed_parse_attempt".
        """
        # Build a fast lookup: step_id -> image validation record
        img_by_step: Dict[int, Dict[str, Any]] = {}
        for rec in (image_validation_log or []):
            sid = rec.get("step_id")
            if sid is not None:
                img_by_step[sid] = rec

        interaction_log: List[Dict[str, Any]] = []

        for step in trajectory:
            if not isinstance(step, dict):
                raise ValueError("each trajectory item must be a dictionary")

            step_id = step.get("step_id")
            img_rec = img_by_step.get(step_id, {})

            interaction_log.append(
                {
                    "step_id":              step_id,
                    "step_type":            step.get("step_type"),
                    "prompt":               step.get("prompt"),
                    "raw_model_output":     step.get("raw_model_output"),
                    "reasoning":            step.get("reasoning"),
                    "parsed_action":        step.get("parsed_action"),
                    "final_equation":       step.get("final_equation"),
                    "finish_reason":        step.get("finish_reason"),
                    "observation_before":   step.get("observation_before"),
                    "observation_after":    step.get("observation_after"),
                    "state_before":         step.get("state_before"),
                    "state_after":          step.get("state_after"),
                    "done":                 step.get("done"),
                    # image validation fields (None when not a visual episode)
                    "image_paths_passed":   img_rec.get("image_paths"),
                    "image_exists":         img_rec.get("image_exists"),
                    "image_error":          img_rec.get("image_error"),
                }
            )

        for attempt in (failed_parse_attempts or []):
            step_id = attempt.get("step_id")
            img_rec = img_by_step.get(step_id, {})
            interaction_log.append(
                {
                    "entry_type":           "failed_parse_attempt",
                    "step_id":              step_id,
                    "attempt_index":        attempt.get("attempt_index"),
                    "prompt_hash":          attempt.get("prompt_hash"),
                    "image_paths":          attempt.get("image_paths", []),
                    "raw_model_output":     attempt.get("raw_model_output", ""),
                    "cleaned_model_output": attempt.get("cleaned_model_output", ""),
                    "parse_error_message":  attempt.get("parse_error_message", ""),
                    # image validation for the same step
                    "image_paths_passed":   img_rec.get("image_paths"),
                    "image_exists":         img_rec.get("image_exists"),
                    "image_error":          img_rec.get("image_error"),
                }
            )

        # Dedicated image_validation entries — one per validation record.
        # Written unconditionally so that validation data is always present even
        # when trajectory is empty (e.g. episode broke at step 0).
        for rec in (image_validation_log or []):
            interaction_log.append(
                {
                    "entry_type":     "image_validation",
                    "step_id":        rec.get("step_id"),
                    "requires_images": rec.get("requires_images"),
                    "image_paths":    rec.get("image_paths", []),
                    "image_exists":   rec.get("image_exists", []),
                    "missing_paths":  rec.get("missing_paths", []),
                    "image_error":    rec.get("image_error"),
                    "has_image_error": rec.get("has_image_error", False),
                }
            )

        return interaction_log