from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from run_episode import build_model_callable, build_vlm_callable

from src.benchmarks.data_generator import BenchmarkDataGenerator
from src.benchmarks.parsers import parse_answer
from src.benchmarks.prompts import build_benchmark_prompt
from src.benchmarks.renderers import BenchmarkVisualRenderer
from src.benchmarks.scorer import score_answer
from src.benchmarks.schemas import BenchmarkCase


class VisualBenchmarkRunner:
    """
    Static visual/text QA benchmark runner.

    This intentionally does NOT use EpisodeRunner.
    It runs:
      case -> render modality -> prompt -> model -> parse answer -> score
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str,
        max_cases: Optional[int] = None,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_cases = max_cases
        self.tolerance = float(config.get("scoring", {}).get("tolerance", 1e-6))

        self.generator = BenchmarkDataGenerator(config)
        self.renderer = BenchmarkVisualRenderer(config, str(self.output_dir))

        self.agent_cfg = config.get("agent", {})
        self.backend = self.agent_cfg.get("backend", "mock_benchmark")

        self.model_type: str
        self.model_callable: Optional[Callable] = None
        self._build_model()

    def run(self) -> Dict[str, Any]:
        cases = self.generator.generate_cases()

        if self.max_cases is not None:
            cases = cases[: self.max_cases]

        self._save_json(
            self.output_dir / "cases.json",
            [case.to_dict() for case in cases],
        )

        results: List[Dict[str, Any]] = []

        for idx, case in enumerate(cases):
            print(f"[{idx + 1}/{len(cases)}] {case.case_id}")

            image_paths = self.renderer.render_case(case)
            prompt = build_benchmark_prompt(case, image_paths)

            raw_output = self._call_model(case, prompt, image_paths)
            parsed_answer = parse_answer(raw_output, case.answer_type)

            score = score_answer(
                predicted=parsed_answer,
                gold=case.gold_answer,
                answer_type=case.answer_type,
                tolerance=self.tolerance,
            )

            result = {
                "case_id": case.case_id,
                "base_id": case.base_id,
                "task_id": case.task_id,
                "task_type": case.task_type,
                "modality": case.modality,
                "chart_type": case.chart_type,
                "question": case.question,
                "gold_answer": case.gold_answer,
                "answer_type": case.answer_type,
                "parsed_answer": parsed_answer,
                "correct": bool(score["correct"]),
                "numeric_error": score.get("numeric_error"),
                "raw_output": raw_output,
                "image_paths": image_paths,
                "states": case.states,
            }

            results.append(result)

            self._append_jsonl(self.output_dir / "results.jsonl", result)

        aggregate = self._aggregate(results)

        self._save_json(self.output_dir / "results.json", results)
        self._save_csv(self.output_dir / "results.csv", results)
        self._save_json(self.output_dir / "aggregate.json", aggregate)

        print("\n=== Benchmark complete ===")
        print(f"Cases: {len(results)}")
        print(f"Overall accuracy: {aggregate['overall']['accuracy']:.2%}")
        print(f"Saved to: {self.output_dir}")

        return aggregate

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        if self.backend == "mock_benchmark":
            self.model_type = "oracle"
            self.model_callable = None
            return

        if self.backend in {"hf_qwen_vl", "mock_vlm"}:
            self.model_type = "vlm"
            self.model_callable = build_vlm_callable(self.agent_cfg)
            return

        if self.backend in {"hf_qwen", "mock"}:
            self.model_type = "llm"
            self.model_callable = build_model_callable(self.agent_cfg)
            return

        raise ValueError(
            f"Unsupported benchmark backend: {self.backend}. "
            "Use mock_benchmark, hf_qwen_vl, mock_vlm, hf_qwen, or mock."
        )

    def _call_model(
        self,
        case: BenchmarkCase,
        prompt: str,
        image_paths: List[str],
    ) -> str:
        if self.model_type == "oracle":
            return json.dumps({"answer": case.gold_answer}, ensure_ascii=False)

        if self.model_callable is None:
            raise RuntimeError("model_callable was not initialized")

        if self.model_type == "vlm":
            # Same VLM can be tested on text-only cases by passing [] images.
            return self.model_callable(prompt, image_paths)

        if self.model_type == "llm":
            if case.modality != "text":
                raise ValueError(
                    f"Backend '{self.backend}' is text-only but case modality is '{case.modality}'."
                )
            return self.model_callable(prompt)

        raise RuntimeError(f"Unknown model_type: {self.model_type}")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "overall": self._aggregate_subset(results),
            "by_modality": self._aggregate_by(results, ["modality"]),
            "by_task_type": self._aggregate_by(results, ["task_type"]),
            "by_modality_and_task": self._aggregate_by(results, ["modality", "task_type"]),
        }

    def _aggregate_by(
        self,
        results: List[Dict[str, Any]],
        keys: List[str],
    ) -> List[Dict[str, Any]]:
        groups: Dict[tuple, List[Dict[str, Any]]] = {}

        for row in results:
            key = tuple(row[k] for k in keys)
            groups.setdefault(key, []).append(row)

        output = []

        for key, rows in sorted(groups.items()):
            summary = self._aggregate_subset(rows)
            for k, v in zip(keys, key):
                summary[k] = v
            output.append(summary)

        return output

    @staticmethod
    def _aggregate_subset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {
                "n": 0,
                "accuracy": 0.0,
                "mean_numeric_error": None,
            }

        numeric_errors = [
            row["numeric_error"]
            for row in rows
            if row.get("numeric_error") is not None
        ]

        return {
            "n": n,
            "accuracy": sum(bool(row["correct"]) for row in rows) / n,
            "mean_numeric_error": (
                statistics.mean(numeric_errors) if numeric_errors else None
            ),
        }

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        fieldnames = [
            "case_id",
            "base_id",
            "task_id",
            "task_type",
            "modality",
            "chart_type",
            "question",
            "gold_answer",
            "answer_type",
            "parsed_answer",
            "correct",
            "numeric_error",
            "raw_output",
            "image_paths",
            "states",
        ]

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                csv_row = dict(row)
                csv_row["image_paths"] = json.dumps(row["image_paths"], ensure_ascii=False)
                csv_row["states"] = json.dumps(row["states"], ensure_ascii=False)
                writer.writerow(csv_row)