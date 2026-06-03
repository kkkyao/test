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

    Optional W&B logging:
        - logs overall accuracy
        - logs per-modality accuracy
        - logs per-task accuracy
        - logs per-modality × per-task accuracy
        - uploads cases/results/aggregate files as an artifact
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str,
        max_cases: Optional[int] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_mode: Optional[str] = None,
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

        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_name = wandb_run_name
        self.wandb_mode = wandb_mode
        self.wandb_run = None

    def run(self) -> Dict[str, Any]:
        self._init_wandb_if_enabled()

        # Avoid duplicated rows if the same output directory is reused.
        jsonl_path = self.output_dir / "results.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()

        cases = self.generator.generate_cases()

        if self.max_cases is not None:
            cases = cases[: self.max_cases]

        self._save_json(
            self.output_dir / "cases.json",
            [case.to_dict() for case in cases],
        )

        results: List[Dict[str, Any]] = []

        try:
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
                    "pair_id": case.pair_id,
                }

                results.append(result)
                self._append_jsonl(self.output_dir / "results.jsonl", result)

            aggregate = self._aggregate(results)

            self._save_json(self.output_dir / "results.json", results)
            self._save_csv(self.output_dir / "results.csv", results)
            self._save_json(self.output_dir / "aggregate.json", aggregate)

            self._log_to_wandb(results=results, aggregate=aggregate)

            print("\n=== Benchmark complete ===")
            print(f"Cases: {len(results)}")
            print(f"Overall accuracy: {aggregate['overall']['accuracy']:.2%}")
            print(f"Saved to: {self.output_dir}")

            return aggregate

        finally:
            self._finish_wandb_if_enabled()

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

        if self.backend in {"hf_internvl", "hf_llava_ov"}:
            raise RuntimeError(
                f"Unsupported backend: '{self.backend}'. "
                "InternVL and LLaVA-OV are out of scope for this project. "
                "Use 'hf_qwen_vl' (Qwen2.5-VL), 'mock_vlm', 'hf_qwen', or 'mock_benchmark'."
            )

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
                    f"Backend '{self.backend}' is text-only but case modality is "
                    f"'{case.modality}'."
                )
            return self.model_callable(prompt)

        raise RuntimeError(f"Unknown model_type: {self.model_type}")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        agg = {
            "overall": self._aggregate_subset(results),
            "by_modality": self._aggregate_by(results, ["modality"]),
            "by_task_type": self._aggregate_by(results, ["task_type"]),
            "by_modality_and_task": self._aggregate_by(
                results,
                ["modality", "task_type"],
            ),
            "paired_accuracy": self._aggregate_paired(results),
            "ocr_gain": self._aggregate_ocr_gain(results),
        }
        return agg

    # ------------------------------------------------------------------
    # Paired Acc++ (Acc+)
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_paired(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Acc++: a pair is correct only if BOTH positive and negative cases are correct.

        Groups cases by pair_id; skips cases without pair_id.
        Reports Acc++ overall and by base_modality.
        """
        from collections import defaultdict

        pairs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in results:
            pid = row.get("pair_id")
            if pid:
                pairs[pid].append(row)

        if not pairs:
            return []

        pair_results: List[Dict[str, Any]] = []
        for pid, rows in pairs.items():
            all_correct = all(bool(r["correct"]) for r in rows)
            pair_results.append({
                "pair_id": pid,
                "pair_correct": all_correct,
                "modality": rows[0].get("modality", ""),
                "task_type": rows[0].get("task_type", "").replace("_assertion", ""),
                "n_cases": len(rows),
            })

        def _summarise(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
            n = len(subset)
            return {
                "n_pairs": n,
                "acc_pp": sum(r["pair_correct"] for r in subset) / n if n else 0.0,
            }

        by_modality: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for pr in pair_results:
            mod = pr["modality"]
            by_modality[mod].append(pr)

        return [
            {
                "scope": "overall",
                **_summarise(pair_results),
            }
        ] + [
            {
                "scope": f"modality/{mod}",
                "modality": mod,
                **_summarise(rows),
            }
            for mod, rows in sorted(by_modality.items())
        ]

    # ------------------------------------------------------------------
    # OCR Gain
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_ocr_gain(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        OCR Gain = accuracy(xxx_ocr) - accuracy(xxx_nocr) for each base viz.

        Only computed for results that use the abstract visual encoding modalities
        (modality names ending in _ocr or _nocr).
        """
        from collections import defaultdict

        ocr_rows: Dict[str, List[bool]] = defaultdict(list)
        nocr_rows: Dict[str, List[bool]] = defaultdict(list)

        for row in results:
            mod: str = row.get("modality", "")
            if mod.endswith("_ocr"):
                base = mod[: -len("_ocr")]
                ocr_rows[base].append(bool(row["correct"]))
            elif mod.endswith("_nocr"):
                base = mod[: -len("_nocr")]
                nocr_rows[base].append(bool(row["correct"]))

        all_bases = sorted(set(ocr_rows) | set(nocr_rows))
        gains = []
        for base in all_bases:
            ocr_acc = sum(ocr_rows[base]) / len(ocr_rows[base]) if ocr_rows[base] else None
            nocr_acc = (
                sum(nocr_rows[base]) / len(nocr_rows[base]) if nocr_rows[base] else None
            )
            gain = (
                round(ocr_acc - nocr_acc, 6)
                if ocr_acc is not None and nocr_acc is not None
                else None
            )
            gains.append(
                {
                    "base_modality": base,
                    "n_ocr": len(ocr_rows[base]),
                    "n_nocr": len(nocr_rows[base]),
                    "ocr_accuracy": ocr_acc,
                    "nocr_accuracy": nocr_acc,
                    "ocr_gain": gain,
                }
            )
        return gains

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
    # W&B
    # ------------------------------------------------------------------

    def _init_wandb_if_enabled(self) -> None:
        if not self.wandb_project:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required when --wandb_project is provided. "
                "Install with: pip install wandb"
            ) from exc

        init_kwargs: Dict[str, Any] = {
            "project": self.wandb_project,
            "entity": self.wandb_entity,
            "name": self.wandb_run_name,
            "config": {
                "benchmark": self.config.get("benchmark", {}),
                "visual": self.config.get("visual", {}),
                "agent": self._safe_agent_config_for_wandb(),
                "scoring": self.config.get("scoring", {}),
                "output_dir": str(self.output_dir),
                "max_cases": self.max_cases,
                "backend": self.backend,
                "model_type": self.model_type,
            },
            "tags": self._wandb_tags(),
        }

        if self.wandb_mode:
            init_kwargs["mode"] = self.wandb_mode

        self.wandb_run = wandb.init(**init_kwargs)

    def _log_to_wandb(
        self,
        results: List[Dict[str, Any]],
        aggregate: Dict[str, Any],
    ) -> None:
        if self.wandb_run is None:
            return

        import wandb

        # Main scalar metrics
        wandb.log(
            {
                "overall/accuracy": aggregate["overall"]["accuracy"],
                "overall/n": aggregate["overall"]["n"],
            }
        )

        if aggregate["overall"].get("mean_numeric_error") is not None:
            wandb.log(
                {
                    "overall/mean_numeric_error": aggregate["overall"][
                        "mean_numeric_error"
                    ]
                }
            )

        # Per-modality summary
        for row in aggregate.get("by_modality", []):
            modality = row["modality"]
            wandb.summary[f"accuracy/modality/{modality}"] = row["accuracy"]
            wandb.summary[f"n/modality/{modality}"] = row["n"]

            if row.get("mean_numeric_error") is not None:
                wandb.summary[f"mean_numeric_error/modality/{modality}"] = row[
                    "mean_numeric_error"
                ]

        # Per-task summary
        for row in aggregate.get("by_task_type", []):
            task_type = row["task_type"]
            wandb.summary[f"accuracy/task/{task_type}"] = row["accuracy"]
            wandb.summary[f"n/task/{task_type}"] = row["n"]

            if row.get("mean_numeric_error") is not None:
                wandb.summary[f"mean_numeric_error/task/{task_type}"] = row[
                    "mean_numeric_error"
                ]

        # Per-modality × task summary
        for row in aggregate.get("by_modality_and_task", []):
            modality = row["modality"]
            task_type = row["task_type"]
            key = f"{modality}/{task_type}"
            wandb.summary[f"accuracy/modality_task/{key}"] = row["accuracy"]
            wandb.summary[f"n/modality_task/{key}"] = row["n"]

            if row.get("mean_numeric_error") is not None:
                wandb.summary[f"mean_numeric_error/modality_task/{key}"] = row[
                    "mean_numeric_error"
                ]

        # W&B tables
        modality_table = wandb.Table(
            columns=["modality", "n", "accuracy", "mean_numeric_error"]
        )
        for row in aggregate.get("by_modality", []):
            modality_table.add_data(
                row.get("modality"),
                row.get("n"),
                row.get("accuracy"),
                row.get("mean_numeric_error"),
            )
        wandb.log({"tables/by_modality": modality_table})

        task_table = wandb.Table(
            columns=["task_type", "n", "accuracy", "mean_numeric_error"]
        )
        for row in aggregate.get("by_task_type", []):
            task_table.add_data(
                row.get("task_type"),
                row.get("n"),
                row.get("accuracy"),
                row.get("mean_numeric_error"),
            )
        wandb.log({"tables/by_task_type": task_table})

        modality_task_table = wandb.Table(
            columns=[
                "modality",
                "task_type",
                "n",
                "accuracy",
                "mean_numeric_error",
            ]
        )
        for row in aggregate.get("by_modality_and_task", []):
            modality_task_table.add_data(
                row.get("modality"),
                row.get("task_type"),
                row.get("n"),
                row.get("accuracy"),
                row.get("mean_numeric_error"),
            )
        wandb.log({"tables/by_modality_and_task": modality_task_table})

        # Per-case table. Keep it compact; full raw outputs are in artifact files.
        case_table = wandb.Table(
            columns=[
                "case_id",
                "base_id",
                "task_id",
                "task_type",
                "modality",
                "gold_answer",
                "parsed_answer",
                "correct",
                "numeric_error",
            ]
        )
        for row in results:
            case_table.add_data(
                row.get("case_id"),
                row.get("base_id"),
                row.get("task_id"),
                row.get("task_type"),
                row.get("modality"),
                str(row.get("gold_answer")),
                str(row.get("parsed_answer")),
                bool(row.get("correct")),
                row.get("numeric_error"),
            )
        wandb.log({"tables/per_case_results": case_table})

        # Upload core result files as artifact.
        artifact_name = self._artifact_name()
        artifact = wandb.Artifact(
            name=artifact_name,
            type="visual_benchmark_results",
            metadata={
                "output_dir": str(self.output_dir),
                "backend": self.backend,
                "model_type": self.model_type,
                "n_results": len(results),
                "overall_accuracy": aggregate["overall"]["accuracy"],
            },
        )

        for filename in [
            "cases.json",
            "results.json",
            "results.jsonl",
            "results.csv",
            "aggregate.json",
        ]:
            path = self.output_dir / filename
            if path.exists():
                artifact.add_file(str(path), name=filename)

        wandb.log_artifact(artifact)

    def _finish_wandb_if_enabled(self) -> None:
        if self.wandb_run is None:
            return

        import wandb

        wandb.finish()
        self.wandb_run = None

    def _safe_agent_config_for_wandb(self) -> Dict[str, Any]:
        """
        Keep the agent config JSON-friendly and avoid logging very large objects.
        """
        safe: Dict[str, Any] = {}

        for key, value in self.agent_cfg.items():
            if key in {"model", "processor", "tokenizer"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[key] = value
            elif isinstance(value, (list, dict)):
                safe[key] = value
            else:
                safe[key] = str(value)

        return safe

    def _wandb_tags(self) -> List[str]:
        benchmark_cfg = self.config.get("benchmark", {})
        modalities = benchmark_cfg.get("modalities", [])

        tags = [
            "visual-benchmark",
            str(self.backend),
            str(self.model_type),
            str(benchmark_cfg.get("name", "benchmark")),
        ]

        model_name = self.agent_cfg.get("model_name")
        if model_name:
            tags.append(str(model_name).replace("/", "_"))

        if isinstance(modalities, list):
            tags.extend([f"modality_{m}" for m in modalities])

        return tags

    def _artifact_name(self) -> str:
        raw_name = self.wandb_run_name or self.output_dir.name
        safe_name = (
            raw_name.replace("/", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )
        return f"{safe_name}_results"

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
            "pair_id",
        ]

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                csv_row = dict(row)
                csv_row["image_paths"] = json.dumps(
                    row["image_paths"],
                    ensure_ascii=False,
                )
                csv_row["states"] = json.dumps(
                    row["states"],
                    ensure_ascii=False,
                )
                writer.writerow(csv_row)