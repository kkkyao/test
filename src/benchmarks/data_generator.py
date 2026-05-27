from __future__ import annotations

import random
from typing import Any, Dict, List

from src.benchmarks.schemas import BenchmarkCase


class BenchmarkDataGenerator:
    """
    Generate matched benchmark cases across modalities.

    The config defines:
    - variables and ranges
    - formula for target variable
    - task templates
    - modalities

    The generator creates numeric sequences and computes gold answers
    automatically.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.benchmark_cfg = config["benchmark"]

        self.seed = int(self.benchmark_cfg.get("seed", 42))
        self.rng = random.Random(self.seed)

        variables_cfg = self.benchmark_cfg["variables"]
        self.input_variables: List[str] = list(variables_cfg["input_variables"])
        self.target_variable: str = variables_cfg["target_variable"]
        self.variable_specs: Dict[str, Dict[str, Any]] = variables_cfg["specs"]

        self.variable_order: List[str] = (
            list(self.input_variables) + [self.target_variable]
        )

        self.formula: str = self.benchmark_cfg["formula"]
        self.sequence_length = int(self.benchmark_cfg.get("sequence_length", 5))
        self.n_sequences = int(self.benchmark_cfg.get("n_sequences", 5))
        self.modalities: List[str] = list(self.benchmark_cfg["modalities"])
        self.tasks: List[Dict[str, Any]] = list(self.benchmark_cfg["tasks"])

        self.round_digits = int(self.benchmark_cfg.get("round_digits", 4))

    def generate_cases(self) -> List[BenchmarkCase]:
        cases: List[BenchmarkCase] = []

        for seq_idx in range(self.n_sequences):
            base_id = f"seq_{seq_idx:04d}"
            states = self._generate_sequence()

            for task in self.tasks:
                task_id = str(task["id"])
                task_type = str(task["type"])

                gold_answer = self._compute_answer(task, states)
                question = self._format_question(task, states)

                for modality in self.modalities:
                    chart_type = modality if modality != "text" else "none"

                    case_id = f"{base_id}__{task_id}__{modality}"

                    cases.append(
                        BenchmarkCase(
                            case_id=case_id,
                            base_id=base_id,
                            task_id=task_id,
                            task_type=task_type,
                            modality=modality,
                            chart_type=chart_type,
                            states=states,
                            question=question,
                            gold_answer=gold_answer,
                            answer_type=str(task.get("answer_type", "number")),
                            target_variable=self.target_variable,
                            metadata={
                                "variable_order": list(self.variable_order),
                                "input_variables": list(self.input_variables),
                                "task": dict(task),
                            },
                        )
                    )

        return cases

    # ---------------------------------------------------------------------
    # Sequence generation
    # ---------------------------------------------------------------------

    def _generate_sequence(self) -> List[Dict[str, float]]:
        input_state = {
            var: self._sample_initial_value(var)
            for var in self.input_variables
        }

        states: List[Dict[str, float]] = []

        for step_idx in range(self.sequence_length):
            full_state = dict(input_state)
            full_state[self.target_variable] = self._evaluate_target(full_state)
            states.append(self._round_state(full_state))

            if step_idx < self.sequence_length - 1:
                input_state = self._mutate_one_input(input_state)

        return states

    def _sample_initial_value(self, var: str) -> float:
        spec = self.variable_specs[var]

        if "values" in spec:
            return float(self.rng.choice(spec["values"]))

        lo = float(spec["min_value"])
        hi = float(spec["max_value"])
        step = spec.get("step")
        decimals = int(spec.get("decimals", 0))

        if step is not None:
            values = []
            cur = lo
            step = float(step)
            while cur <= hi + 1e-9:
                values.append(cur)
                cur += step
            return float(self.rng.choice(values))

        value = self.rng.uniform(lo, hi)
        return round(value, decimals)

    def _mutate_one_input(self, state: Dict[str, float]) -> Dict[str, float]:
        new_state = dict(state)

        candidates = list(self.input_variables)
        self.rng.shuffle(candidates)

        for var in candidates:
            spec = self.variable_specs[var]
            lo = float(spec["min_value"])
            hi = float(spec["max_value"])
            step = float(spec.get("step", 1.0))
            decimals = int(spec.get("decimals", 0))

            current = float(new_state[var])

            possible_values = []
            for direction in (-1, 1):
                candidate = current + direction * step
                if lo <= candidate <= hi:
                    possible_values.append(round(candidate, decimals))

            if possible_values:
                new_state[var] = float(self.rng.choice(possible_values))
                return new_state

        return new_state

    def _evaluate_target(self, values: Dict[str, float]) -> float:
        rhs = self.formula.split("=", 1)[1] if "=" in self.formula else self.formula

        safe_globals = {"__builtins__": {}}
        safe_locals = {
            **values,
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
        }

        value = eval(rhs, safe_globals, safe_locals)
        return float(value)

    def _round_state(self, state: Dict[str, float]) -> Dict[str, float]:
        rounded: Dict[str, float] = {}

        for var, value in state.items():
            spec = self.variable_specs.get(var, {})
            decimals = int(spec.get("decimals", self.round_digits))
            rounded[var] = round(float(value), decimals)

        return rounded

    # ---------------------------------------------------------------------
    # Task logic
    # ---------------------------------------------------------------------

    def _compute_answer(self, task: Dict[str, Any], states: List[Dict[str, float]]) -> Any:
        task_type = str(task["type"])

        if task_type == "value_at_step":
            var = self._resolve_variable(task.get("variable", "target"))
            step = self._resolve_step(task.get("step", "last"), states)
            return states[step][var]

        if task_type == "change_amount":
            var = self._resolve_variable(task.get("variable", "target"))
            from_step = self._resolve_step(task.get("from_step", 0), states)
            to_step = self._resolve_step(task.get("to_step", "last"), states)
            return round(states[to_step][var] - states[from_step][var], self.round_digits)

        if task_type == "change_direction":
            var = self._resolve_variable(task.get("variable", "target"))
            from_step = self._resolve_step(task.get("from_step", 0), states)
            to_step = self._resolve_step(task.get("to_step", "last"), states)
            diff = states[to_step][var] - states[from_step][var]
            if diff > 0:
                return "increase"
            if diff < 0:
                return "decrease"
            return "same"

        if task_type == "changed_variable":
            from_step = self._resolve_step(task.get("from_step", 0), states)
            to_step = self._resolve_step(task.get("to_step", 1), states)

            changed = []
            for var in self.input_variables:
                if states[from_step][var] != states[to_step][var]:
                    changed.append(var)

            if len(changed) == 1:
                return changed[0]
            if not changed:
                return "none"
            return ",".join(changed)

        if task_type == "max_step":
            var = self._resolve_variable(task.get("variable", "target"))
            values = [state[var] for state in states]
            return int(max(range(len(values)), key=lambda i: values[i]))

        if task_type == "largest_drop_step":
            var = self._resolve_variable(task.get("variable", "target"))

            best_later_step = None
            best_delta = 0.0

            for i in range(1, len(states)):
                delta = states[i][var] - states[i - 1][var]
                if delta < best_delta:
                    best_delta = delta
                    best_later_step = i

            if best_later_step is None:
                return "none"
            return int(best_later_step)

        raise ValueError(f"Unsupported task type: {task_type}")

    def _format_question(self, task: Dict[str, Any], states: List[Dict[str, float]]) -> str:
        variable = self._resolve_variable(task.get("variable", "target"))
        from_step = self._resolve_step(task.get("from_step", 0), states)
        to_step = self._resolve_step(task.get("to_step", "last"), states)
        step = self._resolve_step(task.get("step", "last"), states)

        template = str(task["question"])

        return template.format(
            variable=variable,
            target=self.target_variable,
            from_step=from_step,
            to_step=to_step,
            step=step,
            last_step=len(states) - 1,
        )

    def _resolve_variable(self, value: Any) -> str:
        if value in {None, "target"}:
            return self.target_variable
        return str(value)

    @staticmethod
    def _resolve_step(value: Any, states: List[Dict[str, float]]) -> int:
        if value == "last":
            return len(states) - 1
        return int(value)