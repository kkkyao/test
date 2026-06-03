from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from src.benchmarks.schemas import BenchmarkCase


class BenchmarkDataGenerator:
    """
    Generate matched benchmark cases across modalities.

    Supports two modes depending on whether target_variable / formula are set:

    - With target: generates input sequences + computed target values.
      Used by existing beers_wavelength benchmark.

    - Without target (target_variable omitted in config): generates input
      sequences only.  Used by the abstract visual encoding benchmark where
      there is no dependent variable in Phase 1.

    Paired assertion cases
    ----------------------
    For tasks that set generate_paired: true, the generator produces two extra
    cases per (sequence, task, modality) triple:
      - Positive assertion  (gold_answer = "yes")
      - Negative assertion  (gold_answer = "no", wrong value substituted)
    Both share a pair_id for Acc++ evaluation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.benchmark_cfg = config["benchmark"]

        self.seed = int(self.benchmark_cfg.get("seed", 42))
        self.rng = random.Random(self.seed)

        variables_cfg = self.benchmark_cfg["variables"]
        self.input_variables: List[str] = list(variables_cfg["input_variables"])
        self.target_variable: Optional[str] = variables_cfg.get("target_variable")
        self.variable_specs: Dict[str, Dict[str, Any]] = variables_cfg["specs"]

        self.variable_order: List[str] = list(self.input_variables)
        if self.target_variable:
            self.variable_order.append(self.target_variable)

        self.formula: Optional[str] = self.benchmark_cfg.get("formula")
        self.sequence_length = int(self.benchmark_cfg.get("sequence_length", 5))
        self.n_sequences = int(self.benchmark_cfg.get("n_sequences", 5))
        self.modalities: List[str] = list(self.benchmark_cfg["modalities"])
        self.tasks: List[Dict[str, Any]] = list(self.benchmark_cfg["tasks"])

        self.round_digits = int(self.benchmark_cfg.get("round_digits", 4))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_cases(self) -> List[BenchmarkCase]:
        cases: List[BenchmarkCase] = []

        for seq_idx in range(self.n_sequences):
            base_id = f"seq_{seq_idx:04d}"
            states = self._generate_sequence()

            for task in self.tasks:
                task_id = str(task["id"])
                task_type = str(task["type"])

                # For argmax, skip sequences where the max is not unique.
                if task_type == "argmax_at_step":
                    step_idx = self._resolve_step(task.get("step", "last"), states)
                    if not self._has_unique_max(states[step_idx]):
                        continue

                # For comparison, skip sequences where the two vars are equal.
                if task_type == "comparison_at_step":
                    step_idx = self._resolve_step(task.get("step", "last"), states)
                    var_a = str(task["variable_a"])
                    var_b = str(task["variable_b"])
                    if states[step_idx][var_a] == states[step_idx][var_b]:
                        continue

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
                            pair_id=None,
                        )
                    )

                # Paired assertion cases.
                if task.get("generate_paired", False):
                    pair_cases = self._generate_paired_cases(
                        base_id=base_id,
                        task=task,
                        states=states,
                        gold_answer=gold_answer,
                    )
                    cases.extend(pair_cases)

        return cases

    # ------------------------------------------------------------------
    # Sequence generation
    # ------------------------------------------------------------------

    def _generate_sequence(self) -> List[Dict[str, float]]:
        input_state = {
            var: self._sample_initial_value(var)
            for var in self.input_variables
        }

        states: List[Dict[str, float]] = []

        for step_idx in range(self.sequence_length):
            full_state = dict(input_state)
            if self.target_variable and self.formula:
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

    # ------------------------------------------------------------------
    # Task logic — answer computation
    # ------------------------------------------------------------------

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

        if task_type == "argmax_at_step":
            step = self._resolve_step(task.get("step", "last"), states)
            state = states[step]
            candidates = {
                var: state[var]
                for var in self.input_variables
                if var in state
            }
            return max(candidates, key=lambda v: candidates[v])

        if task_type == "comparison_at_step":
            step = self._resolve_step(task.get("step", "last"), states)
            var_a = str(task["variable_a"])
            var_b = str(task["variable_b"])
            return "yes" if states[step][var_a] > states[step][var_b] else "no"

        raise ValueError(f"Unsupported task type: {task_type}")

    # ------------------------------------------------------------------
    # Task logic — question formatting
    # ------------------------------------------------------------------

    def _format_question(self, task: Dict[str, Any], states: List[Dict[str, float]]) -> str:
        task_type = str(task["type"])

        variable = self._resolve_variable_soft(task.get("variable"))
        from_step = self._resolve_step(task.get("from_step", 0), states)
        to_step = self._resolve_step(task.get("to_step", "last"), states)
        step = self._resolve_step(task.get("step", "last"), states)

        template = str(task["question"])

        fmt: Dict[str, Any] = {
            "variable": variable or "",
            "from_step": from_step,
            "to_step": to_step,
            "step": step,
            "last_step": len(states) - 1,
        }

        if task_type == "comparison_at_step":
            fmt["variable_a"] = str(task.get("variable_a", ""))
            fmt["variable_b"] = str(task.get("variable_b", ""))

        return template.format(**fmt)

    # ------------------------------------------------------------------
    # Paired assertion generation
    # ------------------------------------------------------------------

    def _generate_paired_cases(
        self,
        base_id: str,
        task: Dict[str, Any],
        states: List[Dict[str, float]],
        gold_answer: Any,
    ) -> List[BenchmarkCase]:
        """
        Generate positive + negative assertion cases for a task.

        Positive: assertion states the TRUE answer  → gold = "yes"
        Negative: assertion states a WRONG answer   → gold = "no"

        Both cases share a pair_id for Acc++ computation.
        """
        task_id = str(task["id"])
        task_type = str(task["type"])

        pair_id = f"{base_id}__{task_id}__pair"

        pos_question = self._format_assertion(task, states, gold_answer, correct=True)
        wrong_answer = self._generate_wrong_answer(task, states, gold_answer)
        neg_question = self._format_assertion(task, states, wrong_answer, correct=False)

        cases: List[BenchmarkCase] = []

        for modality in self.modalities:
            chart_type = modality if modality != "text" else "none"

            for polarity, question, answer in [
                ("pos", pos_question, "yes"),
                ("neg", neg_question, "no"),
            ]:
                cases.append(
                    BenchmarkCase(
                        case_id=f"{base_id}__{task_id}__{modality}__{polarity}",
                        base_id=base_id,
                        task_id=f"{task_id}_{polarity}",
                        task_type=f"{task_type}_assertion",
                        modality=modality,
                        chart_type=chart_type,
                        states=states,
                        question=question,
                        gold_answer=answer,
                        answer_type="category",
                        target_variable=self.target_variable,
                        metadata={
                            "variable_order": list(self.variable_order),
                            "input_variables": list(self.input_variables),
                            "task": dict(task),
                            "polarity": polarity,
                            "open_ended_answer": gold_answer,
                        },
                        pair_id=pair_id,
                    )
                )

        return cases

    def _format_assertion(
        self,
        task: Dict[str, Any],
        states: List[Dict[str, float]],
        answer_value: Any,
        correct: bool,
    ) -> str:
        """Build the yes/no assertion question string."""
        task_type = str(task["type"])
        step = self._resolve_step(task.get("step", "last"), states)
        from_step = self._resolve_step(task.get("from_step", 0), states)
        to_step = self._resolve_step(task.get("to_step", "last"), states)

        if task_type == "value_at_step":
            var = self._resolve_variable_soft(task.get("variable")) or self.target_variable
            return (
                f"At step {step}, the value of {var} is {self._fmt(answer_value)}. "
                f"Answer yes or no."
            )

        if task_type == "argmax_at_step":
            all_vars = ", ".join(self.input_variables)
            return (
                f"At step {step}, {answer_value} has the largest value "
                f"among {all_vars}. Answer yes or no."
            )

        if task_type == "comparison_at_step":
            var_a = str(task["variable_a"])
            var_b = str(task["variable_b"])
            if correct:
                # Positive: the comparison that IS true
                if states[step][var_a] > states[step][var_b]:
                    return f"At step {step}, {var_a} is larger than {var_b}. Answer yes or no."
                else:
                    return f"At step {step}, {var_b} is larger than {var_a}. Answer yes or no."
            else:
                # Negative: the comparison that is NOT true
                if states[step][var_a] > states[step][var_b]:
                    return f"At step {step}, {var_b} is larger than {var_a}. Answer yes or no."
                else:
                    return f"At step {step}, {var_a} is larger than {var_b}. Answer yes or no."

        if task_type == "changed_variable":
            return (
                f"From step {from_step} to step {to_step}, "
                f"{answer_value} changed. Answer yes or no."
            )

        if task_type == "change_direction":
            var = self._resolve_variable_soft(task.get("variable")) or ""
            direction_phrase = {
                "increase": "increased",
                "decrease": "decreased",
                "same": "stayed the same",
            }.get(str(answer_value), str(answer_value))
            return (
                f"From step {from_step} to step {to_step}, "
                f"{var} {direction_phrase}. Answer yes or no."
            )

        raise ValueError(f"No assertion template for task type: {task_type}")

    def _generate_wrong_answer(
        self,
        task: Dict[str, Any],
        states: List[Dict[str, float]],
        gold_answer: Any,
    ) -> Any:
        """Generate a plausible but incorrect answer for the negative assertion."""
        task_type = str(task["type"])

        if task_type == "value_at_step":
            var = self._resolve_variable_soft(task.get("variable")) or self.target_variable
            spec = self.variable_specs.get(var, {})
            lo = int(spec.get("min_value", 1))
            hi = int(spec.get("max_value", 10))
            candidates = [v for v in range(lo, hi + 1) if v != int(gold_answer)]
            return float(self.rng.choice(candidates))

        if task_type == "argmax_at_step":
            # Pick a random variable that is NOT the true max.
            candidates = [v for v in self.input_variables if v != str(gold_answer)]
            return self.rng.choice(candidates)

        if task_type == "comparison_at_step":
            # Handled directly in _format_assertion by flipping the comparison.
            return gold_answer

        if task_type == "changed_variable":
            # Pick a variable that did NOT change.
            candidates = [v for v in self.input_variables if v != str(gold_answer)]
            return self.rng.choice(candidates) if candidates else gold_answer

        if task_type == "change_direction":
            gold = str(gold_answer)
            if gold == "increase":
                return "decrease"
            if gold == "decrease":
                return "increase"
            # "same" → pick increase or decrease
            return self.rng.choice(["increase", "decrease"])

        raise ValueError(f"No wrong-answer generator for task type: {task_type}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_unique_max(self, state: Dict[str, float]) -> bool:
        """Return True iff exactly one input variable has the strictly highest value."""
        values = [state[v] for v in self.input_variables if v in state]
        if not values:
            return False
        max_val = max(values)
        return values.count(max_val) == 1

    def _resolve_variable(self, value: Any) -> str:
        if value in {None, "target"}:
            if self.target_variable is None:
                raise ValueError("Task references 'target' but no target_variable is configured.")
            return self.target_variable
        return str(value)

    def _resolve_variable_soft(self, value: Any) -> Optional[str]:
        """Like _resolve_variable but returns None instead of raising for missing target."""
        if value in {None, "target"}:
            return self.target_variable
        return str(value)

    @staticmethod
    def _resolve_step(value: Any, states: List[Dict[str, float]]) -> int:
        if value == "last":
            return len(states) - 1
        return int(value)

    @staticmethod
    def _fmt(value: Any) -> str:
        if isinstance(value, float) and abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return str(value)
