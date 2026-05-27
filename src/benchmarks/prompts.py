from __future__ import annotations

from typing import Any, Dict, List

from src.benchmarks.schemas import BenchmarkCase


def build_benchmark_prompt(case: BenchmarkCase, image_paths: List[str]) -> str:
    """
    Build the prompt for one benchmark case.

    Text modality:
        The numeric data table is included directly in the prompt.

    Visual modalities:
        The prompt describes the ordered image sequence.
        Numeric values are not included in text.
    """
    if case.modality == "text":
        return _build_text_prompt(case)

    return _build_visual_prompt(case, image_paths)


def _build_text_prompt(case: BenchmarkCase) -> str:
    variable_order = case.metadata.get("variable_order", [])

    lines: List[str] = []
    lines.append("You are given numeric observations from an experiment.")
    lines.append("Each row is one step.")
    lines.append("Use the table to answer the question.")
    lines.append("")
    lines.append(_format_state_table(case.states, variable_order))
    lines.append("")
    lines.append("Question:")
    lines.append(case.question)
    lines.append("")
    lines.extend(_answer_instructions(case.answer_type))

    return "\n".join(lines)


def _build_visual_prompt(case: BenchmarkCase, image_paths: List[str]) -> str:
    lines: List[str] = []

    if case.modality == "bar":
        lines.append("You are given a sequence of bar-chart images.")
        lines.append("The images are ordered from oldest step to newest step.")
        lines.append("Each image shows all variable values for one step.")
        lines.append("Use the ordered images to answer the question.")

    elif case.modality == "line":
        lines.append("You are given a sequence of line-plot images.")
        lines.append("The images are ordered from oldest step to newest step.")
        lines.append("Each image shows all variable values for one step.")
        lines.append("Use the ordered images to answer the question.")

    elif case.modality == "scatter":
        lines.append("You are given a sequence of scatter-plot images.")
        lines.append("The images are ordered from oldest step to newest step.")
        lines.append("Each image shows all variable values for one step.")
        lines.append("Use the ordered images to answer the question.")

    elif case.modality == "simulation":
        lines.append("You are given a sequence of simulation screenshots.")
        lines.append("The screenshots are ordered from oldest step to newest step.")
        lines.append("Each screenshot shows the experimental state at one step.")
        lines.append("Use the ordered screenshots to answer the question.")

    else:
        raise ValueError(f"Unsupported visual modality: {case.modality}")

    lines.append("")
    lines.append(f"Number of images provided: {len(image_paths)}")
    lines.append("")
    lines.append("Question:")
    lines.append(case.question)
    lines.append("")
    lines.extend(_answer_instructions(case.answer_type))

    return "\n".join(lines)


def _format_state_table(
    states: List[Dict[str, float]],
    variable_order: List[str],
) -> str:
    if not variable_order:
        variable_order = list(states[0].keys()) if states else []

    lines: List[str] = []

    header = ["step"] + variable_order
    lines.append(" | ".join(header))
    lines.append(" | ".join(["---"] * len(header)))

    for step_id, state in enumerate(states):
        row = [str(step_id)]
        for var in variable_order:
            row.append(_format_value(state[var]))
        lines.append(" | ".join(row))

    return "\n".join(lines)


def _answer_instructions(answer_type: str) -> List[str]:
    lines: List[str] = []

    lines.append("Return exactly one JSON object and no other text:")
    lines.append('{"answer": <your_answer>}')
    lines.append("")

    if answer_type == "number":
        lines.append("The answer must be a number. Do not include units.")

    elif answer_type == "category":
        lines.append("The answer must be a short lowercase string category.")
        lines.append('For direction questions, use one of: "increase", "decrease", "same".')
        lines.append("For variable-name questions, return exactly the variable name.")

    elif answer_type == "step":
        lines.append("The answer must be an integer step index.")

    elif answer_type == "step_or_none":
        lines.append('The answer must be an integer step index, or the string "none".')

    else:
        lines.append("The answer should be as short as possible.")

    return lines


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)