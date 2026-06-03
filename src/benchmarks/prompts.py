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

    # ── Original modalities ───────────────────────────────────────────
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

    # ── Abstract visual encoding: bar standalone ──────────────────────
    elif case.modality in {"bar_ocr", "bar_nocr"}:
        lines.append("You are given a sequence of bar-chart images.")
        lines.append("The images are ordered from oldest step (step 0) to newest step.")
        lines.append("Each image shows one step of the experiment.")
        lines.append("The Controls panel contains vertical bars for each input variable.")
        lines.append("All bars share the same scale (range 1 to 10).")
        lines.append("Bar height reflects the variable's value within that range.")
        if case.modality == "bar_ocr":
            lines.append("The numeric value is printed above each bar.")
        else:
            lines.append("No numeric labels are shown; read values from bar heights.")
        lines.append("Variable names are shown below each bar.")
        lines.append("Use the ordered images to answer the question.")

    # ── Abstract visual encoding: slider ─────────────────────────────
    elif case.modality in {"slider_ocr", "slider_nocr"}:
        lines.append("You are given a sequence of slider observation images.")
        lines.append("The images are ordered from oldest step (step 0) to newest step.")
        lines.append("Each image shows one step of the experiment.")
        lines.append(
            "Each variable is represented by a horizontal slider "
            "with a track from 1 (left) to 10 (right)."
        )
        lines.append("Tick marks are placed at each integer position 1 through 10.")
        lines.append("The knob (filled circle) on the track indicates the current value.")
        if case.modality == "slider_ocr":
            lines.append("The numeric value is shown above the knob.")
        else:
            lines.append(
                "No numeric labels are shown; read the value from "
                "the knob's position on the track."
            )
        lines.append("Variable names are shown to the left of each slider.")
        lines.append("Use the ordered images to answer the question.")

    # ── Abstract visual encoding: dot plot (vertical) ────────────────
    elif case.modality in {"dot_ocr", "dot_nocr"}:
        lines.append("You are given a sequence of dot-chart images.")
        lines.append("The images are ordered from oldest step (step 0) to newest step.")
        lines.append("Each image shows one step of the experiment.")
        lines.append(
            "Each image contains a vertical dot chart where the x-axis shows variable "
            "names and the y-axis shows values from 1 to 10."
        )
        lines.append("Each variable is represented by a single dot placed at its current value height.")
        lines.append("The y-axis has tick marks at every integer from 1 to 10.")
        if case.modality == "dot_ocr":
            lines.append("The numeric value is shown above each dot.")
        else:
            lines.append(
                "No numeric labels are shown; read each variable's value from "
                "the vertical position of its dot on the y-axis."
            )
        lines.append("Use the ordered images to answer the question.")

    # ── Abstract visual encoding: grid ────────────────────────────────
    elif case.modality in {"grid_ocr", "grid_nocr"}:
        lines.append("You are given a sequence of unit-grid images.")
        lines.append("The images are ordered from oldest step (step 0) to newest step.")
        lines.append("Each image shows one step of the experiment.")
        lines.append(
            "Each variable is represented by a row of 10 cells. "
            "Filled (coloured) cells indicate the current value: "
            "a variable with value 3 has 3 filled cells and 7 empty cells."
        )
        lines.append(
            "Count the filled cells in a variable's row to determine its value."
        )
        if case.modality == "grid_ocr":
            lines.append("The numeric value is shown to the right of each row of cells.")
        else:
            lines.append(
                "No numeric labels are shown; count filled cells to read each variable's value."
            )
        lines.append("Variable names are shown to the left of each row.")
        lines.append("Use the ordered images to answer the question.")

    # ── Abstract visual encoding: line chart (history) ────────────────
    elif case.modality in {"line_ocr", "line_nocr"}:
        lines.append("You are given a sequence of line-chart images.")
        lines.append("The images are ordered from oldest to newest.")
        lines.append(
            "Each image shows the history of variable values from step 0 "
            "up to and including the current step."
        )
        lines.append(
            "The x-axis shows step index; the y-axis shows variable value (range 1–10)."
        )
        lines.append("Each variable is drawn as a separate coloured line.")
        lines.append(
            "The rightmost data point on each line corresponds to the current step."
        )
        if case.modality == "line_ocr":
            lines.append(
                "Numeric value labels are shown next to every data point "
                "(all steps, all variables)."
            )
        else:
            lines.append(
                "No numeric labels are shown; read values from point positions on the chart."
            )
        lines.append("Use the images to answer the question.")

    # ── Abstract visual encoding: scatter chart (history) ─────────────
    elif case.modality in {"scatter_ocr", "scatter_nocr"}:
        lines.append("You are given a sequence of scatter-plot images.")
        lines.append("The images are ordered from oldest to newest.")
        lines.append(
            "Each image shows the history of variable values from step 0 "
            "up to and including the current step."
        )
        lines.append(
            "The x-axis shows step index; the y-axis shows variable value (range 1–10)."
        )
        lines.append("Each variable is drawn as a separate coloured set of dots (no connecting lines).")
        lines.append(
            "The rightmost dot for each variable corresponds to the current step."
        )
        if case.modality == "scatter_ocr":
            lines.append(
                "Numeric value labels are shown next to every data point "
                "(all steps, all variables)."
            )
        else:
            lines.append(
                "No numeric labels are shown; read values from dot positions on the chart."
            )
        lines.append("Use the images to answer the question.")

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