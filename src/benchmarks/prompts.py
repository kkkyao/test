from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.benchmarks.schemas import BenchmarkCase


def build_benchmark_prompt(
    case: BenchmarkCase,
    image_paths: List[str],
    hint_level: Optional[str] = None,
) -> str:
    """
    Build the prompt for one benchmark case.

    Text modality:
        The numeric data table is included directly in the prompt.

    Visual modalities:
        The prompt describes the ordered image sequence.
        Numeric values are not included in text.

    hint_level controls how much information is given about chart contents:
        None        — no hint (default, original behaviour)
        "attention" — tells model that labels/step/range are readable in the chart
        "full"      — directly states variable names, range, and step size
    """
    if case.modality == "text":
        return _build_text_prompt(case)

    return _build_visual_prompt(case, image_paths, hint_level=hint_level)


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
    lines.extend(_answer_instructions(
        case.answer_type,
        task_type=case.task_type,
        input_variables=case.metadata.get("input_variables", []),
    ))

    return "\n".join(lines)


def _build_hint_lines(hint_level: str, case: BenchmarkCase) -> List[str]:
    """
    Return hint lines to insert into the prompt based on hint_level.

    "attention":  tells the model what is readable without revealing content.
    "full":       directly provides variable names, value range, and step size.
                  Designed for abstract chart modalities.
    "simulation": mirrors the structural context provided to the model in the
                  student simulation setup — variable names, ranges, step sizes,
                  available actions, and which variable is the computed output.
                  Use this to test whether simulation-only student simulation is
                  viable when the model has the same env context it would get in
                  a real episode.
    """
    if hint_level == "attention":
        return [
            "Note: The chart contains readable text. "
            "You can read the variable names, the current step number, "
            "and the value range directly from the image.",
            "",
        ]

    if hint_level == "full":
        vars_ = case.metadata.get("input_variables", [])
        specs = case.metadata.get("variable_specs", {})

        # Determine spatial layout label
        lr_modalities = {"bar_ocr", "bar_nocr", "dot_ocr", "dot_nocr"}
        series_modalities = {"line_ocr", "line_nocr", "scatter_ocr", "scatter_nocr"}
        if case.modality in lr_modalities:
            direction = "left to right"
        elif case.modality in series_modalities:
            direction = "top to bottom in the legend"
        else:
            direction = "top to bottom"

        lines = ["Chart reference information:"]
        if vars_:
            lines.append(
                "- Variables (%s): %s" % (direction, ", ".join(vars_))
            )
        if specs and vars_:
            first = specs.get(vars_[0], {})
            lo   = first.get("min_value", "")
            hi   = first.get("max_value", "")
            step = first.get("step", "")
            if lo != "" and hi != "":
                lines.append("- Value range: %d to %d" % (int(lo), int(hi)))
            if step != "":
                lines.append("- Variable increment per action: %d" % int(step))
        lines.append("")
        return lines

    if hint_level == "simulation":
        vars_ = case.metadata.get("input_variables", [])
        specs = case.metadata.get("variable_specs", {})
        target = case.target_variable or ""

        lines = [
            "Experiment context (same information available in the student simulation setup):",
            "- Input (controllable) variables shown in the Controls panel:",
        ]
        for var in vars_:
            spec = specs.get(var, {})
            lo   = spec.get("min_value", "")
            hi   = spec.get("max_value", "")
            step = spec.get("step", "")
            parts = []
            if lo != "" and hi != "":
                parts.append("range %s to %s" % (int(float(lo)), int(float(hi))))
            if step != "":
                s = float(step)
                parts.append("step %s" % (int(s) if s == int(s) else s))
            desc = (": " + ", ".join(parts)) if parts else ""
            lines.append("    %s%s" % (var, desc))

        if vars_:
            action_list = []
            for var in vars_:
                action_list.append("increase %s / decrease %s" % (var, var))
            lines.append("- Available actions: " + "; ".join(action_list))

        if target:
            lines.append("- Target (output) variable: %s" % target)
            lines.append(
                "  %s is a computed result shown on the output display — "
                "do NOT read %s when answering questions about input variable values."
                % (target, target)
            )

        lines.append(
            "When reading input variable values, look at the yellow highlighted box "
            "next to each variable name — that box shows the exact current value. "
            "Do not read the slider position, the endpoint labels, or any output display."
        )
        lines.append("")
        return lines

    return []


def _build_visual_prompt(
    case: BenchmarkCase,
    image_paths: List[str],
    hint_level: Optional[str] = None,
) -> str:
    lines: List[str] = []

    # ── Open visual perception tasks ─────────────────────────────────
    # Provide minimal structural grammar so the model understands what
    # kind of display it is looking at, without revealing variable names,
    # values, or chart-type details that the model should discover itself.
    if case.answer_type == "open":
        lines.append("You are shown a sequence of images from an experiment.")
        lines.append(
            "Each image is a snapshot of one time step: "
            "it shows the current values of several variables simultaneously. "
            "The step number is shown in the top-left corner of each image "
            "(for example, \"Step 3\" means this is the third time step)."
        )
        lines.append(
            "Each image is independent — it captures a single moment, "
            "not a time series."
        )
        lines.append("")
        lines.append(f"Number of images provided: {len(image_paths)}")
        lines.append("")
        if hint_level:
            lines.extend(_build_hint_lines(hint_level, case))
        lines.append("Question:")
        lines.append(case.question)
        lines.append("")
        lines.extend(_answer_instructions(
            case.answer_type,
            task_type=case.task_type,
            input_variables=case.metadata.get("input_variables", []),
        ))
        return "\n".join(lines)

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
        lines.append(
            "Reading guide: each variable is displayed as a row containing "
            "(1) the variable name in bold, "
            "(2) a yellow highlighted box showing the current value — this is the number to read, "
            "(3) a horizontal slider between the minimum (left) and maximum (right) values."
        )
        lines.append(
            "Always read the value from the yellow box. "
            "Do not estimate from slider position or endpoint labels."
        )
        lines.append("Use the ordered screenshots to answer the question.")

    # ── Abstract visual encoding: bar standalone ──────────────────────
    elif case.modality in {"bar_ocr", "bar_nocr"}:
        lines.append("You are given a sequence of bar-chart images.")
        lines.append("The images are ordered from oldest step (Step 1) to newest step.")
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
        lines.append("The images are ordered from oldest step (Step 1) to newest step.")
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
        lines.append("The images are ordered from oldest step (Step 1) to newest step.")
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
        lines.append("The images are ordered from oldest step (Step 1) to newest step.")
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
    if hint_level:
        lines.extend(_build_hint_lines(hint_level, case))
    lines.append("Question:")
    lines.append(case.question)
    lines.append("")
    lines.extend(_answer_instructions(
        case.answer_type,
        task_type=case.task_type,
        input_variables=case.metadata.get("input_variables", []),
    ))

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


def _answer_instructions(
    answer_type: str,
    task_type: str = "",
    input_variables: Optional[List[str]] = None,
) -> List[str]:
    if answer_type == "open":
        return ["Answer in your own words. Be as detailed and specific as possible."]

    lines: List[str] = []

    lines.append("Return exactly one JSON object and no other text:")
    lines.append('{"answer": <your_answer>}')
    lines.append("")

    if answer_type == "number":
        lines.append("The answer must be a number. Do not include units.")

    elif answer_type == "category":
        lines.append("The answer must be a short lowercase string.")

        if task_type.endswith("_assertion"):
            lines.append('Answer with exactly one of: "yes" or "no".')

        elif task_type == "change_direction":
            lines.append('Answer with exactly one of: "increase", "decrease", or "same".')

        elif task_type in {"changed_variable", "argmax_at_step"}:
            if input_variables:
                options = ", ".join(f'"{v}"' for v in input_variables)
                lines.append(f"Answer with exactly one of: {options}.")
            else:
                lines.append("Answer with exactly the variable name.")

        elif task_type == "comparison_at_step":
            lines.append('Answer with exactly one of: "yes" or "no".')

        else:
            lines.append("Answer with a short lowercase string.")

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