from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Words that, if appearing as standalone tokens in a model's answer to an
# open visual-perception question, likely indicate domain hallucination
# (the model is describing a different scene, not the abstract a/b/c chart).
_HALLUCINATION_WORDS = frozenset({
    "price", "cost", "revenue", "sales", "profit",
    "customer", "customers", "user", "users",
    "employee", "employees", "stock",
    "temperature", "weight", "height", "population",
    "income", "wavelength", "concentration", "absorbance",
    "voltage", "current", "resistance",
    "speed", "distance", "frequency", "amplitude",
    "mass", "force", "pressure",
})

_AXIS_WORDS   = frozenset({"axis", "axes", "xaxis", "yaxis"})
_TICK_WORDS   = frozenset({"tick", "ticks"})
_LABEL_WORDS  = frozenset({"label", "labels", "labeled"})
_CHANGE_WORDS = frozenset({
    "change", "changed", "changes",
    "increase", "increases", "increased",
    "decrease", "decreases", "decreased",
})


def tag_open_answer(
    raw: str,
    input_variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Lightweight rule-based tagging of an open-ended model answer.
    No NLP — keyword presence checks and simple regex only.
    """
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""

    text_lower = raw.lower()
    word_set   = set(re.findall(r"[a-z]+", text_lower))

    gold_vars = [v.lower() for v in (input_variables or [])]

    mentioned    = [v for v in gold_vars if v in text_lower]
    missing      = [v for v in gold_vars if v not in text_lower]
    hallucinated = sorted(_HALLUCINATION_WORDS & word_set)

    step_numbers = [int(n) for n in re.findall(r"step\s*(\d+)", text_lower)]
    numbers      = re.findall(r"\b\d+(?:\.\d+)?\b", text_lower)

    return {
        "mentions_abc":                bool(mentioned),
        "mentioned_variables":         mentioned,
        "missing_gold_variables":      missing,
        "hallucinated_variables":      hallucinated,
        "obvious_domain_hallucination": bool(hallucinated),
        "mentions_step":               bool({"step", "steps"} & word_set),
        "mentioned_step_numbers":      step_numbers,
        "mentions_numbers":            bool(numbers),
        "extracted_numbers":           [float(n) for n in numbers[:10]],
        "mentions_axis":               bool(_AXIS_WORDS & word_set),
        "mentions_ticks":              bool(_TICK_WORDS & word_set),
        "mentions_labels":             bool(_LABEL_WORDS & word_set),
        "mentions_change":             bool(_CHANGE_WORDS & word_set),
        "mentioned_changed_variables": [],
    }


def parse_answer(raw_output: str, answer_type: str) -> Any:
    if answer_type == "open":
        return raw_output.strip() if isinstance(raw_output, str) else str(raw_output)

    obj = extract_json_object(raw_output)

    if isinstance(obj, dict) and "answer" in obj:
        return obj["answer"]

    # Fallbacks for models that ignore JSON instruction.
    if answer_type == "number":
        number = _extract_first_number(raw_output)
        return number

    if answer_type in {"step", "step_or_none"}:
        if re.search(r"\bnone\b", raw_output, flags=re.IGNORECASE):
            return "none"
        number = _extract_first_number(raw_output)
        if number is None:
            return None
        return int(round(number))

    text = raw_output.strip()
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
    return text if text else None


def extract_json_object(raw_output: str) -> Optional[dict]:
    text = raw_output.strip()

    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL,
    )
    if fence_match:
        candidate = fence_match.group(1).strip()
        parsed = _try_json_loads(candidate)
        if parsed is not None:
            return parsed

    candidate = _extract_by_brace_walk(text)
    if candidate is not None:
        parsed = _try_json_loads(candidate)
        if parsed is not None:
            return parsed

    return None


def _extract_by_brace_walk(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if ch == "\\" and in_string:
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _try_json_loads(text: str) -> Optional[dict]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_json(text)
        try:
            obj = json.loads(repaired)
        except json.JSONDecodeError:
            return None

    return obj if isinstance(obj, dict) else None


def _repair_json(text: str) -> str:
    repaired = re.sub(r",(\s*[}\]])", r"\1", text)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    return repaired


def _extract_first_number(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))