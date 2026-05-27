from __future__ import annotations

from typing import Any, Dict, Optional


def score_answer(
    predicted: Any,
    gold: Any,
    answer_type: str,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    if answer_type == "number":
        return _score_number(predicted, gold, tolerance)

    if answer_type == "category":
        return _score_category(predicted, gold)

    if answer_type == "step":
        return _score_step(predicted, gold)

    if answer_type == "step_or_none":
        return _score_step_or_none(predicted, gold)

    return _score_category(predicted, gold)


def _score_number(predicted: Any, gold: Any, tolerance: float) -> Dict[str, Any]:
    pred_num = _to_float(predicted)
    gold_num = _to_float(gold)

    if pred_num is None or gold_num is None:
        return {
            "correct": False,
            "numeric_error": None,
        }

    error = abs(pred_num - gold_num)

    return {
        "correct": error <= tolerance,
        "numeric_error": error,
    }


def _score_category(predicted: Any, gold: Any) -> Dict[str, Any]:
    pred_norm = _norm_text(predicted)
    gold_norm = _norm_text(gold)

    return {
        "correct": pred_norm == gold_norm,
        "numeric_error": None,
    }


def _score_step(predicted: Any, gold: Any) -> Dict[str, Any]:
    pred_step = _to_int(predicted)
    gold_step = _to_int(gold)

    return {
        "correct": pred_step is not None and gold_step is not None and pred_step == gold_step,
        "numeric_error": None,
    }


def _score_step_or_none(predicted: Any, gold: Any) -> Dict[str, Any]:
    if _norm_text(gold) == "none":
        return {
            "correct": _norm_text(predicted) == "none",
            "numeric_error": None,
        }

    return _score_step(predicted, gold)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(round(float(value)))
    except Exception:
        return None


def _norm_text(value: Any) -> str:
    return str(value).strip().lower()