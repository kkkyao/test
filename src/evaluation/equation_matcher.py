from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import sympy as sp


class EquationMatcher:
    """
    Compare predicted equations against ground-truth equations.

    Supports:
    - algebraic equivalence
    - optional variable-name mapping
    """

    def __init__(self, variable_mapping: Optional[Dict[str, str]] = None) -> None:
        if variable_mapping is not None and not isinstance(variable_mapping, dict):
            raise ValueError("variable_mapping must be a dictionary or None")

        self.variable_mapping = variable_mapping or {}

    def match(self, predicted: str, ground_truth: str) -> bool:
        """
        Return True if the predicted equation is equivalent to the ground-truth equation.
        """
        if not isinstance(predicted, str) or not predicted.strip():
            return False

        if not isinstance(ground_truth, str) or not ground_truth.strip():
            return False

        try:
            pred_lhs, pred_rhs = self._normalize_equation(predicted, apply_mapping=True)
            gt_lhs, gt_rhs = self._normalize_equation(ground_truth, apply_mapping=False)

            if pred_lhs is not None and gt_lhs is not None:
                if pred_lhs != gt_lhs:
                    return False

            return self._expressions_equivalent(pred_rhs, gt_rhs)

        except Exception:
            return False

    def _normalize_equation(
        self,
        equation: str,
        apply_mapping: bool,
    ) -> Tuple[Optional[str], str]:
        """
        Normalize an equation into (lhs, rhs).

        If no '=' is present, lhs is None and the whole string is treated as rhs.
        """
        text = equation.strip()

        if apply_mapping:
            text = self._apply_variable_mapping(text)

        if "=" in text:
            lhs, rhs = text.split("=", 1)
            lhs = lhs.strip()
            rhs = rhs.strip()
            if not rhs:
                raise ValueError(f"invalid equation with empty rhs: '{equation}'")
            if not lhs:
                raise ValueError(f"invalid equation with empty lhs: '{equation}'")
            return lhs, rhs

        if not text:
            raise ValueError("equation must be a non-empty string")

        return None, text

    def _apply_variable_mapping(self, text: str) -> str:
        """
        Replace predicted variable names with ground-truth variable names.
        Uses word boundaries to avoid partial replacements.
        """
        mapped_text = text

        for source, target in self.variable_mapping.items():
            pattern = rf"\b{re.escape(source)}\b"
            mapped_text = re.sub(pattern, target, mapped_text)

        return mapped_text

    def _expressions_equivalent(self, expr1: str, expr2: str) -> bool:
        """
        Check whether two expressions are algebraically equivalent using sympy.
        """
        sympy_expr1 = sp.sympify(expr1)
        sympy_expr2 = sp.sympify(expr2)
        return sp.simplify(sympy_expr1 - sympy_expr2) == 0