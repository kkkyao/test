from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# Allows "2x" to be parsed as "2*x", which models sometimes write.
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


class EquationMatcher:
    """
    Compare predicted equations against ground-truth equations.

    Supports:
    - algebraic equivalence via SymPy
    - optional variable-name mapping (abstract -> concrete)
    """

    def __init__(self, variable_mapping: Optional[Dict[str, str]] = None) -> None:
        if variable_mapping is not None and not isinstance(variable_mapping, dict):
            raise ValueError("variable_mapping must be a dictionary or None")

        self.variable_mapping = variable_mapping or {}

    def match(self, predicted: str, ground_truth: str) -> bool:
        """
        Return True if the predicted equation is algebraically equivalent
        to the ground-truth equation.

        Both sides of "lhs = rhs" are converted to a single expression
        (lhs - rhs) before comparison, so equations written with lhs and
        rhs swapped are still recognised as equivalent.

        If the prediction is a full equation but the ground truth is only
        an expression, also compare the predicted right-hand side against
        the ground-truth expression.
        """
        if not isinstance(predicted, str) or not predicted.strip():
            return False

        if not isinstance(ground_truth, str) or not ground_truth.strip():
            return False

        try:
            predicted_mapped = self._apply_variable_mapping(predicted)

            pred_expr = self._equation_to_expr(predicted_mapped)
            gt_expr   = self._equation_to_expr(ground_truth)

            if pred_expr is None or gt_expr is None:
                return False

            if self._expressions_equivalent(pred_expr, gt_expr):
                return True

            predicted_rhs = self._extract_rhs_if_equation(predicted_mapped)
            if predicted_rhs is not None and "=" not in ground_truth:
                rhs_expr = self._equation_to_expr(predicted_rhs)
                return self._expressions_equivalent(rhs_expr, gt_expr)

            return False

        except Exception:
            return False

    def _apply_variable_mapping(self, text: str) -> str:
        """
        Replace predicted variable names with ground-truth variable names.

        Keys are applied longest-first to prevent a short key (e.g. "A")
        from partially replacing a longer one (e.g. "AB") before it gets
        a chance to be substituted.

        Uses word boundaries to avoid replacing substrings inside longer words.
        """
        mapped_text = text

        sorted_items = sorted(
            self.variable_mapping.items(),
            key=lambda kv: -len(kv[0]),
        )

        for source, target in sorted_items:
            pattern = rf"\b{re.escape(source)}\b"
            mapped_text = re.sub(pattern, target, mapped_text)

        return mapped_text

    def _equation_to_expr(self, equation: str) -> Optional[sp.Expr]:
        """
        Parse an equation string into a single SymPy expression.

        "lhs = rhs"  →  parse(lhs) - parse(rhs)
        "expr"       →  parse(expr)

        The lhs - rhs form means side-swapped equations produce negations
        of each other, which is handled by _expressions_equivalent.
        """
        equation = equation.strip()

        if "=" in equation:
            lhs_str, rhs_str = equation.split("=", 1)
            lhs_str = lhs_str.strip()
            rhs_str = rhs_str.strip()
            if not lhs_str:
                raise ValueError(f"invalid equation with empty lhs: '{equation}'")
            if not rhs_str:
                raise ValueError(f"invalid equation with empty rhs: '{equation}'")
            lhs = parse_expr(lhs_str, transformations=_TRANSFORMATIONS)
            rhs = parse_expr(rhs_str, transformations=_TRANSFORMATIONS)
            return lhs - rhs

        if not equation:
            raise ValueError("equation must be a non-empty string")
        return parse_expr(equation, transformations=_TRANSFORMATIONS)

    @staticmethod
    def _extract_rhs_if_equation(equation: str) -> Optional[str]:
        """
        Return the right-hand side if the string contains an equation.
        Otherwise return None.
        """
        if "=" not in equation:
            return None

        _, rhs_str = equation.split("=", 1)
        rhs_str = rhs_str.strip()

        if not rhs_str:
            return None

        return rhs_str

    @staticmethod
    def _expressions_equivalent(expr1: sp.Expr, expr2: sp.Expr) -> bool:
        """
        Check algebraic equivalence between two SymPy expressions.

        Checks both (expr1 - expr2) and (expr1 + expr2) to handle the case
        where one equation has its sides swapped, producing a negation.
        """
        if sp.simplify(expr1 - expr2) == 0:
            return True

        if sp.simplify(expr1 + expr2) == 0:
            return True

        return False