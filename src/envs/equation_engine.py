from __future__ import annotations

import ast
import operator
from typing import Any, Dict


class EquationEngine:
    """
    Compute output variables from config-defined equations and the current state.

    The engine supports simple arithmetic expressions and evaluates equations
    in the order they are defined.
    """

    _BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    _UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, equations: Dict[str, str]) -> None:
        if not equations:
            raise ValueError("equations must be a non-empty dictionary")

        for target, expr in equations.items():
            if not isinstance(target, str) or not target.strip():
                raise ValueError("equation target names must be non-empty strings")
            if not isinstance(expr, str) or not expr.strip():
                raise ValueError(f"equation for '{target}' must be a non-empty string")

        self.equations = dict(equations)

    def compute_one(self, target: str, state: Dict[str, Any]) -> float:
        """
        Compute one target variable using the current state.
        """
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary")

        if target not in self.equations:
            raise KeyError(f"unknown equation target: '{target}'")

        expr = self.equations[target]
        try:
            value = self._safe_eval(expr, state)
        except ZeroDivisionError as e:
            raise ValueError(
                f"division by zero while evaluating equation for '{target}': '{expr}'"
            ) from e

        return float(value)

    def compute_all(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute all configured target variables in definition order.

        Outputs computed earlier in the sequence are added to the local context,
        so later equations may depend on them.
        """
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary")

        context = dict(state)
        results: Dict[str, float] = {}

        for target, expr in self.equations.items():
            try:
                value = float(self._safe_eval(expr, context))
            except ZeroDivisionError as e:
                raise ValueError(
                    f"division by zero while evaluating equation for '{target}': '{expr}'"
                ) from e

            results[target] = value
            context[target] = value

        return results

    def _safe_eval(self, expr: str, variables: Dict[str, Any]) -> float:
        """
        Safely evaluate a simple arithmetic expression using AST.
        """
        try:
            node = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"invalid equation syntax: '{expr}'") from e

        return float(self._eval_node(node.body, variables))

    def _eval_node(self, node: ast.AST, variables: Dict[str, Any]) -> float:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"unsupported constant type: {type(node.value).__name__}")
            return float(node.value)

        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise KeyError(f"missing variable in state: '{node.id}'")
            value = variables[node.id]
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"variable '{node.id}' must be numeric, got {type(value).__name__}"
                )
            return float(value)

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._BIN_OPS:
                raise ValueError(f"unsupported binary operator: {op_type.__name__}")
            left = self._eval_node(node.left, variables)
            right = self._eval_node(node.right, variables)
            return float(self._BIN_OPS[op_type](left, right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._UNARY_OPS:
                raise ValueError(f"unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand, variables)
            return float(self._UNARY_OPS[op_type](operand))

        raise ValueError(f"unsupported expression node: {type(node).__name__}")