from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

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

    Matching strategy
    -----------------
    Models in abstract naming mode may write equations using:
      (a) the display names they were shown  (e.g. "epsilon", "c", "l")
      (b) the concrete names from prior knowledge  (e.g. "molar_absorptivity")
      (c) a mix of both

    To handle all three cases the matcher tries the following strategies
    IN ORDER and returns True on the first success:

      1. Apply variable_mapping  (display → concrete)
         Converts the model's display names to concrete names, then
         compares against the concrete ground-truth expression.

      2. No mapping
         Compares the equation as-is against the ground truth.
         Catches the common case where the model used concrete names
         directly and the ground truth is also in concrete form.

      3. Apply reverse_mapping  (concrete → display)
         In case the model wrote concrete names but the ground truth
         uses display names (rare with current configs, but safe to try).

    Each strategy also tries the RHS-only fallback: if the model writes
    "lhs = rhs" but the ground truth is a bare expression, the RHS is
    compared separately.
    """

    def __init__(
        self,
        variable_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        if variable_mapping is not None and not isinstance(variable_mapping, dict):
            raise ValueError("variable_mapping must be a dictionary or None")

        self.variable_mapping = variable_mapping or {}

        # Pre-build the reverse mapping for strategy 3.
        self._reverse_mapping: Dict[str, str] = {
            v: k for k, v in self.variable_mapping.items()
        }

        # Pre-declare every known variable name (both display and concrete) as
        # a SymPy Symbol so that parse_expr never misinterprets them.
        # Without this local_dict:
        #   - "concentration" is split into c*o*n*c*... by implicit_multiplication
        #   - x0 becomes x*0 = 0, Y2 becomes Y*2, I becomes ImaginaryUnit
        all_names = (
            set(self.variable_mapping.keys()) | set(self.variable_mapping.values())
        )
        self._known_symbols: Dict[str, sp.Symbol] = {
            name: sp.Symbol(name) for name in all_names
        }

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def match(self, predicted: str, ground_truth: str) -> bool:
        """
        Return True if the predicted equation is algebraically equivalent
        to the ground-truth equation under any of the supported strategies.
        """
        if not isinstance(predicted, str) or not predicted.strip():
            return False
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            return False

        # Build the ordered list of mappings to try.
        # Strategy 1 is only meaningful when there is a variable_mapping.
        mappings_to_try: List[Dict[str, str]] = []
        if self.variable_mapping:
            mappings_to_try.append(self.variable_mapping)   # strategy 1
        mappings_to_try.append({})                           # strategy 2: no mapping
        if self._reverse_mapping and self._reverse_mapping != self.variable_mapping:
            mappings_to_try.append(self._reverse_mapping)   # strategy 3

        for mapping in mappings_to_try:
            try:
                if self._match_with_mapping(predicted, ground_truth, mapping):
                    return True
            except Exception:
                continue  # this strategy failed; try the next one

        return False

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _match_with_mapping(
        self,
        predicted: str,
        ground_truth: str,
        mapping: Dict[str, str],
    ) -> bool:
        """
        Try a single mapping strategy.

        Strategy (in order):
        1. Normalize surface formatting (Unicode Greek letters, LaTeX, etc.)
           so that e.g. the Unicode character ε becomes the ASCII string
           epsilon before any further processing.
        2. Parse both equations into SymPy expressions using self._known_symbols
           as a local_dict.  Passing the local_dict is critical for two reasons:
           a) It prevents SymPy from misinterpreting names like x0 (→ x*0=0),
              Y2 (→ Y*2), or I (→ ImaginaryUnit).
           b) implicit_multiplication_application can split concatenated single-
              letter variables correctly: "cl" → c*l when both c and l are in
              the local_dict.  This handles model outputs like A = εcl.
        3. Apply the variable mapping as a symbolic substitution on the parsed
           SymPy expression rather than as a string-level replacement.  This
           avoids false positives such as mapping c → concentration inside the
           word "concentration" that was already placed by a prior substitution.
        4. Also build case-insensitive substitutions for short variable names so
           that C and c are treated as the same variable.
        5. RHS-only fallback for equations where the model includes a lhs.
        """
        # 1. Normalize surface formatting
        predicted_pre  = self._preprocess(predicted)
        gt_pre         = self._preprocess(ground_truth)

        # 2. Parse with known_symbols as local_dict
        pred_expr = self._equation_to_expr(predicted_pre)
        gt_expr   = self._equation_to_expr(gt_pre)

        if pred_expr is None or gt_expr is None:
            return False

        # 3 & 4. Build symbolic substitution map (display → concrete).
        #    Add case-insensitive variants for short names so that C matches c.
        subs: Dict[sp.Symbol, sp.Symbol] = {}
        for display, concrete in mapping.items():
            subs[sp.Symbol(display)] = sp.Symbol(concrete)
            if len(display) <= 3:
                subs[sp.Symbol(display.upper())] = sp.Symbol(concrete)
                subs[sp.Symbol(display.lower())] = sp.Symbol(concrete)

        pred_expr = pred_expr.subs(subs)

        if self._expressions_equivalent(pred_expr, gt_expr):
            return True

        # 5. RHS-only fallback
        predicted_rhs = self._extract_rhs_if_equation(predicted_pre)
        if predicted_rhs is not None and "=" not in gt_pre:
            rhs_expr = self._equation_to_expr(predicted_rhs)
            if rhs_expr is not None:
                rhs_expr = rhs_expr.subs(subs)
                return self._expressions_equivalent(rhs_expr, gt_expr)

        return False

    @staticmethod
    def _apply_mapping(text: str, mapping: Dict[str, str]) -> str:
        """
        Replace variable names in text according to mapping.

        Keys are applied longest-first to prevent a short key (e.g. "A")
        from partially replacing a longer one (e.g. "AB").
        Word boundaries prevent substring replacement inside longer identifiers.
        """
        if not mapping:
            return text

        mapped = text
        for source, target in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            pattern = rf"\b{re.escape(source)}\b"
            mapped = re.sub(pattern, target, mapped, flags=re.IGNORECASE)
        return mapped

    @staticmethod
    def _preprocess(equation: str) -> str:
        """
        Normalize surface-level formatting before parsing.

        Transformations applied (in order):
        1. Unicode Greek letters → ASCII names (e.g. ε → epsilon, ρ → rho).
           Models sometimes output the actual Unicode character instead of the
           spelled-out name used in variable_mapping, causing mapping to fail.
        2. LaTeX multiplication symbols (\times, \cdot) → *
        3. LaTeX fraction \frac{a}{b} → (a)/(b)
        4. LaTeX parentheses (\left(, \right)) → (, )
        5. Escaped asterisk \* → *
        6. Caret exponent ^ → **
        """
        # 1. Unicode Greek letters → ASCII names
        _GREEK = {
            "α": "alpha",   "β": "beta",    "γ": "gamma",   "δ": "delta",
            "ε": "epsilon", "ζ": "zeta",    "η": "eta",     "θ": "theta",
            "λ": "lambda",  "μ": "mu",      "ν": "nu",      "ξ": "xi",
            "π": "pi",      "ρ": "rho",     "σ": "sigma",   "τ": "tau",
            "φ": "phi",     "χ": "chi",     "ψ": "psi",     "ω": "omega",
            "Γ": "Gamma",   "Δ": "Delta",   "Θ": "Theta",   "Λ": "Lambda",
            "Ξ": "Xi",      "Π": "Pi",      "Σ": "Sigma",   "Φ": "Phi",
            "Ψ": "Psi",     "Ω": "Omega",
        }
        for unicode_char, ascii_name in _GREEK.items():
            # Add spaces around the replacement so that implicit multiplication
            # is preserved: e.g. εcl -> " epsilon cl" -> epsilon * c * l.
            equation = equation.replace(unicode_char, f" {ascii_name} ")

        # 2. LaTeX multiplication  (match literal backslash + command)
        equation = re.sub(r"\\times", "*", equation)
        equation = re.sub(r"\\cdot", "*", equation)

        # 3. LaTeX fraction
        equation = re.sub(r"\\frac\{(.*?)\}\{(.*?)\}", r"(\1)/(\2)", equation)

        # 4. LaTeX parentheses
        equation = re.sub(r"\\left\(", "(", equation)
        equation = re.sub(r"\\right\)", ")", equation)
        equation = re.sub(r"\\\(", "(", equation)
        equation = re.sub(r"\\\)", ")", equation)

        # 5. Escaped asterisk
        equation = equation.replace("\\*", "*")

        # 6. Caret exponent
        equation = equation.replace("^", "**")

        return equation

    def _equation_to_expr(self, equation: str) -> Optional[sp.Expr]:
        """
        Parse an equation string into a single SymPy expression.

        "lhs = rhs"  →  parse(lhs) - parse(rhs)
        "expr"       →  parse(expr)

        self._known_symbols is passed as local_dict so that:
        - Multi-letter concrete names like "concentration" are treated as a
          single Symbol and not split into implicit products by
          implicit_multiplication_application (e.g. c*o*n*c*...).
        - Short names like x0, Y2, I keep their intended meaning instead of
          being misinterpreted as x*0, Y*2, or ImaginaryUnit.
        """
        equation = equation.strip()

        if "=" in equation:
            lhs_str, rhs_str = equation.split("=", 1)
            lhs_str = lhs_str.strip()
            rhs_str = rhs_str.strip()
            if not lhs_str or not rhs_str:
                return None
            lhs = parse_expr(lhs_str, transformations=_TRANSFORMATIONS,
                             local_dict=self._known_symbols)
            rhs = parse_expr(rhs_str, transformations=_TRANSFORMATIONS,
                             local_dict=self._known_symbols)
            return lhs - rhs

        if not equation:
            return None
        return parse_expr(equation, transformations=_TRANSFORMATIONS,
                         local_dict=self._known_symbols)

    @staticmethod
    def _extract_rhs_if_equation(equation: str) -> Optional[str]:
        if "=" not in equation:
            return None
        _, rhs_str = equation.split("=", 1)
        rhs_str = rhs_str.strip()
        return rhs_str if rhs_str else None

    @staticmethod
    def _expressions_equivalent(expr1: sp.Expr, expr2: sp.Expr) -> bool:
        """
        Check algebraic equivalence.  Tests both (e1 - e2) and (e1 + e2)
        to handle side-swapped equations (which produce a negation).
        """
        if sp.simplify(expr1 - expr2) == 0:
            return True
        if sp.simplify(expr1 + expr2) == 0:
            return True
        return False