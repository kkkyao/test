"""
Minimal tests for parse error handling and _repair() logic.

Run with:
    cd /home/lly/projects/project
    python -m pytest tests/test_parse_repair.py -v
or (no pytest):
    python tests/test_parse_repair.py
"""
from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.agent import ParseError, TextLLMAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(raw: str):
    """Convenience wrapper that runs the full _parse_output pipeline."""
    return TextLLMAgent._parse_output(raw, strip_markdown_fences=True)


def repair_and_loads(raw: str) -> dict:
    """Run _repair then json.loads and return the dict."""
    repaired = TextLLMAgent._repair(raw)
    return json.loads(repaired)


# ---------------------------------------------------------------------------
# Test 1: normal valid JSON parses without error
# ---------------------------------------------------------------------------

def test_normal_action_json():
    raw = json.dumps({
        "reasoning": "I will increase concentration.",
        "step_type": "action",
        "action": {"action_type": "increase", "variable": "concentration"},
        "final_equation": None,
    })
    step = parse(raw)
    assert step.step_type == "action"
    assert step.action.action_type == "increase"
    assert step.action.variable == "concentration"
    print("PASS test_normal_action_json")


def test_normal_finish_json():
    raw = json.dumps({
        "reasoning": "Done exploring.",
        "step_type": "finish",
        "action": None,
        "final_equation": "A = epsilon * c * l",
    })
    step = parse(raw)
    assert step.step_type == "finish"
    assert step.final_equation == "A = epsilon * c * l"
    print("PASS test_normal_finish_json")


# ---------------------------------------------------------------------------
# Test 2: LaTeX \epsilon repaired and parsed successfully
# ---------------------------------------------------------------------------

def test_latex_epsilon_repaired():
    """
    Model writes "final_equation": "A = \epsilon * c * l"
    \e is an invalid JSON escape; _repair must double the backslash.
    After repair the JSON is valid and final_equation contains the raw string.
    """
    # Simulate exactly what the model outputs (raw bytes, not Python-escaped)
    raw = '{\n  "reasoning": "Beer-Lambert law.",\n  "step_type": "finish",\n  "action": null,\n  "final_equation": "A = \\epsilon * c * l"\n}'
    # json.loads on raw should FAIL (invalid \escape)
    try:
        json.loads(raw)
        assert False, "Expected JSONDecodeError before repair"
    except json.JSONDecodeError:
        pass

    # After _repair it should succeed
    repaired = TextLLMAgent._repair(raw)
    data = json.loads(repaired)
    assert data["final_equation"].endswith("epsilon * c * l")

    # Full _parse_output pipeline should also succeed
    step = parse(raw)
    assert step.step_type == "finish"
    assert "epsilon" in step.final_equation
    print("PASS test_latex_epsilon_repaired")


def test_latex_frac_repaired():
    """
    Model writes "final_equation": "A = \frac{x}{y}" — \f is a valid JSON
    escape (form feed), but \fr... makes the equation garbage.  Actually the
    issue manifests when \frac is followed by invalid chars.  Regardless,
    _repair should not break valid JSON, and should escape the problematic
    backslash from the LaTeX perspective.
    """
    # Use \cdot which has \c — definitely invalid JSON escape
    raw = '{"reasoning": "cdot test.", "step_type": "finish", "action": null, "final_equation": "A = c \\cdot l"}'
    repaired = TextLLMAgent._repair(raw)
    data = json.loads(repaired)
    # The value now has a literal backslash before "cdot"
    assert "cdot" in data["final_equation"]
    print("PASS test_latex_frac_repaired")


# ---------------------------------------------------------------------------
# Test 3: _repair preserves valid escapes untouched
# ---------------------------------------------------------------------------

def test_repair_preserves_valid_escapes():
    """Valid JSON escapes (\n, \\, \", \t) must not be double-escaped."""
    raw = '{"reasoning": "line1\\nline2", "step_type": "action", "action": {"action_type": "increase", "variable": "x"}, "final_equation": null}'
    # Should parse fine before and after repair
    data_before = json.loads(raw)
    repaired = TextLLMAgent._repair(raw)
    data_after = json.loads(repaired)
    assert data_before["reasoning"] == data_after["reasoning"] == "line1\nline2"
    print("PASS test_repair_preserves_valid_escapes")


# ---------------------------------------------------------------------------
# Test 4: missing comma still fails — but ParseError carries full raw_output
# ---------------------------------------------------------------------------

def test_missing_comma_fails_with_full_raw_output():
    """
    JSON with missing comma between fields cannot be repaired.
    _parse_output raises ParseError, and the exception carries the complete
    raw model output (not a 300-char truncation).
    """
    # Simulate EP1: missing comma after "reasoning" value
    raw = (
        '{\n'
        '  "reasoning": "I notice the orange bar is tallest."\n'   # <- no comma here
        '  "step_type": "action",\n'
        '  "action": {"action_type": "increase", "variable": "epsilon"},\n'
        '  "final_equation": null\n'
        '}'
    )
    try:
        parse(raw)
        assert False, "Expected ParseError for missing-comma JSON"
    except ParseError as exc:
        # The full raw output must be preserved
        assert exc.raw_output == raw, (
            f"raw_output truncated or missing. "
            f"Got {len(exc.raw_output)} chars, expected {len(raw)}"
        )
        assert "json_decode" in str(exc).lower() or "json" in str(exc).lower()
        # cleaned_output should also be non-empty
        assert exc.cleaned_output  # brace-walk should extract a candidate
        print(f"PASS test_missing_comma_fails_with_full_raw_output  "
              f"(raw_output len={len(exc.raw_output)}, "
              f"error={str(exc)[:80]})")
    except Exception as exc:
        assert False, f"Expected ParseError, got {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Test 5: literal newline inside string value is repaired
# ---------------------------------------------------------------------------

def test_literal_newline_in_string_repaired():
    """Literal newlines inside JSON string values are escaped by _repair."""
    reasoning_with_newline = "line1\nline2"
    # Build raw text with a literal newline inside the JSON string (not escaped)
    raw = (
        '{"reasoning": "line1\nline2", "step_type": "action",'
        ' "action": {"action_type": "increase", "variable": "x"},'
        ' "final_equation": null}'
    )
    # json.loads should fail on unescaped newline
    try:
        json.loads(raw)
        assert False, "Expected JSONDecodeError for literal newline"
    except json.JSONDecodeError:
        pass

    step = parse(raw)
    assert "line1" in step.reasoning and "line2" in step.reasoning
    print("PASS test_literal_newline_in_string_repaired")


# ---------------------------------------------------------------------------
# Test 6: parse_error_attempts in run_episode result (integration smoke test)
# ---------------------------------------------------------------------------

def test_parse_error_attempts_captured_in_result():
    """
    When the agent always fails to parse, run_episode should record the full
    raw output in parse_error_attempts (not an empty string).
    """
    import types
    from src.runners.runner import EpisodeRunner
    from src.envs.env import EquationEnv
    from src.observation.renderer import TextRenderer
    from src.prompts.prompt_builder import PromptBuilder

    # Minimal env
    variables = {
        "x": {"manipulable": True, "initial_value": 1.0, "step_size": 1.0,
              "min_value": 0.0, "max_value": 10.0},
        "y": {"initial_value": 1.0},
    }
    equations = {"y": "x * 2"}
    env = EquationEnv(variables=variables, equations=equations,
                      action_mode="increase_decrease")

    renderer = TextRenderer(
        variables=variables,
        action_mode="increase_decrease",
        target_variable="y",
        naming_mode="concrete",
        metadata_level="minimal",
    )

    prompt_cfg = {
        "system_intro": "You are a scientific explorer.",
        "task_template": "Find the equation for {target_variable}.",
        "exploration_lines": ["You may take up to {max_steps} steps."],
        "step_type_descriptions": {"action": "take an action", "finish": "finish"},
        "rules": ["Return exactly one valid JSON object."],
        "output_format_header": "Output format:",
        "output_format_action_increase_decrease": '{"step_type":"action","reasoning":"r","action":{"action_type":"increase","variable":"x"},"final_equation":null}',
        "output_format_finish": '{"step_type":"finish","reasoning":"r","action":null,"final_equation":"y=x"}',
        "section_headers": {
            "step_types": "Steps:", "rules": "Rules:",
            "observation": "Obs:", "history": "History:",
            "output_format": "Format:",
        },
        "history_labels": {"empty": "none", "step_prefix": "Step {step_id}:", "step_type": "{value}"},
        "forced_finish_template": "Write the equation for {target_variable}:\n",
    }

    prompt_builder = PromptBuilder(
        prompt_config=prompt_cfg,
        target_variable="y",
        max_steps=3,
        action_mode="increase_decrease",
        equation_variables=["x"],
    )

    BAD_RAW = "THIS IS NOT JSON AT ALL — full raw output preserved here"

    class AlwaysFailAgent:
        def act(self, prompt, image_paths=None):
            raise ParseError(
                "[json_decode] not json",
                raw_output=BAD_RAW,
                cleaned_output="",
            )

    runner = EpisodeRunner(
        env=env,
        renderer=renderer,
        prompt_builder=prompt_builder,
        agent=AlwaysFailAgent(),
        max_steps=3,
        max_parse_retries=0,
        task_mode="formula_discovery",
    )

    result = runner.run_episode()

    attempts = result.get("parse_error_attempts", [])
    assert len(attempts) == 1, f"Expected 1 failed attempt, got {len(attempts)}"
    assert attempts[0]["raw_model_output"] == BAD_RAW, (
        f"raw_model_output was truncated or missing. "
        f"Got: {attempts[0]['raw_model_output']!r}"
    )
    assert attempts[0]["step_id"] == 0
    assert result["parse_error"] is not None

    print(f"PASS test_parse_error_attempts_captured_in_result  "
          f"(raw_output len={len(attempts[0]['raw_model_output'])})")


# ---------------------------------------------------------------------------
# Test 7: logger writes parse_errors.jsonl with full content
# ---------------------------------------------------------------------------

def test_logger_writes_parse_errors_jsonl(tmp_path=None):
    import tempfile
    from pathlib import Path
    from src.tracing.logger import EpisodeLogger

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())

    logger = EpisodeLogger(output_dir=str(tmp_path))

    BAD_RAW = "X" * 2000  # 2000 chars — well beyond the old 300-char truncation

    fake_result = {
        "steps": [],
        "trajectory": [],
        "final_equation": None,
        "finish_reason": None,
        "finish_reached": False,
        "finish_step_id": None,
        "num_steps": 1,
        "parse_error": "[json_decode] not json",
        "forced_finish": False,
        "image_paths": [],
        "parse_error_attempts": [
            {
                "step_id": 0,
                "attempt_index": 0,
                "prompt_hash": "abc123",
                "image_paths": [],
                "raw_model_output": BAD_RAW,
                "cleaned_model_output": "",
                "parse_error_message": "[json_decode] not json",
            }
        ],
    }

    saved = logger.save_episode(fake_result)
    assert "parse_errors" in saved, "parse_errors.jsonl should be in saved_paths"

    pe_path = Path(saved["parse_errors"])
    assert pe_path.exists()

    lines = pe_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    row = json.loads(lines[0])
    assert row["raw_model_output"] == BAD_RAW, (
        f"Expected {len(BAD_RAW)} chars, got {len(row['raw_model_output'])}"
    )
    assert row["step_id"] == 0

    print(f"PASS test_logger_writes_parse_errors_jsonl  "
          f"(raw_model_output len={len(row['raw_model_output'])}, file={pe_path})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_normal_action_json()
    test_normal_finish_json()
    test_latex_epsilon_repaired()
    test_latex_frac_repaired()
    test_repair_preserves_valid_escapes()
    test_missing_comma_fails_with_full_raw_output()
    test_literal_newline_in_string_repaired()
    test_parse_error_attempts_captured_in_result()
    test_logger_writes_parse_errors_jsonl()
    print("\nAll tests passed.")
