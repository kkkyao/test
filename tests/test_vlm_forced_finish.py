"""
Tests for _run_forced_finish fix that aligns VLM behaviour with Text-only.

Bug fixed: VisionLanguageAgent._generate(prompt, image_paths) was called as
_generate(prompt) (missing image_paths), raising TypeError that was swallowed
by a broad except-pass, leaving VLM episodes stuck at termination_reason="parse_error".

Run with:
    cd /home/lly/projects/project
    python tests/test_vlm_forced_finish.py
"""
from __future__ import annotations

import json
import sys
import os
import tempfile
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.agent import ParseError, TextLLMAgent
from src.envs.env import EquationEnv
from src.observation.renderer import TextRenderer
from src.prompts.prompt_builder import PromptBuilder
from src.runners.runner import EpisodeRunner
from src.tracing.logger import EpisodeLogger


# ── Shared fixtures ──────────────────────────────────────────────────────────

_VARS = {
    "x": {"manipulable": True, "initial_value": 1.0, "step_size": 1.0,
          "min_value": 0.0, "max_value": 10.0},
    "y": {"initial_value": 1.0},
}
_EQS = {"y": "x * 2"}

_PROMPT_CFG = {
    "system_intro": "You are a scientific explorer.",
    "task_template": "Find the equation for {target_variable}.",
    "exploration_lines": ["You may take up to {max_steps} steps."],
    "step_type_descriptions": {"action": "take an action", "finish": "finish"},
    "rules": ["Return exactly one valid JSON object."],
    "output_format_header": "Output format:",
    "output_format_action_increase_decrease": (
        '{"step_type":"action","reasoning":"r",'
        '"action":{"action_type":"increase","variable":"x"},"final_equation":null}'
    ),
    "output_format_finish": (
        '{"step_type":"finish","reasoning":"r","action":null,"final_equation":"y=x"}'
    ),
    "section_headers": {
        "step_types": "Steps:", "rules": "Rules:", "observation": "Obs:",
        "history": "History:", "output_format": "Format:",
    },
    "history_labels": {
        "empty": "none", "step_prefix": "Step {step_id}:", "step_type": "{value}",
    },
    "forced_finish_template": "Write the equation for {target_variable}:\n",
}


def _make_runner(agent, max_parse_retries=0, max_steps=5) -> EpisodeRunner:
    env = EquationEnv(variables=_VARS, equations=_EQS,
                      action_mode="increase_decrease")
    renderer = TextRenderer(
        variables=_VARS, action_mode="increase_decrease",
        target_variable="y", naming_mode="concrete", metadata_level="minimal",
    )
    prompt_builder = PromptBuilder(
        prompt_config=_PROMPT_CFG, target_variable="y",
        max_steps=max_steps, action_mode="increase_decrease",
        equation_variables=["x"],
    )
    return EpisodeRunner(
        env=env, renderer=renderer, prompt_builder=prompt_builder,
        agent=agent, max_steps=max_steps, task_mode="formula_discovery",
        max_parse_retries=max_parse_retries,
    )


# ── Mock agents ──────────────────────────────────────────────────────────────

class ParseErrorThenSilentTextAgent:
    """Always raises ParseError; also has _generate() for forced_finish."""

    def __init__(self, equation: str):
        self.equation = equation
        self.generate_calls: list = []

    def act(self, prompt, image_paths=None):
        raise ParseError("[json_decode] bad json", raw_output="BAD", cleaned_output="")

    def _generate(self, prompt: str) -> str:
        """Text-only style: single argument, returns equation as free text."""
        self.generate_calls.append({"prompt": prompt})
        return f"The final equation is {self.equation}"


class ParseErrorThenVLMAgent:
    """
    Simulates VisionLanguageAgent: has _generate(prompt, image_paths),
    always raises ParseError on act(), then returns a correct equation string
    from _generate.
    """

    def __init__(self, equation: str):
        self.equation = equation
        self.generate_calls: list = []

    def act(self, prompt, image_paths=None):
        raise ParseError("[json_decode] bad json", raw_output="BAD", cleaned_output="")

    def _generate(self, prompt: str, image_paths: List[str]) -> str:
        """VLM style: two arguments, records both for assertion."""
        self.generate_calls.append({"prompt": prompt, "image_paths": image_paths})
        return f"The equation is {self.equation}"


class ParseErrorThenVLMRaisesAgent:
    """VLM-style _generate that always raises — tests error recording."""

    def act(self, prompt, image_paths=None):
        raise ParseError("[json_decode] bad json", raw_output="BAD", cleaned_output="")

    def _generate(self, prompt: str, image_paths: List[str]) -> str:
        raise RuntimeError("GPU OOM during forced_finish generation")


class NormalFinishAgent:
    """Model that voluntarily finishes on step 0 — forced_finish should NOT run."""

    def act(self, prompt, image_paths=None):
        raw = json.dumps({"step_type": "finish", "reasoning": "done",
                          "action": None, "final_equation": "y = x * 2"})
        return TextLLMAgent._parse_output(raw), raw


# ── Test 1: Text-only forced_finish unchanged ─────────────────────────────────

def test_text_only_forced_finish_still_works():
    """
    Text-only agent: parse_error → forced_finish → _generate(prompt) → equation.
    termination_reason must NOT be 'parse_error'; must be finish_success or finish_wrong.
    """
    agent = ParseErrorThenSilentTextAgent(equation="y = x * 2")
    runner = _make_runner(agent, max_parse_retries=0)
    result = runner.run_episode()

    assert result["finish_reached"] is True, "forced_finish should set finish_reached"
    assert result["final_equation"] is not None, "equation must be extracted"
    assert result["forced_finish"] is True
    assert result["forced_finish_trigger"] == "parse_error"
    assert result["forced_finish_used_images"] is False  # text agent, no images
    assert result["forced_finish_error_type"] is None

    # _generate was called with just the prompt (text-only signature)
    assert len(agent.generate_calls) == 1
    assert "image_paths" not in agent.generate_calls[0], (
        "Text-only _generate must NOT receive image_paths"
    )

    # parse_error diagnostic is preserved
    assert result["parse_error"] is not None

    print(f"PASS test_text_only_forced_finish_still_works  "
          f"eq={result['final_equation']!r}  trigger={result['forced_finish_trigger']}")


# ── Test 2: VLM _generate called with (prompt, image_paths) ──────────────────

def test_vlm_generate_called_with_image_paths():
    """
    VLM agent: _generate(prompt, image_paths) must be called correctly.
    Previously TypeError was silently swallowed, now it must succeed.
    """
    agent = ParseErrorThenVLMAgent(equation="y = x * 2")
    runner = _make_runner(agent, max_parse_retries=0)
    result = runner.run_episode()

    assert result["finish_reached"] is True, (
        f"VLM forced_finish should succeed. "
        f"forced_finish_error_type={result['forced_finish_error_type']}, "
        f"forced_finish_error_message={result['forced_finish_error_message']}"
    )
    assert result["final_equation"] is not None
    assert result["forced_finish"] is True
    assert result["forced_finish_trigger"] == "parse_error"

    # _generate was called with BOTH prompt and image_paths
    assert len(agent.generate_calls) == 1, (
        f"Expected 1 generate call, got {len(agent.generate_calls)}"
    )
    call = agent.generate_calls[0]
    assert "image_paths" in call, (
        "VLM _generate must receive image_paths as second argument"
    )
    assert isinstance(call["image_paths"], list), "image_paths must be a list"

    print(f"PASS test_vlm_generate_called_with_image_paths  "
          f"image_paths={call['image_paths']}  eq={result['final_equation']!r}")


# ── Test 3: VLM termination_reason is not parse_error after forced_finish ─────

def test_vlm_termination_reason_not_parse_error():
    """
    VLM: after forced_finish succeeds, evaluator must see finish_reached=True
    so termination_reason is finish_success or finish_wrong, NOT parse_error.
    We verify via the result fields (evaluator itself is tested elsewhere).
    """
    # Correct equation is y = x * 2; agent will return this too
    agent = ParseErrorThenVLMAgent(equation="y = x * 2")
    runner = _make_runner(agent, max_parse_retries=0)
    result = runner.run_episode()

    assert result["finish_reached"] is True
    # parse_error is logged for diagnostics but must not suppress finish_reached
    assert result["parse_error"] is not None, "parse_error diagnostic should be preserved"
    # final_equation is set, so evaluator can compare
    assert result["final_equation"] is not None

    # The evaluator logic: finish_reached=True takes priority over parse_error
    # (verified here via result fields; evaluator's _termination_reason is tested separately)
    print("PASS test_vlm_termination_reason_not_parse_error  "
          f"finish_reached={result['finish_reached']}  "
          f"final_equation={result['final_equation']!r}")


# ── Test 4: VLM forced_finish exception is recorded, not silently swallowed ───

def test_vlm_forced_finish_error_recorded():
    """
    When VLM _generate raises during forced_finish, the error must be recorded
    in the result dict instead of silently swallowed.  Episode must not crash.
    """
    agent = ParseErrorThenVLMRaisesAgent()
    runner = _make_runner(agent, max_parse_retries=0)
    result = runner.run_episode()

    # Forced_finish failed → finish_reached stays False
    assert result["finish_reached"] is False
    assert result["forced_finish"] is False
    assert result["forced_finish_error_type"] == "RuntimeError", (
        f"Expected 'RuntimeError', got {result['forced_finish_error_type']!r}"
    )
    assert "GPU OOM" in (result["forced_finish_error_message"] or ""), (
        f"Error message not recorded: {result['forced_finish_error_message']!r}"
    )
    assert result["forced_finish_trigger"] == "parse_error"

    print(f"PASS test_vlm_forced_finish_error_recorded  "
          f"error_type={result['forced_finish_error_type']}  "
          f"msg={result['forced_finish_error_message'][:40]!r}")


# ── Test 5: Normal voluntary finish does not trigger forced_finish ────────────

def test_normal_finish_not_affected():
    """
    When agent finishes voluntarily, forced_finish must NOT run.
    All new diagnostic fields should be None/False.
    """
    agent = NormalFinishAgent()
    runner = _make_runner(agent, max_parse_retries=0)
    result = runner.run_episode()

    assert result["finish_reached"] is True
    assert result["forced_finish"] is False
    assert result["forced_finish_trigger"] is None, (
        "forced_finish_trigger should be None when voluntary finish occurs"
    )
    assert result["forced_finish_used_images"] is False
    assert result["forced_finish_error_type"] is None
    assert result["forced_finish_error_message"] is None
    assert result["parse_error"] is None

    print("PASS test_normal_finish_not_affected  "
          f"final_equation={result['final_equation']!r}")


# ── Test 6: Logger writes diagnostic fields to summary.json ──────────────────

def test_logger_writes_forced_finish_diagnostics():
    """summary.json must include all forced_finish_* diagnostic fields."""
    agent = ParseErrorThenVLMAgent(equation="y = x * 3")  # wrong equation
    runner = _make_runner(agent, max_parse_retries=0)
    result = runner.run_episode()

    with tempfile.TemporaryDirectory() as td:
        logger = EpisodeLogger(output_dir=td)
        saved = logger.save_episode(result)
        summary = json.loads(Path(saved["summary"]).read_text(encoding="utf-8"))

    for field in (
        "forced_finish_trigger",
        "forced_finish_used_images",
        "forced_finish_error_type",
        "forced_finish_error_message",
    ):
        assert field in summary, f"Missing field {field!r} in summary.json"

    assert summary["forced_finish_trigger"] == "parse_error"
    assert summary["forced_finish_error_type"] is None  # succeeded, no error

    print(f"PASS test_logger_writes_forced_finish_diagnostics  "
          f"trigger={summary['forced_finish_trigger']}  "
          f"used_images={summary['forced_finish_used_images']}")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_text_only_forced_finish_still_works()
    test_vlm_generate_called_with_image_paths()
    test_vlm_termination_reason_not_parse_error()
    test_vlm_forced_finish_error_recorded()
    test_normal_finish_not_affected()
    test_logger_writes_forced_finish_diagnostics()
    print("\nAll tests passed.")
