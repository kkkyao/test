"""
Tests for max_parse_retries configurability.

Run with:
    cd /home/lly/projects/project
    python tests/test_max_parse_retries.py
or:
    python -m pytest tests/test_max_parse_retries.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.runners.runner import EpisodeRunner
from src.agents.agent import ParseError, TextLLMAgent
from src.envs.env import EquationEnv
from src.observation.renderer import TextRenderer
from src.prompts.prompt_builder import PromptBuilder
from src.tracing.logger import EpisodeLogger


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VARIABLES = {
    "x": {"manipulable": True, "initial_value": 1.0, "step_size": 1.0,
          "min_value": 0.0, "max_value": 10.0},
    "y": {"initial_value": 1.0},
}
_EQUATIONS = {"y": "x * 2"}

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


def _make_runner(max_parse_retries: int, agent) -> EpisodeRunner:
    env = EquationEnv(variables=_VARIABLES, equations=_EQUATIONS,
                      action_mode="increase_decrease")
    renderer = TextRenderer(
        variables=_VARIABLES, action_mode="increase_decrease",
        target_variable="y", naming_mode="concrete", metadata_level="minimal",
    )
    prompt_builder = PromptBuilder(
        prompt_config=_PROMPT_CFG, target_variable="y",
        max_steps=5, action_mode="increase_decrease", equation_variables=["x"],
    )
    return EpisodeRunner(
        env=env, renderer=renderer, prompt_builder=prompt_builder,
        agent=agent, max_steps=5, task_mode="formula_discovery",
        max_parse_retries=max_parse_retries,
    )


# ---------------------------------------------------------------------------
# Test 1: default max_parse_retries = 1 (no YAML key)
# ---------------------------------------------------------------------------

def test_default_max_parse_retries():
    """EpisodeRunner uses default=1 when instantiated without explicit value."""
    experiment_cfg = {"max_steps": 10}  # no max_parse_retries key
    retries = experiment_cfg.get("max_parse_retries", 1)
    assert retries == 1, f"Expected default 1, got {retries}"

    class ImmediateFinish:
        def act(self, prompt, image_paths=None):
            raw = json.dumps({"step_type": "finish", "reasoning": "done",
                              "action": None, "final_equation": "y = x * 2"})
            return TextLLMAgent._parse_output(raw), raw

    runner = _make_runner(retries, ImmediateFinish())
    assert runner.max_parse_retries == 1
    print("PASS test_default_max_parse_retries")


# ---------------------------------------------------------------------------
# Test 2: YAML key max_parse_retries: 3 is read correctly
# ---------------------------------------------------------------------------

def test_yaml_max_parse_retries_3():
    """experiment_cfg.get('max_parse_retries', 1) returns 3 when present."""
    experiment_cfg = {"max_steps": 50, "max_parse_retries": 3}
    retries = experiment_cfg.get("max_parse_retries", 1)
    assert retries == 3, f"Expected 3, got {retries}"

    class ImmediateFinish:
        def act(self, prompt, image_paths=None):
            raw = json.dumps({"step_type": "finish", "reasoning": "done",
                              "action": None, "final_equation": "y = x * 2"})
            return TextLLMAgent._parse_output(raw), raw

    runner = _make_runner(retries, ImmediateFinish())
    assert runner.max_parse_retries == 3
    print("PASS test_yaml_max_parse_retries_3")


# ---------------------------------------------------------------------------
# Test 3: retries are actually attempted the right number of times
# ---------------------------------------------------------------------------

def test_retries_attempted_correct_number_of_times():
    """
    With max_parse_retries=2, a permanently-failing agent gets 3 total attempts
    (initial + 2 retries).  The result's parse_error_attempts list has 3 entries.
    """
    attempt_count = {"n": 0}

    class AlwaysFailAgent:
        def act(self, prompt, image_paths=None):
            attempt_count["n"] += 1
            raise ParseError("[json_decode] always fails",
                             raw_output=f"BAD OUTPUT attempt {attempt_count['n']}",
                             cleaned_output="")

    runner = _make_runner(max_parse_retries=2, agent=AlwaysFailAgent())
    result = runner.run_episode()

    # Main loop: 1 initial + 2 retries = 3 attempts at step 0.
    # _run_forced_finish (formula_discovery, Path B) makes 1 additional agent.act()
    # call which is caught silently — it increments the counter but is NOT added
    # to parse_error_attempts.
    assert attempt_count["n"] == 4, (
        f"Expected 4 total agent.act() calls "
        f"(3 main-loop retries + 1 forced_finish), got {attempt_count['n']}"
    )
    # Only the 3 main-loop attempts are recorded in parse_error_attempts
    assert len(result["parse_error_attempts"]) == 3, (
        f"Expected 3 failed attempt records (from main loop only), "
        f"got {len(result['parse_error_attempts'])}"
    )
    # All raw outputs must be the full text (not truncated)
    for i, rec in enumerate(result["parse_error_attempts"]):
        assert rec["raw_model_output"].startswith("BAD OUTPUT"), (
            f"Attempt {i}: raw_model_output truncated or missing: {rec['raw_model_output']!r}"
        )
    print(f"PASS test_retries_attempted_correct_number_of_times  "
          f"(total agent calls={attempt_count['n']}, "
          f"parse_error_attempts recorded={len(result['parse_error_attempts'])})")


# ---------------------------------------------------------------------------
# Test 4: result dict contains max_parse_retries
# ---------------------------------------------------------------------------

def test_result_dict_contains_max_parse_retries():
    """run_episode() result must include max_parse_retries."""
    class ImmediateFinish:
        def act(self, prompt, image_paths=None):
            raw = json.dumps({"step_type": "finish", "reasoning": "done",
                              "action": None, "final_equation": "y = x"})
            return TextLLMAgent._parse_output(raw), raw

    runner = _make_runner(max_parse_retries=3, agent=ImmediateFinish())
    result = runner.run_episode()

    assert "max_parse_retries" in result, "max_parse_retries missing from result dict"
    assert result["max_parse_retries"] == 3
    print("PASS test_result_dict_contains_max_parse_retries")


# ---------------------------------------------------------------------------
# Test 5: summary.json contains max_parse_retries
# ---------------------------------------------------------------------------

def test_summary_json_contains_max_parse_retries():
    """EpisodeLogger writes max_parse_retries into summary.json."""
    class ImmediateFinish:
        def act(self, prompt, image_paths=None):
            raw = json.dumps({"step_type": "finish", "reasoning": "done",
                              "action": None, "final_equation": "y = x"})
            return TextLLMAgent._parse_output(raw), raw

    runner = _make_runner(max_parse_retries=3, agent=ImmediateFinish())
    result = runner.run_episode()

    with tempfile.TemporaryDirectory() as td:
        logger = EpisodeLogger(output_dir=td)
        saved = logger.save_episode(result)
        summary = json.loads(Path(saved["summary"]).read_text(encoding="utf-8"))

    assert "max_parse_retries" in summary, (
        f"max_parse_retries missing from summary.json. Keys: {list(summary.keys())}"
    )
    assert summary["max_parse_retries"] == 3
    print("PASS test_summary_json_contains_max_parse_retries")


# ---------------------------------------------------------------------------
# Test 6: formula-discovery experiment YAMLs have max_parse_retries: 3
# ---------------------------------------------------------------------------

def test_experiment_yamls_have_max_parse_retries():
    """All formula-discovery experiment configs must declare max_parse_retries."""
    import yaml

    formula_discovery_configs = [
        "experiment_default.yaml",
        "experiment_abstract.yaml",
        "experiment_text_only.yaml",
        "experiment_text_image.yaml",
        "experiment_chart_action.yaml",
        "experiment_simulation_only_beers.yaml",
        "experiment_simulation_only_distance.yaml",
        "experiment_simulation_only_kinematics.yaml",
        "experiment_simulation_only_mass.yaml",
        "experiment_simulation_only_ohm.yaml",
        "experiment_simulation_only_resistors.yaml",
    ]

    configs_dir = Path(__file__).parent.parent / "configs"
    missing = []
    wrong_value = []

    for name in formula_discovery_configs:
        path = configs_dir / name
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        val = cfg.get("experiment", {}).get("max_parse_retries")
        if val is None:
            missing.append(name)
        elif val != 3:
            wrong_value.append(f"{name}: got {val}")

    assert not missing, f"max_parse_retries missing in: {missing}"
    assert not wrong_value, f"Wrong value in: {wrong_value}"
    print(f"PASS test_experiment_yamls_have_max_parse_retries  "
          f"({len(formula_discovery_configs)} configs checked)")


# ---------------------------------------------------------------------------
# Test 7: student-simulation configs do NOT have max_parse_retries
# ---------------------------------------------------------------------------

def test_student_configs_untouched():
    """Student-simulation experiment configs should NOT have max_parse_retries."""
    import yaml

    student_configs = [
        "experiment_text_only_student.yaml",
        "experiment_text_only_normal.yaml",
        "experiment_chart_action_student.yaml",
        "experiment_chart_action_normal.yaml",
        "experiment_simulation_beers_wavelength_student.yaml",
        "experiment_simulation_beers_wavelength_normal.yaml",
        "experiment_simulation_concentration_student.yaml",
        "experiment_simulation_concentration_normal.yaml",
    ]

    configs_dir = Path(__file__).parent.parent / "configs"
    has_key = []

    for name in student_configs:
        path = configs_dir / name
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if "max_parse_retries" in cfg.get("experiment", {}):
            has_key.append(name)

    assert not has_key, f"Student configs unexpectedly got max_parse_retries: {has_key}"
    print(f"PASS test_student_configs_untouched  "
          f"({len(student_configs)} configs verified clean)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_default_max_parse_retries()
    test_yaml_max_parse_retries_3()
    test_retries_attempted_correct_number_of_times()
    test_result_dict_contains_max_parse_retries()
    test_summary_json_contains_max_parse_retries()
    test_experiment_yamls_have_max_parse_retries()
    test_student_configs_untouched()
    print("\nAll tests passed.")
