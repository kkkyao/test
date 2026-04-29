# Scientific Exploration Framework

A config-driven, multi-step research framework for studying how language models explore scientific environments and discover underlying equations through iterative interaction.

---

## What This Is

This is **not** a question-answering system. It is an interactive, trajectory-oriented research pipeline where a model acts as an explorer inside a controlled scientific environment.

The model interacts with the environment step by step:
1. Receives an observation of the current state
2. Produces a JSON output (`action` or `finish`)
3. If `action`: environment updates, new observation is generated
4. If `finish`: model submits its final equation, episode ends

The primary research questions are:
- How do models explore scientific environments step by step?
- Does variable representation (concrete names vs. abstract symbols) affect exploration behavior?
- Does contextual metadata (rich descriptions vs. none) affect equation discovery?
- Do larger or newer models explore more efficiently?

The framework is fully **equation-driven**: any equation supplied via config automatically defines the environment, the action space, the ground truth, and the evaluation target. No code changes are needed to add a new environment.

---

## Project Structure

```
project/
  configs/
    config.yaml                    # Main entry point — points to sub-configs
    experiment_default.yaml        # Runtime parameters (max_steps, logging, etc.)
    prompt_default.yaml            # All prompt text (no hardcoded strings in code)

    # Environment configs (concrete = real names + descriptions, abstract = A/B/C/Y)
    env_beers_concrete.yaml        # Beer's Law: absorbance = ε * c * l
    env_beers_abstract.yaml
    env_resistors_concrete.yaml    # Series resistors: R_total = R1 + R2
    env_resistors_abstract.yaml
    env_ohm_concrete.yaml          # Ohm's Law: V = I * R
    env_ohm_abstract.yaml
    env_fma_concrete.yaml          # Newton's 2nd Law: F = m * a
    env_fma_abstract.yaml
    env_mass_concrete.yaml         # Mass: m = ρ * V * k
    env_mass_abstract.yaml
    env_distance_concrete.yaml     # Distance: d = d1 + d2 + d3
    env_distance_abstract.yaml
    env_kinematics_concrete.yaml   # Kinematics: x = x0 + v * t
    env_kinematics_abstract.yaml

    # Model configs
    model_qwen25_3b.yaml           # Qwen2.5-3B-Instruct
    model_qwen25_7b.yaml           # Qwen2.5-7B-Instruct
    model_qwen35_4b.yaml           # Qwen3.5-4B (disable_thinking: true)
    model_qwen35_9b.yaml           # Qwen3.5-9B (disable_thinking: true)
    model_llama31_8b.yaml          # Meta Llama 3.1 8B Instruct
    model_mistral_7b.yaml          # Mistral 7B Instruct v0.3
    model_gemma3_4b.yaml           # Google Gemma 3 4B Instruct

  src/
    schemas/
      action_schema.py             # ActionSpec dataclass
      agent_output_schema.py       # AgentStep dataclass + StepType
      observation_schema.py        # Observation dataclass + ObservationMode
      trace_schema.py              # TraceStep dataclass (full trajectory record)

    envs/
      env.py                       # EquationEnv: state management + action execution
      equation_engine.py           # Safe AST-based equation parsing and evaluation

    observation/
      renderer.py                  # TextRenderer: state → Observation text

    prompts/
      prompt_builder.py            # PromptBuilder: observation + history → prompt

    agents/
      agent.py                     # TextLLMAgent: prompt → model → AgentStep

    runners/
      runner.py                    # EpisodeRunner: main episode loop

    tracing/
      logger.py                    # EpisodeLogger: saves all output files

    evaluation/
      evaluator.py                 # EpisodeEvaluator: computes all episode metrics
      equation_matcher.py          # EquationMatcher: algebraic equivalence via SymPy

    utils/
      config_loader.py             # YAML loading, merging, and validation

  run_episode.py                   # Entry point: single episode
  run_experiment.py                # Entry point: N runs with W&B logging
```

---

## How an Episode Works

```
env.reset()
    → initial state
    → renderer.render(state) → Observation

for step in range(max_steps):

    prompt_builder.build_prompt(observation, history) → prompt
    agent.act(prompt) → AgentStep, raw_output

    if step_type == "action":
        env.step(action)               ← only actions change the environment
        renderer.render(new_state)     → new observation

    elif step_type == "finish":
        record final_equation
        break                          ← episode ends

    record TraceStep                   ← every step is fully logged

EpisodeEvaluator.evaluate(result)
EpisodeLogger.save(result, evaluation)
```

---

## Model Output Format

The model must return a strict JSON object on every step. Two step types are supported:

**Action step (increase/decrease mode):**
```json
{
  "reasoning": "Concentration doubled and absorbance doubled. I want to test path_length next.",
  "step_type": "action",
  "action": {
    "action_type": "increase",
    "variable": "path_length"
  },
  "final_equation": null
}
```

**Action step (set mode):**
```json
{
  "reasoning": "I want to set concentration to a specific value to test proportionality.",
  "step_type": "action",
  "action": {
    "action_type": "set",
    "variable": "concentration",
    "value": 10
  },
  "final_equation": null
}
```

**Finish step:**
```json
{
  "reasoning": "All three variables show a proportional relationship with absorbance.",
  "step_type": "finish",
  "action": null,
  "final_equation": "absorbance = molar_absorptivity * concentration * path_length"
}
```

`reasoning` is placed first intentionally: because LLMs generate tokens left-to-right, writing `reasoning` before `step_type` forces the model to think before it decides what to do.

### Step types

| `step_type` | Required fields | Changes environment? |
|---|---|---|
| `action` | `reasoning`, `action` | **Yes** |
| `finish` | `reasoning`, `final_equation` | No — ends episode |

### Action types

| `action_type` | Mode | `value` field |
|---|---|---|
| `increase` | `increase_decrease` | Must not be present |
| `decrease` | `increase_decrease` | Must not be present |
| `set` | `set_value` | Required, must be numeric |

Action mode is fixed per episode (configured in the env YAML). The prompt only shows templates for the active mode. Format constraints are enforced by the parser — invalid outputs are rejected with a clear error message.

---

## Experimental Conditions

Two representational dimensions combine into conditions:

| Dimension | Options | Effect |
|---|---|---|
| `naming_mode` | `concrete` / `abstract` | Variable names shown to the model |
| `metadata_level` | `rich` / `minimal` / `none` | Whether descriptions are shown |

| Condition | Variable names | Descriptions |
|---|---|---|
| Concrete + Rich | `concentration`, `path_length`, ... | Full scientific descriptions |
| Abstract + None | `A`, `B`, `C`, `Y` | None |

Each environment has a `_concrete` and `_abstract` config file. Switching conditions requires only changing `--env_config`.

---

## Available Environments

| Config | Equation | Variables |
|---|---|---|
| `env_beers` | `absorbance = ε * c * l` | molar_absorptivity, concentration, path_length |
| `env_resistors` | `R_total = R1 + R2` | R1, R2 |
| `env_ohm` | `V = I * R` | current, resistance |
| `env_fma` | `F = m * a` | mass, acceleration |
| `env_mass` | `m = ρ * V * k` | density, volume, k |
| `env_distance` | `d = d1 + d2 + d3` | d1, d2, d3 |
| `env_kinematics` | `x = x0 + v * t` | initial_position, velocity, time |

All equations have no hardcoded numeric constants.

---

## Supported Models

All models use `backend: hf_qwen` and are loaded locally via HuggingFace Transformers.

| Config | Model | Params | Notes |
|---|---|---|---|
| `model_qwen25_3b.yaml` | `Qwen/Qwen2.5-3B-Instruct` | 3B | Baseline |
| `model_qwen25_7b.yaml` | `Qwen/Qwen2.5-7B-Instruct` | 7B | |
| `model_qwen35_4b.yaml` | `Qwen/Qwen3.5-4B` | 4B | `disable_thinking: true` required |
| `model_qwen35_9b.yaml` | `Qwen/Qwen3.5-9B` | 9B | `disable_thinking: true` required |
| `model_llama31_8b.yaml` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Requires HF access |
| `model_mistral_7b.yaml` | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | |
| `model_gemma3_4b.yaml` | `google/gemma-3-4b-it` | 4B | Requires HF access |

**Approximate VRAM requirements (bfloat16):**

| Model | VRAM |
|---|---|
| Qwen2.5-3B | ~7 GB |
| Qwen3.5-4B / Gemma3-4B | ~10 GB |
| Llama3.1-8B / Mistral-7B / Qwen2.5-7B | ~15 GB |
| Qwen3.5-9B | ~19 GB |

---

## Installation

```bash
pip install pyyaml sympy transformers torch wandb
```

For Qwen3.5 models, `transformers >= 4.51` is required:
```bash
pip install --upgrade transformers
```

---

## Running Experiments

### 1. Test the pipeline with mock model

```bash
python run_episode.py --config configs/config.yaml
```

Expected: 3 steps (2 actions + 1 finish), `finish_reached: True`, no `parse_error`.

### 2. Run a single episode with a real model

```bash
python run_episode.py \
  --config configs/config.yaml \
  --env_config configs/env_beers_concrete.yaml \
  --model_config configs/model_qwen25_7b.yaml
```

### 3. Run a full experiment (N runs + W&B logging)

```bash
wandb login

python run_experiment.py \
  --config configs/config.yaml \
  --env_config configs/env_beers_concrete.yaml \
  --model_config configs/model_qwen25_7b.yaml \
  --n_runs 30 \
  --run_name beers_concrete_qwen25_7b
```

### 4. Run multiple experiments overnight

Save the following as `run_all.sh`:

```bash
#!/bin/bash

COMMANDS=(
  "python run_experiment.py --config configs/config.yaml --env_config configs/env_beers_concrete.yaml --model_config configs/model_qwen25_7b.yaml --n_runs 30 --run_name beers_concrete_qwen25_7b"
  "python run_experiment.py --config configs/config.yaml --env_config configs/env_beers_abstract.yaml --model_config configs/model_qwen25_7b.yaml --n_runs 30 --run_name beers_abstract_qwen25_7b"
  # add more commands here...
)

LOG="run_all_$(date +%Y%m%d_%H%M%S).log"
echo "Started at $(date)" | tee "$LOG"

for CMD in "${COMMANDS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "========================================" | tee -a "$LOG"
  echo "Running: $CMD" | tee -a "$LOG"
  echo "Started: $(date)" | tee -a "$LOG"

  eval "$CMD" 2>&1 | tee -a "$LOG"
  EXIT_CODE=${PIPESTATUS[0]}

  if [ $EXIT_CODE -eq 0 ]; then
    echo "Finished OK: $(date)" | tee -a "$LOG"
  else
    echo "FAILED (exit $EXIT_CODE): $(date)" | tee -a "$LOG"
  fi
done

echo "All done at $(date)" | tee -a "$LOG"
```

```bash
chmod +x run_all.sh
nohup bash run_all.sh &

# Monitor progress
tail -f run_all_*.log

# Check for failures next morning
grep -E "Running:|Finished OK:|FAILED" run_all_*.log
```

---

## Output Files

Each episode saves to `outputs/<run_name>/run_XX/`:

```
outputs/beers_concrete_qwen25_7b/
  run_00/
    steps.json            ← lightweight step summary (step_type, reasoning, action)
    trajectory.json       ← full trace: state + observation before/after each step
    interaction_log.json  ← debug log: prompt + raw model output at every step
    summary.json          ← episode outcome + full evaluation metrics
  run_01/
    ...
  aggregate.json          ← metrics averaged across all N runs
```

All files are also uploaded to W&B as Artifacts (one per episode), accessible under the **Artifacts** tab.

---

## W&B Metrics

### Per-episode

| Metric | Description |
|---|---|
| `success` | Finish reached AND equation algebraically correct |
| `equation_correct` | Submitted equation matches ground truth |
| `finish_called` | Model emitted a finish step |
| `total_steps` | Steps executed in this episode |
| `steps_to_success` | Steps taken when succeeded (null if failed) |
| `state_coverage_ratio` | Unique states visited / total steps |
| `redundancy_penalty` | Repeated actions / total steps |
| `variable_isolation_score` | Fraction of action steps where the target variable actually changed |
| `discovery_efficiency` | 1 / total_steps if success, else 0 |

### Experiment summary

| Metric | Description |
|---|---|
| `success_rate` | Fraction of runs that succeeded |
| `finish_rate` | Fraction of runs where model called finish |
| `parse_error_rate` | Fraction of runs that ended due to invalid JSON output |
| `mean_total_steps` | Average steps per run |
| `mean_steps_to_success` | Average steps among successful runs |
| `termination_breakdown` | Count of each termination reason |

---

## Module Reference

### `env.py` — `EquationEnv`
Maintains the true environment state. On each `step(action)`, updates the manipulated variable and recomputes all output variables via `EquationEngine`. Supports any number of variables and any equation from config. Raises `ValueError` if a non-manipulable variable is targeted.

### `equation_engine.py` — `EquationEngine`
Parses and evaluates equation strings using Python's AST module (safe — no `eval()`). Equations are evaluated in definition order so later equations can depend on earlier results.

### `renderer.py` — `TextRenderer`
Converts environment state into a text `Observation`. Applies `naming_mode` (concrete/abstract) and `metadata_level` (rich/minimal/none). Available actions are rendered as human-readable strings (e.g. `increase R1`, `set concentration to <number>`).

### `prompt_builder.py` — `PromptBuilder`
Assembles the full prompt from config-supplied templates plus runtime data (observation, history). Takes `action_mode` at init and shows only the output format template matching the current mode. No hardcoded strings — all text comes from `prompt_default.yaml`.

### `agent.py` — `TextLLMAgent`
Accepts any `model_callable: Callable[[str], str]`. Calls the model, strips markdown fences and Qwen3 `<think>` blocks, extracts the JSON object, and parses it into an `AgentStep`. Raises `ValueError` on invalid output.

### `runner.py` — `EpisodeRunner`
The main episode loop. On invalid actions (e.g. targeting a non-manipulable variable), injects an `[ERROR]` message into the next observation instead of terminating, allowing the model to self-correct. Terminates on `finish`, `max_steps`, or unrecoverable parse error.

### `evaluator.py` — `EpisodeEvaluator`
Computes all episode metrics from the runner result. Delegates equation comparison to `EquationMatcher`.

### `equation_matcher.py` — `EquationMatcher`
Uses SymPy to check algebraic equivalence between the model's submitted equation and the ground truth. Supports variable name mapping so abstract equations (e.g. `Y = A * B * C`) can be matched against concrete ground truth (e.g. `absorbance = molar_absorptivity * concentration * path_length`).

### `logger.py` — `EpisodeLogger`
Saves all output files to disk and returns their paths for W&B upload.

### `config_loader.py`
Loads and merges main config and all sub-configs into a single dict. Validates required keys. CLI overrides (`--env_config`, `--model_config`) take precedence over values in the main config.

---

## Adding a New Environment

No code changes needed. Create a new env YAML:

```yaml
# configs/env_velocity.yaml
environment:
  target_variable: velocity
  variables:
    distance:
      manipulable: true
      initial_value: 100
      step_size: 10
      min_value: null
      max_value: null
      description: distance travelled in metres
    time:
      manipulable: true
      initial_value: 5
      step_size: 1
      min_value: null
      max_value: null
      description: time elapsed in seconds
    velocity:
      manipulable: false
      description: speed of the object in metres per second
  equations:
    velocity: "distance / time"

representation:
  naming_mode: concrete
  metadata_level: rich
  name_mapping:
    distance: distance
    time: time
    velocity: velocity

actions:
  action_mode: increase_decrease

evaluation:
  variable_mapping:
    distance: distance
    time: time
    velocity: velocity

noise:
  enabled: false
  type: null
  params: {}
```

Then run:

```bash
python run_episode.py \
  --config configs/config.yaml \
  --env_config configs/env_velocity.yaml \
  --model_config configs/model_qwen25_7b.yaml
```

---

## Design Principles

- **Config-driven**: all experimental conditions controlled through YAML, not source code
- **Equation-driven**: any equation in config automatically defines the environment, ground truth, and evaluation target
- **Trajectory-first**: the primary research artifact is the full exploration trace, not just the final answer
- **Think before act**: JSON schema puts `reasoning` first so the model writes its reasoning before committing to a step type
- **Clean separation**: observation layer sits between environment state and prompt — enables future image/multimodal observations without changing env or agent code
- **Graceful error handling**: invalid model actions produce corrective feedback rather than crashing