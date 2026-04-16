# Scientific Exploration Framework

A config-driven, multi-step framework for studying how language models explore scientific environments, form hypotheses, and discover underlying equations through iterative interaction.

---

## Overview

This framework is **not** a question-answering system. It is an interactive, trajectory-oriented research pipeline in which a model acts as an explorer inside a controlled scientific environment.

On each step, the model:
1. Observes the current environment state
2. Produces a **thought**, **hypothesis**, **action**, or **finish**
3. If an action is taken, the environment updates and a new observation is generated
4. The full process is recorded as a trajectory for research analysis

The primary research questions are:
- How do models explore scientific environments?
- Do models form hypotheses strategically?
- Does the variable representation (concrete vs. abstract) affect exploration behavior?
- Does the amount of contextual metadata (rich vs. none) affect whether a model can discover the underlying equation?

The default environment is a Beer's Law-style setup, but the framework is fully **equation-driven**: any equation supplied via config will automatically define the environment, the action space, the ground truth, and the evaluation target.

---

## Project Structure

```
project/
  configs/
    config.yaml                  # Main config router
    experiment_default.yaml      # Experiment runtime parameters
    env_beers_concrete.yaml      # Beer's Law env, concrete naming
    env_beers_abstract.yaml      # Beer's Law env, abstract naming
    model_qwen.yaml              # Model backend config

  src/
    schemas/
      action_schema.py           # ActionSpec dataclass
      agent_output_schema.py     # AgentStep dataclass + StepType
      observation_schema.py      # Observation dataclass + ObservationMode
      trace_schema.py            # TraceStep dataclass (full trajectory record)

    envs/
      env.py                     # EquationEnv: state management + action execution
      equation_engine.py         # Equation parsing and evaluation

    observation/
      renderer.py                # TextRenderer: state → Observation

    prompts/
      prompt_builder.py          # PromptBuilder: observation + history → prompt

    agents/
      agent.py                   # TextLLMAgent: prompt → model → AgentStep

    runners/
      runner.py                  # EpisodeRunner: main episode loop

    tracing/
      logger.py                  # EpisodeLogger: saves steps/trajectory/interaction logs

    evaluation/
      evaluator.py               # EpisodeEvaluator: computes episode metrics
      equation_matcher.py        # EquationMatcher: algebraic equivalence checking

    utils/
      config_loader.py           # YAML config loading and merging

  run_episode.py                 # Entry point
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install pyyaml sympy
```

### 2. Run an episode

```bash
python run_episode.py --config configs/config.yaml
```

### 3. Switch to abstract condition

In `configs/config.yaml`, change:

```yaml
env_config: env_beers_concrete.yaml
```

to:

```yaml
env_config: env_beers_abstract.yaml
```

No code changes needed.

---

## Configuration System

The framework is fully **config-driven**. All experiment conditions are controlled through YAML files. No source code modification is required to switch between experimental conditions.

### `configs/config.yaml`

The main config acts as a router to sub-configs:

```yaml
experiment_config: experiment_default.yaml
env_config: env_beers_concrete.yaml
model_config: model_qwen.yaml
```

### `configs/experiment_default.yaml`

Controls runtime behavior:

```yaml
experiment:
  name: beers_text_default
  max_steps: 10
  history_window: null

logging:
  output_dir: outputs/beers_text_default
  save_steps: true
  save_trajectory: true
  save_interaction_log: true
```

### `configs/env_beers_concrete.yaml` / `env_beers_abstract.yaml`

Defines the environment, equations, representation, actions, evaluation, and noise:

```yaml
environment:
  target_variable: absorbance
  variables:
    concentration:
      manipulable: true
      initial_value: 100
      step_size: 100
      description: amount of solute per volume
    # ... other variables
  equations:
    absorbance: "(concentration * path_length) / 200"
    transmittance: "100 - absorbance"

representation:
  naming_mode: concrete   # or "abstract"
  metadata_level: rich    # or "minimal" or "none"
  name_mapping:
    concentration: concentration  # concrete: identity; abstract: A, B, C...

actions:
  action_mode: increase_decrease  # or "set_value"

evaluation:
  variable_mapping:
    concentration: concentration  # abstract: A -> concentration
```

### `configs/model_qwen.yaml`

Controls the model backend:

```yaml
agent:
  backend: mock               # switch to hf_qwen for real inference
  model_name: Qwen2.5-14B-Instruct
  generation:
    temperature: 0.0
    max_new_tokens: 512
```

---

## Supported Experiment Conditions

The framework supports two independent representational dimensions, producing four experimental conditions:

| Condition | `naming_mode` | `metadata_level` |
|---|---|---|
| Concrete + Rich | `concrete` | `rich` |
| Concrete + None | `concrete` | `none` |
| Abstract + Rich | `abstract` | `rich` |
| Abstract + None | `abstract` | `none` |

All conditions share identical environment logic and equations. Only the representation layer changes.

---

## How an Episode Works

```
env.reset()
    → initial state
    → renderer.render(state)
    → observation

for step in range(max_steps):
    prompt_builder.build_prompt(observation, history)
        → prompt
    agent.act(prompt)
        → AgentStep, raw_output

    if step_type == "action":
        env.step(action)         ← only actions change the environment
        renderer.render(new_state)
        → new observation

    elif step_type == "finish":
        record final_equation
        break                    ← episode ends

    record TraceStep             ← every step is logged

→ EpisodeEvaluator.evaluate(result)
→ EpisodeLogger.save(result, evaluation)
```

---

## Model Output Format

The model must return a strict JSON object on every step. The schema is:

```json
{
  "step_type": "action",
  "reasoning": "I want to increase concentration while keeping the others fixed.",
  "hypothesis": null,
  "action": {
    "action_type": "increase",
    "variable": "concentration",
    "value": null
  },
  "final_equation": null
}
```

### Step type rules

| `step_type` | Required fields | Changes environment? |
|---|---|---|
| `thought` | `reasoning` | No |
| `hypothesis` | `hypothesis` | No |
| `action` | `reasoning`, `action` | **Yes** |
| `finish` | `final_equation` | No — ends episode |

### Action types

| `action_type` | When used | `value` required? |
|---|---|---|
| `increase` | `increase_decrease` mode | No (`null`) |
| `decrease` | `increase_decrease` mode | No (`null`) |
| `set` | `set_value` mode | **Yes** |

Only one action may be taken per step.

---

## Module Reference

### `equation_engine.py`

Parses equation strings from config and evaluates them against the current state. Supports arbitrary equations with any number of variables. Does not require hardcoded logic per environment.

```python
engine = EquationEngine({"absorbance": "(concentration * path_length) / 200"})
result = engine.evaluate({"concentration": 100, "path_length": 2})
# → {"absorbance": 1.0}
```

### `env.py` — `EquationEnv`

Maintains the true environment state. Handles `reset()`, `get_state()`, and `step(action)`. Delegates output calculation to `EquationEngine`. Supports arbitrary variable counts and equations driven entirely by config.

### `renderer.py` — `TextRenderer`

Converts state into a text `Observation`. Applies `naming_mode` (concrete/abstract) and `metadata_level` (rich/minimal/none). Also generates the available action list. Interface is designed to support `image` and `text_image` modes in future phases.

### `prompt_builder.py` — `PromptBuilder`

Assembles the full prompt from the current observation, history, task description, action format instructions, and JSON output requirements. Template is fixed; content updates every step.

### `agent.py` — `TextLLMAgent`

Accepts any `model_callable: Callable[[str], str]` so the backend is fully pluggable. Calls the model, strips markdown fences, parses JSON, and returns `(AgentStep, raw_output)`. Strict parsing with clear errors; no silent fallbacks.

### `runner.py` — `EpisodeRunner`

The main episode loop. Connects all components:
`env → renderer → prompt_builder → agent → trace`

Enforces `max_steps`. Returns a structured result dict with `steps`, `trajectory`, `final_equation`, `finish_reached`, and `num_steps`.

### `evaluator.py` — `EpisodeEvaluator`

Computes episode-level metrics from the runner result:

| Metric | Description |
|---|---|
| `total_steps` | Steps executed |
| `finish_reached` | Whether model explicitly finished |
| `unique_states_visited` | Distinct `state_after` values observed |
| `state_novelty_rate` | `unique_states / total_steps` |
| `repeated_state_ratio` | `(total_steps - unique_states) / total_steps` |
| `final_equation` | Model's final submitted equation |
| `equation_match` | Whether it matches ground truth (via `EquationMatcher`) |

### `equation_matcher.py` — `EquationMatcher`

Checks algebraic equivalence between the model's final equation and the ground truth. Supports:

- **Algebraic equivalence**: `A * B / 200` and `0.005 * A * B` are the same
- **Variable mapping equivalence**: `Y2 = 0.005 * A * B` matches `absorbance = concentration * path_length / 200` when the mapping `{A: concentration, B: path_length, Y2: absorbance}` is provided

### `logger.py` — `EpisodeLogger`

Saves three output files per episode:

| File | Contents |
|---|---|
| `steps.json` | Lightweight step-level view (type, reasoning, action, equation) |
| `trajectory.json` | Full trace per step including state, observation, prompt, raw output |
| `interaction_log.json` | Detailed debug log for replay and step-by-step review |

### `config_loader.py`

Loads and merges the main config + sub-configs into a single dict. Validates required keys. Preserves `_config_sources` for reproducibility and W&B tracking.

---

## Output Files

Each episode produces output in the directory specified by `logging.output_dir`:

```
outputs/beers_text_default/
  steps.json            ← high-level step summary
  trajectory.json       ← full trace for research analysis
  interaction_log.json  ← debug-level replay log
  evaluation.json       ← episode evaluation metrics
```

---

## Adding a New Environment

No code changes are required. Create a new env config YAML:

```yaml
# configs/env_velocity.yaml
environment:
  target_variable: velocity
  variables:
    distance:
      manipulable: true
      initial_value: 100
      step_size: 50
    time:
      manipulable: true
      initial_value: 5
      step_size: 1
    velocity:
      manipulable: false
  equations:
    velocity: "distance / time"

representation:
  naming_mode: concrete
  metadata_level: minimal
  name_mapping:
    distance: distance
    time: time
    velocity: velocity

actions:
  action_mode: set_value

evaluation:
  variable_mapping:
    distance: distance
    time: time
    velocity: velocity
```

Then point `config.yaml` to it:

```yaml
env_config: env_velocity.yaml
```

---

## Planned Extensions (Future Phases)

The following are explicitly **not** part of Phase 1 but are supported at the interface level:

- **Image observation** (`ObservationMode.IMAGE`): renderer interface reserved
- **Text + image observation** (`ObservationMode.TEXT_IMAGE`): renderer interface reserved
- **VLM backend**: pluggable via `model_callable`
- **GUI / browser agent backend**: no coupling in current architecture
- **Noisy environments**: `noise` block reserved in env config
- **W&B integration**: evaluation output and trace format are W&B-ready
- **Advanced process metrics**: information gain, variable coverage, hypothesis revision count

---

## Design Principles

- **Config-driven**: all experimental conditions are controlled through YAML, not source code
- **Equation-driven**: any equation in config automatically defines the environment, action space, ground truth, and evaluation target
- **Trajectory-first**: the primary research artifact is the full exploration trace, not the final answer
- **Clean module boundaries**: env, observation, prompt, agent, runner, logger, and evaluator each own exactly one responsibility
- **Debug-friendly**: every step records the prompt, raw model output, parsed step, state before/after, and observation before/after