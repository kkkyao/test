# Scientific Exploration Framework

A config-driven, multi-step research framework for studying how language models explore scientific environments, form hypotheses, and discover underlying equations through iterative interaction.

---

## What This Is

This is **not** a question-answering system. It is an interactive, trajectory-oriented research pipeline where a model acts as an explorer inside a controlled scientific environment.

The primary research questions are:
- How do models explore scientific environments step by step?
- Does variable representation (concrete names vs. abstract symbols) affect exploration behavior?
- Does contextual metadata (rich descriptions vs. none) affect whether a model can discover the underlying equation?
- Do larger or newer models explore more efficiently?

The default environment is a Beer's Law-style setup, but the framework is fully **equation-driven**: any equation supplied via config will automatically define the environment, the action space, the ground truth, and the evaluation target.

---

## Project Structure

```
project/
  configs/
    config.yaml                  # Main entry point — points to sub-configs
    experiment_default.yaml      # Runtime parameters (steps, logging, evaluate)
    experiment_concrete.yaml     # Experiment config for concrete condition
    experiment_abstract.yaml     # Experiment config for abstract condition
    env_beers_concrete.yaml      # Beer's Law env, concrete naming, rich metadata
    env_beers_abstract.yaml      # Beer's Law env, abstract naming, no metadata
    model_mock.yaml              # Mock backend for pipeline testing
    model_qwen25_3b.yaml         # Qwen2.5-3B-Instruct
    model_qwen25_7b.yaml         # Qwen2.5-7B-Instruct
    model_qwen35_4b.yaml         # Qwen3.5-4B
    model_qwen35_9b.yaml         # Qwen3.5-9B
    prompt_default.yaml          # All prompt text (no hardcoded strings in code)

  src/
    schemas/
      action_schema.py           # ActionSpec dataclass
      agent_output_schema.py     # AgentStep dataclass + StepType
      observation_schema.py      # Observation dataclass + ObservationMode
      trace_schema.py            # TraceStep dataclass (full trajectory record)

    envs/
      env.py                     # EquationEnv: state management + action execution
      equation_engine.py         # Safe AST-based equation parsing and evaluation

    observation/
      renderer.py                # TextRenderer: state → Observation text

    prompts/
      prompt_builder.py          # PromptBuilder: observation + history → prompt

    agents/
      agent.py                   # TextLLMAgent: prompt → model → AgentStep

    runners/
      runner.py                  # EpisodeRunner: main episode loop

    tracing/
      logger.py                  # EpisodeLogger: saves all output files

    evaluation/
      evaluator.py               # EpisodeEvaluator: computes all episode metrics
      equation_matcher.py        # EquationMatcher: algebraic equivalence via SymPy

    utils/
      config_loader.py           # YAML loading, merging, and validation

  run_episode.py                 # Entry point for a single episode
  run_experiment.py              # Entry point for N runs with W&B logging
```

---

## How an Episode Works

```
env.reset()
    → initial state
    → renderer.render(state) → observation

for step in range(max_steps):

    prompt_builder.build_prompt(observation, history) → prompt
    agent.act(prompt) → AgentStep, raw_output

    if step_type == "action":
        env.step(action)              ← only actions change the environment
        renderer.render(new_state)    → new observation

    elif step_type == "finish":
        record final_equation
        break                         ← episode ends

    record TraceStep                  ← every step is fully logged

EpisodeEvaluator.evaluate(result)
EpisodeLogger.save(result, evaluation)
```

---

## Model Output Format

The model must return a strict JSON object on every step, with **`reasoning` first** so the model is forced to think before committing to an action:

```json
{
  "reasoning": "Concentration increased from 100 to 110 and absorbance went from 1.0 to 1.1. I want to test path_length next.",
  "hypothesis": "Absorbance may be proportional to both concentration and path_length.",
  "step_type": "action",
  "action": {
    "action_type": "increase",
    "variable": "path_length",
    "value": null
  },
  "final_equation": null
}
```

`reasoning` is placed first intentionally: because LLMs generate tokens left-to-right, putting `reasoning` before `step_type` forces the model to think before it decides what to do.

### Step types

| `step_type` | Required fields | Changes environment? |
|---|---|---|
| `action` | `reasoning`, `action` | **Yes** |
| `finish` | `reasoning`, `final_equation` | No — ends episode |

`hypothesis` is optional on any `action` step. It is not a step type — it is an accompanying belief field.

---

## Experimental Conditions

Two representational dimensions combine into four conditions:

| Condition | `naming_mode` | `metadata_level` | Variable names | Descriptions |
|---|---|---|---|---|
| Concrete + Rich | `concrete` | `rich` | concentration, path_length | Full descriptions + step sizes |
| Abstract + None | `abstract` | `none` | A, B, C, Y1, Y2 | None |

The default setup compares **concrete** vs **abstract** using the two env configs provided.

---

## Supported Models

All models use `backend: hf_qwen` and are loaded locally via HuggingFace Transformers.

| Config file | Model | Parameters | Notes |
|---|---|---|---|
| `model_mock.yaml` | Mock | — | Pipeline testing only, no GPU needed |
| `model_qwen25_3b.yaml` | `Qwen/Qwen2.5-3B-Instruct` | 3B | Baseline, no thinking mode |
| `model_qwen25_7b.yaml` | `Qwen/Qwen2.5-7B-Instruct` | 7B | Scaling comparison |
| `model_qwen35_4b.yaml` | `Qwen/Qwen3.5-4B` | 4B | Requires transformers ≥ 4.51 |
| `model_qwen35_9b.yaml` | `Qwen/Qwen3.5-9B` | 9B | Requires transformers ≥ 4.51 |

**Generation settings** (all models): `temperature=0.7`, `top_p=0.8`, `top_k=20`, `do_sample=true` — following official Qwen recommendations for non-thinking instruct mode.

**GPU memory requirements** (bfloat16):

| Model | VRAM needed |
|---|---|
| Qwen2.5-3B | ~7 GB |
| Qwen3.5-4B | ~10 GB |
| Qwen2.5-7B | ~15 GB |
| Qwen3.5-9B | ~19 GB |

---

## Installation

```bash
pip install pyyaml sympy transformers torch wandb
```

For Qwen3.5 models, install the optional fast linear attention library (optional but recommended):

```bash
pip install flash-linear-attention causal-conv1d
```

---

## Running Experiments

### 1. Test the pipeline with mock model

```bash
python run_episode.py --config configs/config.yaml
```

Expected output: 3 steps (2 actions + 1 finish), `finish_reached: true`, no `parse_error`.

### 2. Run a single episode with a real model

```bash
# Concrete condition
python run_episode.py \
  --env_config configs/env_beers_concrete.yaml \
  --model_config configs/model_qwen25_3b.yaml

# Abstract condition
python run_episode.py \
  --env_config configs/env_beers_abstract.yaml \
  --model_config configs/model_qwen25_3b.yaml

# With evaluation
python run_episode.py \
  --env_config configs/env_beers_concrete.yaml \
  --model_config configs/model_qwen25_3b.yaml \
  --evaluate
```

### 3. Run a full experiment (N runs + W&B logging)

```bash
# Login to W&B first
wandb login

python run_experiment.py \
  --config configs/config.yaml \
  --env_config configs/env_beers_concrete.yaml \
  --model_config configs/model_qwen25_3b.yaml \
  --n_runs 30 \
  --run_name concrete_qwen25_3b
```

### 4. Run all 8 experimental conditions overnight

```bash
nohup bash -c '
python run_experiment.py --config configs/config.yaml --env_config configs/env_beers_concrete.yaml --model_config configs/model_qwen25_3b.yaml --n_runs 30 --run_name concrete_qwen25_3b && \
python run_experiment.py --config configs/config.yaml --env_config configs/env_beers_concrete.yaml --model_config configs/model_qwen35_4b.yaml --n_runs 30 --run_name concrete_qwen35_4b && \
python run_experiment.py --config configs/config.yaml --env_config configs/env_beers_concrete.yaml --model_config configs/model_qwen25_7b.yaml --n_runs 30 --run_name concrete_qwen25_7b && \
python run_experiment.py --config configs/config.yaml --env_config configs/env_beers_concrete.yaml --model_config configs/model_qwen35_9b.yaml --n_runs 30 --run_name concrete_qwen35_9b && \
python run_experiment.py --config configs/config.yaml --env_config configs/env_beers_abstract.yaml --model_config configs/model_qwen25_3b.yaml --n_runs 30 --run_name abstract_qwen25_3b && \
python run_experiment.py --config configs/config.yaml --env_config configs/env_beers_abstract.yaml --model_config configs/model_qwen35_4b.yaml --n_runs 30 --run_name abstract_qwen35_4b && \
python run_experiment.py --config configs/config.yaml --env_config configs/env_beers_abstract.yaml --model_config configs/model_qwen25_7b.yaml --n_runs 30 --run_name abstract_qwen25_7b && \
python run_experiment.py --config configs/config.yaml --env_config configs/env_beers_abstract.yaml --model_config configs/model_qwen35_9b.yaml --n_runs 30 --run_name abstract_qwen35_9b
' > experiment_log.txt 2>&1 &

# Monitor progress
tail -f experiment_log.txt
```

---

## Output Files

Each episode saves to the directory specified in `experiment_*.yaml`:

```
outputs/beers_text_default/
  run_00/
    steps.json            ← lightweight step summary (step_type, reasoning, action)
    trajectory.json       ← full trace with state/observation before and after each step
    interaction_log.json  ← debug log with prompt + raw model output at every step
    summary.json          ← episode outcome + full evaluation metrics
  run_01/
    ...
  aggregate.json          ← metrics averaged across all runs
```

All files are also uploaded to W&B as Artifacts (one artifact per episode), accessible under the **Artifacts** tab in the W&B project.

---

## W&B Metrics

### Per-episode (logged in real time)

| Metric | Description |
|---|---|
| `success` | Finish reached AND equation correct |
| `equation_correct` | Submitted equation matches ground truth algebraically |
| `finish_called` | Model emitted a finish step |
| `total_steps` | Steps executed in this episode |
| `steps_to_success` | Steps taken when succeeded (null if not) |
| `state_coverage_ratio` | Unique states visited / total steps |
| `redundancy_penalty` | Repeated actions / total steps |
| `variable_isolation_score` | Fraction of steps where exactly one variable changed |
| `discovery_efficiency` | 1 / total_steps if success, else 0 |

### Experiment summary (after all runs)

| Metric | Description |
|---|---|
| `success_rate` | Fraction of runs that succeeded |
| `finish_rate` | Fraction of runs where model called finish |
| `parse_error_rate` | Fraction of runs that ended due to invalid output |
| `mean_total_steps` | Average steps per run |
| `termination_breakdown` | Count of each termination reason |

---

## Module Reference

### `env.py` — `EquationEnv`
Maintains the true environment state. On each `step(action)`, updates the manipulated variable and recomputes all output variables using `EquationEngine`. Supports any number of variables and any equation defined in config. Raises `ValueError` if a non-manipulable variable is targeted.

### `equation_engine.py` — `EquationEngine`
Parses equation strings from config using Python's AST module (safe — no `eval()`). Evaluates equations against the current state in definition order so later equations can depend on earlier ones.

### `renderer.py` — `TextRenderer`
Converts environment state into a text `Observation`. Applies `naming_mode` (concrete/abstract) and `metadata_level` (rich/minimal/none). In `increase_decrease` mode, step sizes are shown in the observation so the model knows the exact magnitude of each action.

### `prompt_builder.py` — `PromptBuilder`
Assembles the full prompt from config-supplied text templates plus runtime data (current observation, history, step count). No hardcoded strings — all text comes from `prompt_default.yaml`. The JSON schema in the prompt puts `reasoning` first to enforce think-before-act behavior.

### `agent.py` — `TextLLMAgent`
Accepts any `model_callable: Callable[[str], str]`. Calls the model, strips markdown fences, extracts the JSON object (robust to surrounding prose), and parses it into an `AgentStep`. Raises `ValueError` on invalid output with a detailed error message.

### `runner.py` — `EpisodeRunner`
The main episode loop. On invalid actions (non-manipulable variable), injects an `[ERROR]` message into the next observation instead of terminating, allowing the model to self-correct. Terminates on `finish`, `max_steps`, or unrecoverable parse error.

### `evaluator.py` — `EpisodeEvaluator`
Computes all episode metrics from the runner result. Delegates equation comparison to `EquationMatcher`. Returns a flat JSON-friendly dict suitable for W&B logging.

### `equation_matcher.py` — `EquationMatcher`
Uses SymPy to check algebraic equivalence between the model's final equation and the ground truth. Supports variable name mapping so abstract equations (`Y2 = 0.005 * A * B`) can be matched against concrete ground truth (`absorbance = concentration * path_length / 200`).

### `logger.py` — `EpisodeLogger`
Saves all output files to disk and returns their paths. Used by `run_experiment.py` to upload files as W&B Artifacts.

### `config_loader.py`
Loads and merges the main config and all sub-configs into one dict. CLI overrides (`--env_config`, `--model_config`) take precedence over values in the main config file.

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
      step_size: 50
      description: distance travelled
    time:
      manipulable: true
      initial_value: 5
      step_size: 1
      description: time elapsed
    velocity:
      manipulable: false
      description: speed of the object
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
python run_episode.py --env_config configs/env_velocity.yaml --model_config configs/model_qwen25_3b.yaml
```

---

## Design Principles

- **Config-driven**: all experimental conditions are controlled through YAML, not source code
- **Equation-driven**: any equation in config automatically defines the environment, ground truth, and evaluation target
- **Trajectory-first**: the primary research artifact is the full exploration trace, not just the final answer
- **Think before act**: the JSON schema puts `reasoning` first so the model writes its reasoning before committing to a step type
- **Graceful error handling**: invalid model actions produce corrective feedback rather than crashing
- **Clean module boundaries**: each module owns exactly one responsibility; no cross-cutting logic
