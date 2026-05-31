# Experiment Inventory — Three W&B Download Directories

Scanned directories (exclusive):
1. `wandb_downloads_text_image_chart_only2`
2. `wandb_downloads_text_image_chart_only_action_text`
3. `wandb_downloads_image_only`

---

## 1. What Each Directory Represents

### `wandb_downloads_text_image_chart_only2` → **Chart-only**

| Field | Value | Source |
|---|---|---|
| Modality | Chart-only | inferred from dir name + output.log |
| Prompt type | Bar-chart images, **no action history** in prompt | from history_labels in prompt (step_type only, no action key) |
| W&B project | `scientific-exploration-increase-text-image-chart-only2` | from directory name |
| Original base config | `configs/config_chart_only_no_action.yaml` | from output.log — **file no longer exists** |
| Current equivalent | `configs/config_strict_chart_only.yaml` | inferred |
| Prompt config used | `prompt_strict_chart_only_fomula.yaml` | inferred (no action in history_labels) |
| Experiment config | `experiment_chart_action.yaml` | inferred (contains max_parse_retries=3) |
| Task mode | `formula_discovery` | from output.log |
| Observation mode | `text_image` | from output.log |
| Models | Qwen2.5-VL-3B, Qwen2.5-VL-7B | from output.log |
| Formulas | beers, distance, kinematics, mass, ohm, resistors (6) | from env configs in output.log |
| Setups | concrete, abbrev, abstract (3) | from env configs |
| N runs per condition | 10 | from output.log |
| Total episodes | 360 (36 conditions × 10) | counted |

### `wandb_downloads_text_image_chart_only_action_text` → **Text+image**

| Field | Value | Source |
|---|---|---|
| Modality | Text+image | inferred from dir name + output.log |
| Prompt type | Bar-chart images **with action history text** in prompt | from history_labels (has action key) |
| W&B project | `scientific-exploration-increase-text-image-chart-only-action-text` | from directory name |
| Original base config | `configs/config_chart_only.yaml` | from output.log — **file no longer exists** |
| Current equivalent | `configs/config_chart_action.yaml` | inferred |
| Prompt config used | `prompt_chart_action_fomula.yaml` | inferred (has action in history_labels) |
| Experiment config | `experiment_chart_action.yaml` | inferred |
| Task mode | `formula_discovery` | from output.log |
| Observation mode | `text_image` | from output.log |
| Models | Qwen2.5-VL-3B, Qwen2.5-VL-7B | from output.log |
| Formulas | beers, distance, kinematics, mass, ohm, resistors (6) | from env configs |
| Setups | concrete, abbrev, abstract (3) | from env configs |
| N runs per condition | 10 | from output.log |
| Total episodes | 360 | counted |

### `wandb_downloads_image_only` → **Simulation-only** (in this project's naming)

| Field | Value | Source |
|---|---|---|
| Modality | Simulation-only (same setup as Chart-only; different W&B project) | inferred from dir name |
| Prompt type | Bar-chart images, **no action history** | same config as Chart-only |
| W&B project | `scientific-exploration-increase-image-only-` | from directory name |
| Original base config | `configs/config_chart_only_no_action.yaml` | from output.log — **file no longer exists** |
| Current equivalent | `configs/config_strict_chart_only.yaml` | inferred |
| Prompt config used | `prompt_strict_chart_only_fomula.yaml` | inferred |
| Experiment config | `experiment_chart_action.yaml` | inferred |
| Task mode | `formula_discovery` | from output.log |
| Observation mode | `text_image` | from output.log |
| Models | Qwen2.5-VL-3B, Qwen2.5-VL-7B | from output.log |
| Formulas | beers, distance, kinematics, mass, ohm, resistors (6) | from env configs |
| Setups | concrete, abbrev, abstract (3) | from env configs |
| N runs per condition | 10 | from output.log |
| Total episodes | 360 | counted |

> **Note:** `wandb_downloads_text_image_chart_only2` and `wandb_downloads_image_only` used the
> **same base config** at run time (`config_chart_only_no_action.yaml`). The only difference is
> the W&B project name and run_name prefix. This explains why both map to
> `config_strict_chart_only.yaml` as the current equivalent.

---

## 2. Fields: Direct vs Inferred

| Field | Source | Reliability |
|---|---|---|
| formula, setup | env_config path in output.log | **Direct** — high confidence |
| model | model_config path in output.log | **Direct** — high confidence |
| n_runs | output.log "N runs: 10" | **Direct** |
| termination_reason | summary.json evaluation.termination_reason | **Direct** |
| parse_error | summary.json parse_error field | **Direct** |
| forced_finish | summary.json forced_finish field | **Direct** |
| base config at runtime | output.log "Config: ..." | **Direct** but file **no longer exists** |
| current equivalent base config | inferred from prompt type | **Inferred** |
| experiment_config | inferred from current config files | **Inferred** |
| prompt_config | inferred from history_labels structure | **Inferred** |
| wandb_project | directory name | **Direct** |
| modality label | directory name + output.log | **Inferred** |

---

## 3. Condition Table with Parse Errors

### Chart-only (36 conditions)

| formula | setup | model | total | parse_err | finish_success | finish_wrong |
|---|---|---|---:|---:|---:|---:|
| beers | abbrev | qwen25_vl_3b | 10 | **2** | 0 | 8 |
| beers | abbrev | qwen25_vl_7b | 10 | 0 | 1 | 9 |
| beers | abstract | qwen25_vl_3b | 10 | 0 | 0 | 10 |
| beers | abstract | qwen25_vl_7b | 10 | 0 | 0 | 10 |
| beers | concrete | qwen25_vl_3b | 10 | **1** | 1 | 8 |
| beers | concrete | qwen25_vl_7b | 10 | 0 | 5 | 5 |
| distance | concrete | qwen25_vl_3b | 10 | **1** | 2 | 7 |
| kinematics | abbrev | qwen25_vl_7b | 10 | **2** | 0 | 8 |
| *(remaining 28 conditions)* | | | 10 | 0 | varies | varies |

Total parse_error in Chart-only: **6** (1.7%)

### Text+image (36 conditions)

| formula | setup | model | total | parse_err | finish_success | finish_wrong |
|---|---|---|---:|---:|---:|---:|
| beers | concrete | qwen25_vl_3b | 10 | **1** | 6 | 3 |
| ohm | abstract | qwen25_vl_3b | 10 | **1** | 0 | 9 |
| *(remaining 34 conditions)* | | | 10 | 0 | varies | varies |

Total parse_error in Text+image: **2** (0.6%)

### Simulation-only / image_only (36 conditions)

| formula | setup | model | total | parse_err | finish_success | finish_wrong |
|---|---|---|---:|---:|---:|---:|
| beers | abbrev | qwen25_vl_3b | 10 | **1** | 1 | 8 |
| kinematics | abbrev | qwen25_vl_7b | 10 | **1** | 1 | 8 |
| kinematics | abstract | qwen25_vl_3b | 10 | **1** | 0 | 9 |
| *(remaining 33 conditions)* | | | 10 | 0 | varies | varies |

Total parse_error in Simulation-only: **3** (0.8%)

---

## 4. Conditions That Cannot Be Reliably Mapped to Config

**All 108 conditions** can be mapped with **high confidence** because:
- All env configs (6 formulas × 3 setups) exist in `configs/`
- Both model configs (`model_qwen25_vl_3b.yaml`, `model_qwen25_vl_7b.yaml`) exist
- `experiment_chart_action.yaml` exists (with `max_parse_retries: 3`)
- `prompt_strict_chart_only_fomula.yaml` and `prompt_chart_action_fomula.yaml` exist

**One caveat (inferred, not low-confidence):**
The original base configs `config_chart_only_no_action.yaml` and `config_chart_only.yaml`
no longer exist. The rerun commands use the current equivalent configs
(`config_strict_chart_only.yaml` and `config_chart_action.yaml`).
These are functionally equivalent but the mapping is inferred, not directly verified.

---

## 5. Rerun Priority Recommendations

### Priority 1 — Conditions with parse_error (now fixed by code changes)

The following code fixes are now in place:
- `_repair()` handles invalid JSON backslash escapes (`\epsilon` → valid JSON)
- `prompt_strict_chart_only_fomula.yaml` has "no LaTeX" rules
- `max_parse_retries: 3` in `experiment_chart_action.yaml`
- `_run_forced_finish()` VLM bug fixed: now correctly calls `_generate(prompt, image_paths)`

Conditions with ≥1 parse_error that are most likely to benefit from rerun:

| dir | formula | setup | model | parse_errors |
|---|---|---|---|---:|
| chart_only2 | beers | abbrev | qwen25_vl_3b | 2 |
| chart_only2 | kinematics | abbrev | qwen25_vl_7b | 2 |
| chart_only2 | beers | concrete | qwen25_vl_3b | 1 |
| chart_only2 | distance | concrete | qwen25_vl_3b | 1 |
| image_only | beers | abbrev | qwen25_vl_3b | 1 |
| image_only | kinematics | abbrev | qwen25_vl_7b | 1 |
| image_only | kinematics | abstract | qwen25_vl_3b | 1 |
| chart_action_text | beers | concrete | qwen25_vl_3b | 1 |
| chart_action_text | ohm | abstract | qwen25_vl_3b | 1 |

### Priority 2 — All conditions (full rerun for clean data with fixed code)

Since the VLM forced_finish bug is now fixed, ALL 108 conditions are candidates for a full
rerun with the patched code. Even conditions without parse_error may have had episodes
where forced_finish silently failed (those would appear as finish_wrong but might actually
have extractable equations now).

---

## 6. Generated Files

| File | Description |
|---|---|
| `parse_error_analysis/three_dirs_experiment_inventory.csv` | Per-condition table (108 rows) |
| `parse_error_analysis/three_dirs_rerun_commands.csv` | One rerun command per condition |
| `scripts/rerun_three_dirs_conditions.sh` | Executable rerun script (108 commands) |
| `scripts/rerun_three_dirs_conditions_dry_run.sh` | Dry-run: only echoes commands |

---

*Generated by static analysis of output.log and summary.json files in the 3 target directories.*
