# Parse Error Case Inventory — 4 Directories

## 1. Scan Scope

Only the following 4 directories were scanned:

- **`wandb_downloads_image_only`** → modality: Simulation-only  | 360 episodes | **3 parse_errors**
- **`wandb_downloads_text_image_chart_only_action_text`** → modality: Text+image  | 360 episodes | **2 parse_errors**
- **`wandb_downloads_text_image_chart_only2`** → modality: Chart-only  | 360 episodes | **6 parse_errors**
- **`wandb_downloads_text_only`** → modality: Text-only  | 1259 episodes | **59 parse_errors**

**Total parse_error cases: 70** across 2339 episodes (3.0% overall rate)

---

## 2. Distribution by Source Directory

| Source Dir | Parse Errors |
|---|---:|
| `wandb_downloads_image_only` | 3 |
| `wandb_downloads_text_image_chart_only_action_text` | 2 |
| `wandb_downloads_text_image_chart_only2` | 6 |
| `wandb_downloads_text_only` | 59 |

## 3. Distribution by Modality

| Modality | Parse Errors |
|---|---:|
| Chart-only | 6 |
| Simulation-only | 3 |
| Text+image | 2 |
| Text-only | 59 |

## 4. Distribution by Model

| Model | Parse Errors |
|---|---:|
| model_mistral_7b | 50 |
| model_qwen25_vl_3b | 8 |
| model_qwen35_9b | 5 |
| model_qwen35_4b | 4 |
| model_qwen25_vl_7b | 3 |

## 5. Distribution by Error Type

| diagnosed_error_type | Count | Fixable by code? |
|---|---:|---|
| `missing_comma` | 53 | No — `_repair()` cannot add missing commas; relies on `max_parse_retries` |
| `invalid_json_general` | 8 | Partial — unterminated string needs `max_new_tokens` tuning |
| `latex_backslash_escape` | 5 | Yes — `_repair()` fix already merged + prompt rule added |
| `finish_schema_error` | 2 | Partial — relies on `max_parse_retries`; prompt already specifies finish_reason/final_equation |
| `action_schema_error` | 1 | Partial — relies on `max_parse_retries`; prompt specifies 'variable' required |
| `missing_required_field` | 1 | Partial — relies on `max_parse_retries` |

---

## 6. Cases by Modality

### Chart-only (6 cases)

| case_id | formula | setup | model | error_type | failed_step | raw_len |
|---|---|---|---|---|---:|---:|
| 6 | distance | concrete | model_qwen25_vl_3b | `finish_schema_error` | 3 | 0 |
| 7 | kinematics | abbrev | model_qwen25_vl_7b | `latex_backslash_escape` | 0 | 315 |
| 8 | beers | concrete | model_qwen25_vl_3b | `missing_comma` | 7 | 287 |
| 9 | beers | abbrev | model_qwen25_vl_3b | `missing_comma` | 6 | 308 |
| 10 | kinematics | abbrev | model_qwen25_vl_7b | `latex_backslash_escape` | 0 | 315 |
| 11 | beers | abbrev | model_qwen25_vl_3b | `latex_backslash_escape` | 2 | 310 |

### Simulation-only (3 cases)

| case_id | formula | setup | model | error_type | failed_step | raw_len |
|---|---|---|---|---|---:|---:|
| 1 | kinematics | abstract | model_qwen25_vl_3b | `missing_comma` | 3 | 302 |
| 2 | beers | abbrev | model_qwen25_vl_3b | `finish_schema_error` | 2 | 0 |
| 3 | kinematics | abbrev | model_qwen25_vl_7b | `latex_backslash_escape` | 0 | 310 |

### Text+image (2 cases)

| case_id | formula | setup | model | error_type | failed_step | raw_len |
|---|---|---|---|---|---:|---:|
| 4 | beers | concrete | model_qwen25_vl_3b | `action_schema_error` | 11 | 0 |
| 5 | ohm | abstract | model_qwen25_vl_3b | `missing_comma` | 0 | 266 |

### Text-only (59 cases)

| case_id | formula | setup | model | error_type | failed_step | raw_len |
|---|---|---|---|---|---:|---:|
| 12 | ohm | abstract | model_mistral_7b | `missing_comma` | 11 | 304 |
| 13 | distance | abstract | model_mistral_7b | `missing_comma` | 4 | 240 |
| 14 | ohm | abstract | model_mistral_7b | `missing_required_field` | 5 | 0 |
| 15 | kinematics | abstract | model_qwen35_9b | `invalid_json_general` | 3 | 302 |
| 16 | kinematics | abstract | model_qwen35_4b | `missing_comma` | 3 | 302 |
| 17 | mass | abstract | model_mistral_7b | `missing_comma` | 4 | 261 |
| 18 | kinematics | abstract | model_qwen35_4b | `invalid_json_general` | 2 | 303 |
| 19 | kinematics | abstract | model_qwen35_9b | `invalid_json_general` | 3 | 303 |
| 20 | beers | abbrev | model_mistral_7b | `missing_comma` | 3 | 324 |
| 21 | mass | abbrev | model_mistral_7b | `missing_comma` | 3 | 260 |
| 22 | mass | abstract | model_mistral_7b | `missing_comma` | 5 | 289 |
| 23 | resistors | abbrev | model_mistral_7b | `missing_comma` | 7 | 281 |
| 24 | mass | abstract | model_mistral_7b | `missing_comma` | 8 | 298 |
| 25 | ohm | concrete | model_mistral_7b | `missing_comma` | 6 | 290 |
| 26 | ohm | concrete | model_mistral_7b | `missing_comma` | 8 | 308 |
| 27 | beers | abbrev | model_mistral_7b | `missing_comma` | 6 | 310 |
| 28 | ohm | abstract | model_mistral_7b | `missing_comma` | 6 | 304 |
| 29 | resistors | abbrev | model_mistral_7b | `missing_comma` | 3 | 266 |
| 30 | kinematics | abstract | model_mistral_7b | `missing_comma` | 8 | 307 |
| 31 | distance | abbrev | model_mistral_7b | `missing_comma` | 4 | 258 |
| 32 | mass | abstract | model_mistral_7b | `missing_comma` | 7 | 271 |
| 33 | kinematics | abstract | model_mistral_7b | `missing_comma` | 3 | 190 |
| 34 | beers | abstract | model_mistral_7b | `latex_backslash_escape` | 12 | 309 |
| 35 | kinematics | abstract | model_mistral_7b | `missing_comma` | 12 | 277 |
| 36 | kinematics | abstract | model_qwen35_9b | `invalid_json_general` | 4 | 303 |
| 37 | mass | abstract | model_mistral_7b | `missing_comma` | 6 | 254 |
| 38 | kinematics | abstract | model_mistral_7b | `missing_comma` | 9 | 262 |
| 39 | mass | abstract | model_mistral_7b | `missing_comma` | 4 | 308 |
| 40 | distance | abbrev | model_mistral_7b | `missing_comma` | 6 | 194 |
| 41 | ohm | abstract | model_mistral_7b | `missing_comma` | 3 | 306 |
| 42 | resistors | abstract | model_mistral_7b | `missing_comma` | 7 | 278 |
| 43 | distance | abbrev | model_mistral_7b | `missing_comma` | 11 | 304 |
| 44 | mass | abbrev | model_mistral_7b | `missing_comma` | 8 | 304 |
| 45 | distance | abbrev | model_mistral_7b | `missing_comma` | 13 | 203 |
| 46 | mass | abbrev | model_mistral_7b | `missing_comma` | 3 | 307 |
| 47 | beers | concrete | model_mistral_7b | `missing_comma` | 6 | 302 |
| 48 | kinematics | abbrev | model_mistral_7b | `missing_comma` | 4 | 287 |
| 49 | distance | abbrev | model_mistral_7b | `missing_comma` | 5 | 308 |
| 50 | mass | concrete | model_mistral_7b | `missing_comma` | 2 | 303 |
| 51 | resistors | abstract | model_mistral_7b | `missing_comma` | 8 | 236 |
| 52 | kinematics | concrete | model_mistral_7b | `missing_comma` | 8 | 302 |
| 53 | kinematics | abstract | model_qwen35_9b | `invalid_json_general` | 5 | 302 |
| 54 | kinematics | abstract | model_qwen35_9b | `invalid_json_general` | 3 | 303 |
| 55 | kinematics | abstract | model_qwen35_4b | `invalid_json_general` | 1 | 302 |
| 56 | beers | concrete | model_mistral_7b | `missing_comma` | 5 | 302 |
| 57 | kinematics | abbrev | model_mistral_7b | `missing_comma` | 2 | 293 |
| 58 | distance | abstract | model_mistral_7b | `missing_comma` | 3 | 303 |
| 59 | mass | concrete | model_mistral_7b | `missing_comma` | 14 | 302 |
| 60 | beers | abstract | model_mistral_7b | `missing_comma` | 11 | 244 |
| 61 | resistors | concrete | model_mistral_7b | `missing_comma` | 3 | 308 |
| 62 | beers | abbrev | model_mistral_7b | `missing_comma` | 8 | 307 |
| 63 | mass | abstract | model_mistral_7b | `missing_comma` | 4 | 207 |
| 64 | kinematics | abstract | model_mistral_7b | `missing_comma` | 7 | 272 |
| 65 | kinematics | abstract | model_qwen35_4b | `invalid_json_general` | 2 | 302 |
| 66 | beers | abstract | model_mistral_7b | `missing_comma` | 4 | 179 |
| 67 | resistors | abstract | model_mistral_7b | `missing_comma` | 20 | 227 |
| 68 | ohm | abbrev | model_mistral_7b | `missing_comma` | 4 | 283 |
| 69 | beers | concrete | model_mistral_7b | `missing_comma` | 5 | 303 |
| 70 | ohm | abstract | model_mistral_7b | `missing_comma` | 2 | 304 |

---

## 7. Raw Output Availability

All 70 cases are **old-style** (pre-fix runs). None have new-style `failed_parse_attempt` entries in `interaction_log.json`.

| Availability | Count |
|---|---:|
| Raw output snippet available (truncated ~200–300 chars from parse_error field) | 66 |
| Cleaned output available | 0 |
| No raw output at all (schema errors only) | 4 |

**Schema errors** (finish_schema_error, action_schema_error, missing_required_field) have no raw output snippet because the parse_error message is a schema validation string, not a JSON decode error. The full raw output is lost in these cases.

---

## 8. Which Cases Are Likely Fixed by Recent Code Changes?

### Fixed by `_repair()` invalid-escape patch + prompt LaTeX rule

**`latex_backslash_escape` cases (5) — all should now succeed after the fix:**

| case_id | modality | formula | setup | model | snippet |
|---|---|---|---|---|---|
| 3 | Simulation-only | kinematics | abbrev | model_qwen25_vl_7b | `[json_decode] model output is not valid JSON (error: Invalid \escape at pos 64):` |
| 7 | Chart-only | kinematics | abbrev | model_qwen25_vl_7b | `[json_decode] model output is not valid JSON (error: Invalid \escape at pos 64):` |
| 10 | Chart-only | kinematics | abbrev | model_qwen25_vl_7b | `[json_decode] model output is not valid JSON (error: Invalid \escape at pos 64):` |
| 11 | Chart-only | beers | abbrev | model_qwen25_vl_3b | `[json_decode] model output is not valid JSON (error: Invalid \escape at pos 285)` |
| 34 | Text-only | beers | abstract | model_mistral_7b | `[json_decode] model output is not valid JSON (error: Invalid \escape at pos 298)` |

### Remains Dependent on `max_parse_retries`

**`missing_comma` cases (53)** — `_repair()` cannot add missing commas. With `max_parse_retries=3`, each failing step gets 3 additional attempts to regenerate valid JSON.

**`invalid_json_general` (unterminated string) cases (8)** — Caused by output truncation when reasoning exceeds `max_new_tokens`. Needs `max_new_tokens` increase in model config, not a parser fix.

---

*Generated by inline diagnostic script. Data source: old-style episode logs (pre-fix runs).*