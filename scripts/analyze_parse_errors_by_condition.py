#!/usr/bin/env python3
"""
Analyze parse errors across all W&B downloaded episode results.

Scans wandb_downloads* directories, reads summary.json for each episode,
and produces:
  - parse_error_episode_details.csv   (episode-level)
  - parse_error_by_condition.csv      (modality × formula × setup × model)
  - parse_error_by_modality.csv       (modality-level)
  - rerun_targets.csv                 (prioritised rerun recommendations)
  - parse_error_rerun_recommendation.md

Usage:
  python scripts/analyze_parse_errors_by_condition.py \
      --project_dir /home/lly/projects/project \
      --output_dir  /home/lly/projects/project/parse_error_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Map wandb_downloads_* directory suffix → canonical modality label.
# Order matters for prefix matching (longer suffix first is safer).
_ROOT_MODALITY: Dict[str, str] = {
    "wandb_downloads_text_only":                         "Text-only",
    "wandb_downloads_text_image_chart_only_action_text": "Text+image",
    "wandb_downloads_text_image_chart_only2":            "Chart-only",
    "wandb_downloads_image_only":                        "Simulation-only",
    "wandb_downloads_student_simulation_5projects":      "Student-simulation",
    "wandb_downloads_student_simulation":                "Student-simulation",
}

# Artifact name prefix tokens → modality override (for cross-check).
_PREFIX_MODALITY: Dict[str, str] = {
    "text_only":              "Text-only",
    "text_normal":            "Student-simulation",  # from student_simulation_5projects
    "text_student":           "Student-simulation",
    "chart_only_no_action":   "Chart-only",
    "chart_only":             "Text+image",
    "chart_normal":           "Student-simulation",
    "chart_student":          "Student-simulation",
    "image_only":             "Simulation-only",
    "simulation_normal":      "Student-simulation",
    "student":                "Student-simulation",
}

_FORMULAS = [
    "beers_wavelength",  # must come before "beers"
    "beers", "concentration", "ohm", "resistors",
    "kinematics", "mass", "distance",
]

_SETUPS = ["concrete", "abbrev", "abstract"]

_KNOWN_MODELS = [
    "model_qwen25_vl_3b", "model_qwen25_vl_7b",
    "model_qwen25_3b",    "model_qwen25_7b",
    "model_qwen35_4b",    "model_qwen35_9b",
    "model_llama31_8b",   "model_mistral_7b", "model_gemma3_4b",
]

# Compact model aliases used in some older run-names (no underscores)
_MODEL_ALIASES: Dict[str, str] = {
    "qwen25vl3b": "model_qwen25_vl_3b",
    "qwen25vl7b": "model_qwen25_vl_7b",
}

# parse_error_rate thresholds for rerun priority
_P1_RATE  = 0.05   # > 5% → Priority 1
_P1_COUNT = 2      # OR ≥ 2 episodes → Priority 1
_P2_RATE  = 0.0    # > 0% but ≤ 5% → Priority 2

# Suggested n_runs per priority
_SUGG_RUNS = {1: 30, 2: 10, 3: 0}


# ──────────────────────────────────────────────────────────────────────────────
# Path utilities
# ──────────────────────────────────────────────────────────────────────────────

def normalize_path(path_str: str) -> Path:
    """
    Convert Windows UNC or file:// URI to a local Path for exists-checking.
    \\wsl.localhost\\Ubuntu\\home\\lly\\... → /home/lly/...
    file:///home/lly/...                  → /home/lly/...
    """
    if not isinstance(path_str, str):
        return Path(str(path_str))
    if path_str.lower().startswith("file://"):
        parsed = urlparse(path_str)
        return Path(unquote(parsed.path))
    unc = re.match(r"^\\\\[^\\]+\\[^\\]+(.*)", path_str)
    if unc:
        return Path(unc.group(1).replace("\\", "/"))
    return Path(path_str)


# ──────────────────────────────────────────────────────────────────────────────
# Metadata inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def infer_formula(text: str) -> Optional[str]:
    t = text.lower()
    for f in _FORMULAS:
        if f in t:
            return f
    return None


def infer_setup(text: str) -> Optional[str]:
    t = text.lower()
    for s in _SETUPS:
        if re.search(rf"(^|[_\-/]){s}($|[_\-/\.\-])", t):
            return s
    return None


def infer_model(text: str) -> Optional[str]:
    for m in _KNOWN_MODELS:
        if m in text:
            return m
    t = text.lower()
    for alias, canonical in _MODEL_ALIASES.items():
        if alias in t:
            return canonical
    return None


def infer_modality_from_root(root_name: str) -> str:
    return _ROOT_MODALITY.get(root_name, "Unknown")


def infer_modality_from_prefix(artifact_name: str) -> Optional[str]:
    """
    Try to derive modality from the artifact directory name prefix.
    Uses longest-match so 'chart_only_no_action' wins over 'chart_only'.
    """
    name = artifact_name.lower()
    # Strip 'episode-' prefix if present
    name = re.sub(r"^episode-", "", name)
    for prefix in sorted(_PREFIX_MODALITY.keys(), key=len, reverse=True):
        if name.startswith(prefix):
            return _PREFIX_MODALITY[prefix]
    return None


def parse_output_log(log_path: Optional[Path]) -> Dict[str, Optional[str]]:
    """Extract metadata from output.log (env/model/run_name/observation_mode)."""
    result: Dict[str, Optional[str]] = {
        "log_env_config":        None,
        "log_model_config":      None,
        "log_run_name":          None,
        "log_observation_mode":  None,
        "log_task_mode":         None,
    }
    if log_path is None or not log_path.exists():
        return result
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return result
    patterns = {
        "log_env_config":       r"Env override:\s*(.+)",
        "log_model_config":     r"Model override:\s*(.+)",
        "log_run_name":         r"Run name:\s*(.+)",
        "log_observation_mode": r"Observation mode:\s*(.+)",
        "log_task_mode":        r"Task mode:\s*(.+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            result[key] = m.group(1).strip()
    return result


def find_output_log(summary_path: Path, max_up: int = 6) -> Optional[Path]:
    """Walk up from summary_path's parent to find output.log."""
    d = summary_path.parent
    for _ in range(max_up):
        candidate = d / "output.log"
        if candidate.exists():
            return candidate
        if d == d.parent:
            break
        d = d.parent
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Episode record builder
# ──────────────────────────────────────────────────────────────────────────────

def build_episode_record(
    summary_path: Path,
    root_name: str,
    project_dir: Path,
) -> Dict[str, Any]:
    """
    Build a flat dict describing one episode from its summary.json.
    """
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc), "episode_path": str(summary_path.parent)}

    artifact_name = summary_path.parent.name   # e.g. episode-text_only_beers_concrete_model_qwen25_vl_3b-03_latest
    combined_text = f"{root_name} {artifact_name}"

    # ── modality ──────────────────────────────────────────────────────────────
    modality = infer_modality_from_prefix(artifact_name)
    if not modality:
        modality = infer_modality_from_root(root_name)

    # ── formula / setup / model ───────────────────────────────────────────────
    formula = infer_formula(artifact_name)
    setup   = infer_setup(artifact_name)
    model   = infer_model(artifact_name)

    # ── output.log enrichment (lazy — only when formula/setup/model incomplete) ─
    log_info: Dict[str, Optional[str]] = {}
    if formula is None or setup is None or model is None:
        log_path = find_output_log(summary_path)
        log_info = parse_output_log(log_path)
        combined_text += " " + " ".join(v for v in log_info.values() if v)
        if formula is None:
            formula = infer_formula(log_info.get("log_env_config") or combined_text)
        if setup is None:
            setup   = infer_setup(log_info.get("log_env_config") or combined_text)
        if model is None:
            model   = infer_model(log_info.get("log_model_config") or combined_text)
        if modality == "Unknown" and log_info.get("log_observation_mode"):
            obs = log_info["log_observation_mode"].lower()
            if "simulation" in obs:
                modality = "Simulation-only"
            elif "text_image" in obs:
                modality = "Text+image"
            elif "text" in obs:
                modality = "Text-only"

    # ── parse_error detection ──────────────────────────────────────────────────
    raw_parse_error = summary.get("parse_error")
    evaluation      = summary.get("evaluation") or {}
    term_reason     = evaluation.get("termination_reason", "")

    has_parse_error = bool(
        (raw_parse_error and raw_parse_error not in ("", False, None))
        or term_reason == "parse_error"
    )

    # Short error message (first 200 chars of parse_error field)
    parse_error_msg = ""
    if raw_parse_error and isinstance(raw_parse_error, str):
        parse_error_msg = raw_parse_error[:200]
    elif term_reason == "parse_error":
        err_msg = evaluation.get("error_message") or ""
        parse_error_msg = err_msg[:200]

    # ── other outcome fields ───────────────────────────────────────────────────
    finish_reached  = summary.get("finish_reached", False)
    forced_finish   = summary.get("forced_finish", False)
    final_equation  = summary.get("final_equation") or evaluation.get("final_equation")
    num_steps       = summary.get("num_steps", 0)

    # run_name derived from artifact_name
    run_name = re.sub(r"^episode-", "", artifact_name)
    run_name = re.sub(r"-\d+_latest$", "", run_name)

    return {
        "episode_path":       str(summary_path.parent.relative_to(project_dir)),
        "run_name":           run_name,
        "root":               root_name,
        "modality":           modality,
        "formula":            formula or "unknown",
        "setup":              setup   or "unknown",
        "model":              model   or "unknown",
        "has_parse_error":    has_parse_error,
        "failed_step":        num_steps if has_parse_error else None,
        "parse_error_message": parse_error_msg,
        "finish_reached":     finish_reached,
        "forced_finish":      forced_finish,
        "final_equation":     final_equation,
        "termination_reason": term_reason or ("parse_error" if has_parse_error else ""),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main scan
# ──────────────────────────────────────────────────────────────────────────────

def scan_episodes(project_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan all wandb_downloads* directories for summary.json files.
    Returns a list of episode record dicts.
    """
    episodes: List[Dict[str, Any]] = []

    roots = sorted(
        d for d in project_dir.iterdir()
        if d.is_dir() and d.name.startswith("wandb_downloads")
        and "visual_benchmark" not in d.name   # skip benchmark directory
    )

    for root_dir in roots:
        root_name = root_dir.name
        summary_files = list(root_dir.rglob("summary.json"))
        print(f"  {root_name}: {len(summary_files)} episodes", flush=True)

        for sf in summary_files:
            rec = build_episode_record(sf, root_name, project_dir)
            if "_error" not in rec:
                episodes.append(rec)
            else:
                print(f"    WARN: failed to parse {sf}: {rec['_error']}", flush=True)

    return episodes


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _agg_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    pe = sum(1 for r in rows if r["has_parse_error"])
    return {
        "total_episodes":             n,
        "parse_error_episodes":       pe,
        "parse_error_rate":           round(pe / n, 4) if n else 0.0,
        "finish_reached_count":       sum(1 for r in rows if r["finish_reached"]),
        "forced_finish_count":        sum(1 for r in rows if r["forced_finish"]),
        "final_equation_non_null_count": sum(1 for r in rows if r["final_equation"]),
    }


def aggregate_by_condition(
    episodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List] = defaultdict(list)
    for ep in episodes:
        key = (ep["modality"], ep["formula"], ep["setup"], ep["model"])
        groups[key].append(ep)

    rows = []
    for (modality, formula, setup, model), eps in sorted(groups.items()):
        agg = _agg_rows(eps)
        rows.append({
            "modality": modality, "formula": formula,
            "setup": setup,       "model": model,
            **agg,
        })
    return rows


def aggregate_by_modality(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List] = defaultdict(list)
    for ep in episodes:
        groups[ep["modality"]].append(ep)

    rows = []
    for modality, eps in sorted(groups.items()):
        agg = _agg_rows(eps)
        rows.append({
            "modality": modality,
            "total_episodes":       agg["total_episodes"],
            "parse_error_episodes": agg["parse_error_episodes"],
            "parse_error_rate":     agg["parse_error_rate"],
        })
    return rows


def build_rerun_targets(condition_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    targets = []
    for row in condition_rows:
        pe_rate  = row["parse_error_rate"]
        pe_count = row["parse_error_episodes"]
        total    = row["total_episodes"]

        if pe_rate > _P1_RATE or pe_count >= _P1_COUNT:
            priority = 1
            reason   = (
                f"parse_error_rate={pe_rate:.1%} (>{_P1_RATE:.0%}) or "
                f"parse_error_episodes={pe_count} (≥{_P1_COUNT})"
            )
        elif pe_count > 0:
            priority = 2
            reason   = f"parse_error_rate={pe_rate:.1%} (low but non-zero)"
        else:
            priority = 3
            reason   = "no parse errors observed"

        targets.append({
            "priority":             priority,
            "modality":             row["modality"],
            "formula":              row["formula"],
            "setup":                row["setup"],
            "model":                row["model"],
            "total_episodes":       total,
            "parse_error_episodes": pe_count,
            "parse_error_rate":     pe_rate,
            "suggested_n_runs":     _SUGG_RUNS[priority],
            "reason":               reason,
        })

    targets.sort(key=lambda r: (r["priority"], -r["parse_error_rate"]))
    return targets


# ──────────────────────────────────────────────────────────────────────────────
# CSV writers
# ──────────────────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────

def write_markdown_report(
    output_dir: Path,
    episodes: List[Dict[str, Any]],
    by_modality: List[Dict[str, Any]],
    by_condition: List[Dict[str, Any]],
    rerun_targets: List[Dict[str, Any]],
) -> None:

    total     = len(episodes)
    total_pe  = sum(1 for e in episodes if e["has_parse_error"])
    overall_rate = total_pe / total if total else 0.0

    # Top 10 conditions by parse_error_rate (among those with at least 1 error)
    top10 = sorted(
        [r for r in by_condition if r["parse_error_episodes"] > 0],
        key=lambda r: (-r["parse_error_rate"], -r["parse_error_episodes"])
    )[:10]

    p1 = [r for r in rerun_targets if r["priority"] == 1]
    p2 = [r for r in rerun_targets if r["priority"] == 2]

    # Missing fields summary
    unknown_formula  = sum(1 for e in episodes if e["formula"]  == "unknown")
    unknown_setup    = sum(1 for e in episodes if e["setup"]    == "unknown")
    unknown_model    = sum(1 for e in episodes if e["model"]    == "unknown")
    unknown_modality = sum(1 for e in episodes if e["modality"] == "Unknown")

    lines: List[str] = [
        "# Parse Error Analysis — Rerun Recommendation",
        "",
        f"**Total episodes scanned:** {total}  ",
        f"**Episodes with parse error:** {total_pe} ({overall_rate:.1%})",
        "",
        "---",
        "",
        "## 1. Parse Error by Modality",
        "",
        "| Modality | Total | Parse Errors | Rate |",
        "|---|---:|---:|---:|",
    ]
    for row in by_modality:
        lines.append(
            f"| {row['modality']} | {row['total_episodes']} "
            f"| {row['parse_error_episodes']} | {row['parse_error_rate']:.1%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Top 10 Conditions by Parse Error Rate",
        "",
        "| Modality | Formula | Setup | Model | Total | PE | Rate |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in top10:
        lines.append(
            f"| {row['modality']} | {row['formula']} | {row['setup']} "
            f"| {row['model']} | {row['total_episodes']} "
            f"| {row['parse_error_episodes']} | {row['parse_error_rate']:.1%} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Rerun Recommendations",
        "",
        "**Priority rules:**",
        f"- **Priority 1** (must rerun): parse_error_rate > {_P1_RATE:.0%} OR "
        f"parse_error_episodes ≥ {_P1_COUNT} → suggested_n_runs = {_SUGG_RUNS[1]}",
        f"- **Priority 2** (consider rerun): 0 < parse_error_rate ≤ {_P1_RATE:.0%} "
        f"→ suggested_n_runs = {_SUGG_RUNS[2]}",
        f"- **Priority 3**: no parse errors → no rerun needed",
        "",
        f"### Priority 1 — {len(p1)} condition(s)",
        "",
    ]
    if p1:
        lines += [
            "| Modality | Formula | Setup | Model | Total | PE | Rate | Suggested N |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
        for row in p1:
            lines.append(
                f"| {row['modality']} | {row['formula']} | {row['setup']} "
                f"| {row['model']} | {row['total_episodes']} "
                f"| {row['parse_error_episodes']} | {row['parse_error_rate']:.1%} "
                f"| {row['suggested_n_runs']} |"
            )
    else:
        lines.append("*No Priority 1 conditions.*")

    lines += [
        "",
        f"### Priority 2 — {len(p2)} condition(s)",
        "",
    ]
    if p2:
        lines += [
            "| Modality | Formula | Setup | Model | Total | PE | Rate | Suggested N |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
        for row in p2:
            lines.append(
                f"| {row['modality']} | {row['formula']} | {row['setup']} "
                f"| {row['model']} | {row['total_episodes']} "
                f"| {row['parse_error_episodes']} | {row['parse_error_rate']:.1%} "
                f"| {row['suggested_n_runs']} |"
            )
    else:
        lines.append("*No Priority 2 conditions.*")

    lines += [
        "",
        "---",
        "",
        "## 4. Data Quality Notes",
        "",
        "Fields that could not be reliably extracted from some episodes:",
        "",
        f"- **formula** unknown in {unknown_formula} episodes",
        f"- **setup** unknown in {unknown_setup} episodes",
        f"- **model** unknown in {unknown_model} episodes",
        f"- **modality** unknown in {unknown_modality} episodes",
        "",
        "These episodes appear in the `(unknown)` groups of the condition CSV.",
        "If unknown counts are high, check artifact naming conventions or",
        "add entries to the `_ROOT_MODALITY` / `_PREFIX_MODALITY` maps in the script.",
        "",
        "---",
        "",
        "*Generated by `scripts/analyze_parse_errors_by_condition.py`*",
    ]

    md_path = output_dir / "parse_error_rerun_recommendation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {md_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze parse errors across W&B episode results."
    )
    parser.add_argument(
        "--project_dir",
        default="/home/lly/projects/project",
        help="Root project directory (contains wandb_downloads* subdirs).",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/lly/projects/project/parse_error_analysis",
        help="Directory to write output CSVs and Markdown report.",
    )
    args = parser.parse_args()

    project_dir = normalize_path(args.project_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Parse error analysis")
    print(f"  project_dir : {project_dir}")
    print(f"  output_dir  : {output_dir}")
    print(f"{'='*60}\n")

    # ── 1. Scan ───────────────────────────────────────────────────────────────
    print("Scanning episodes...")
    episodes = scan_episodes(project_dir)
    print(f"\nTotal episodes found: {len(episodes)}")
    total_pe = sum(1 for e in episodes if e["has_parse_error"])
    print(f"Parse error episodes: {total_pe} ({total_pe/len(episodes):.1%})\n")

    # ── 2. Episode-level CSV ──────────────────────────────────────────────────
    ep_fields = [
        "episode_path", "run_name", "modality", "formula", "setup", "model",
        "has_parse_error", "failed_step", "parse_error_message",
        "finish_reached", "forced_finish", "final_equation",
    ]
    write_csv(output_dir / "parse_error_episode_details.csv", episodes, ep_fields)

    # ── 3. Condition-level CSV ────────────────────────────────────────────────
    by_condition = aggregate_by_condition(episodes)
    cond_fields  = [
        "modality", "formula", "setup", "model",
        "total_episodes", "parse_error_episodes", "parse_error_rate",
        "finish_reached_count", "forced_finish_count", "final_equation_non_null_count",
    ]
    write_csv(output_dir / "parse_error_by_condition.csv", by_condition, cond_fields)

    # ── 4. Modality-level CSV ─────────────────────────────────────────────────
    by_modality   = aggregate_by_modality(episodes)
    modal_fields  = ["modality", "total_episodes", "parse_error_episodes", "parse_error_rate"]
    write_csv(output_dir / "parse_error_by_modality.csv", by_modality, modal_fields)

    # ── 5. Rerun targets CSV ──────────────────────────────────────────────────
    rerun_targets = build_rerun_targets(by_condition)
    rerun_fields  = [
        "priority", "modality", "formula", "setup", "model",
        "total_episodes", "parse_error_episodes", "parse_error_rate",
        "suggested_n_runs", "reason",
    ]
    write_csv(output_dir / "rerun_targets.csv", rerun_targets, rerun_fields)

    # ── 6. Markdown report ────────────────────────────────────────────────────
    write_markdown_report(output_dir, episodes, by_modality, by_condition, rerun_targets)

    # ── 7. Console summary ────────────────────────────────────────────────────
    print("\n=== Parse Error by Modality ===")
    for row in by_modality:
        print(f"  {row['modality']:30s}  {row['parse_error_episodes']:4d}/{row['total_episodes']:<4d}  "
              f"({row['parse_error_rate']:.1%})")

    print("\n=== Top 10 Conditions by Parse Error Rate ===")
    top10 = sorted(
        [r for r in by_condition if r["parse_error_episodes"] > 0],
        key=lambda r: (-r["parse_error_rate"], -r["parse_error_episodes"])
    )[:10]
    for row in top10:
        print(f"  {row['modality']:18s} {row['formula']:20s} {row['setup']:10s} "
              f"{row['model']:22s}  {row['parse_error_episodes']}/{row['total_episodes']}  "
              f"({row['parse_error_rate']:.1%})")

    p1 = [r for r in rerun_targets if r["priority"] == 1]
    print(f"\n=== Priority 1 Rerun Targets ({len(p1)} conditions) ===")
    for row in p1:
        print(f"  {row['modality']:18s} {row['formula']:20s} {row['setup']:10s} "
              f"{row['model']:22s}  PE={row['parse_error_episodes']}/{row['total_episodes']}  "
              f"rate={row['parse_error_rate']:.1%}  n_runs={row['suggested_n_runs']}")

    print(f"\nAll outputs written to: {output_dir}")
    print("Done.\n")


if __name__ == "__main__":
    main()
