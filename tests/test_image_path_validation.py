"""
Tests for image path normalization and existence validation.

Run with:
    cd /home/lly/projects/project
    python tests/test_image_path_validation.py
or:
    python -m pytest tests/test_image_path_validation.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.image_path_utils import (
    normalize_image_path_for_exists_check,
    validate_image_paths,
)


# ---------------------------------------------------------------------------
# normalize_image_path_for_exists_check — path format conversion
# ---------------------------------------------------------------------------

def test_posix_path_unchanged():
    path = "/home/lly/projects/project/outputs/run_00/images/step_0000.png"
    assert normalize_image_path_for_exists_check(path) == path
    print("PASS test_posix_path_unchanged")


def test_file_triple_slash_uri():
    """file:///home/lly/... -> /home/lly/..."""
    uri = "file:///home/lly/projects/project/outputs/run_00/images/step_0000.png"
    result = normalize_image_path_for_exists_check(uri)
    assert result == "/home/lly/projects/project/outputs/run_00/images/step_0000.png", (
        f"Got: {result!r}"
    )
    print("PASS test_file_triple_slash_uri")


def test_windows_unc_wsl_path():
    r"""\\wsl.localhost\Ubuntu\home\lly\... -> /home/lly/..."""
    unc = r"\\wsl.localhost\Ubuntu\home\lly\projects\project\outputs\run_00\images\step_0000.png"
    result = normalize_image_path_for_exists_check(unc)
    assert result == "/home/lly/projects/project/outputs/run_00/images/step_0000.png", (
        f"Got: {result!r}"
    )
    print("PASS test_windows_unc_wsl_path")


def test_file_uri_with_spaces():
    """file:// URI with a space in the filename (%20 percent-encoding) resolves correctly."""
    with tempfile.TemporaryDirectory() as td:
        # Create a file whose name contains a space
        png = Path(td) / "step 0000.png"
        png.write_bytes(b"\x89PNG")

        uri = png.as_uri()          # e.g. file:///tmp/.../step%200000.png
        assert "%20" in uri or " " not in uri, (
            f"Expected percent-encoded space in URI, got: {uri!r}"
        )

        normalized = normalize_image_path_for_exists_check(uri)
        assert Path(normalized).exists(), (
            f"normalized path does not exist: {normalized!r}  (from URI: {uri!r})"
        )
        assert " " in normalized, (
            f"Expected space in normalized path, got: {normalized!r}"
        )
    print("PASS test_file_uri_with_spaces")


# ---------------------------------------------------------------------------
# validate_image_paths — existence checks and error messages
# ---------------------------------------------------------------------------

def test_existing_path_returns_true():
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "step_0000.png"
        png.write_bytes(b"\x89PNG")

        result = validate_image_paths([str(png)], requires_images=True, step_id=0)

        assert result["image_exists"] == [True]
        assert result["missing_paths"] == []
        assert result["has_image_error"] is False
        assert result["image_error"] is None
    print("PASS test_existing_path_returns_true")


def test_missing_path_returns_false():
    with tempfile.TemporaryDirectory() as td:
        missing = str(Path(td) / "nonexistent.png")

        result = validate_image_paths([missing], requires_images=True, step_id=1)

        assert result["image_exists"] == [False]
        assert missing in result["missing_paths"]
        assert result["has_image_error"] is True
        assert "image_missing" in result["image_error"]
        assert "step_id=1" in result["image_error"]
    print("PASS test_missing_path_returns_false")


def test_empty_image_paths_requires_images_true():
    """Empty list + requires_images=True -> has_image_error=True."""
    result = validate_image_paths([], requires_images=True, step_id=2)
    assert result["has_image_error"] is True
    assert "image_missing" in result["image_error"]
    print("PASS test_empty_image_paths_requires_images_true")


def test_empty_image_paths_requires_images_false():
    """Empty list + requires_images=False -> has_image_error=False (text-only run)."""
    result = validate_image_paths([], requires_images=False, step_id=3)
    assert result["has_image_error"] is False
    assert result["image_error"] is None
    print("PASS test_empty_image_paths_requires_images_false")


# ---------------------------------------------------------------------------
# EpisodeLogger.save_episode — summary.json and interaction_log.json
# ---------------------------------------------------------------------------

def test_logger_writes_summary_and_interaction_log():
    """
    save_episode() with an image_validation_log entry must:
    a. write image_error_count and has_image_error to summary.json
    b. write at least one entry_type='image_validation' to interaction_log.json
    c. that entry contains step_id, image_paths, image_exists, image_error, has_image_error
    """
    from src.tracing.logger import EpisodeLogger

    with tempfile.TemporaryDirectory() as td:
        logger = EpisodeLogger(output_dir=td)

        fake_result = {
            "steps": [],
            "trajectory": [],
            "final_equation": None,
            "finish_reason": None,
            "finish_reached": False,
            "finish_step_id": None,
            "num_steps": 0,
            "parse_error": None,
            "forced_finish": False,
            "image_paths": [],
            "parse_error_attempts": [],
            "image_validation_log": [
                {
                    "step_id": 0,
                    "requires_images": True,
                    "image_paths": [],
                    "image_exists": [],
                    "missing_paths": [],
                    "image_error": (
                        "[image_missing] step_id=0: image-required condition "
                        "but image_paths is empty — VLM received no images this step"
                    ),
                    "has_image_error": True,
                }
            ],
            "image_error_count": 1,
            "has_image_error": True,
        }

        saved = logger.save_episode(fake_result)

        # ── a. summary.json ────────────────────────────────────────────────
        summary = json.loads(Path(saved["summary"]).read_text(encoding="utf-8"))
        assert summary["image_error_count"] == 1, (
            f"Expected image_error_count=1, got {summary.get('image_error_count')}"
        )
        assert summary["has_image_error"] is True

        # ── b. interaction_log.json has image_validation entry ─────────────
        ilog = json.loads(Path(saved["interaction_log"]).read_text(encoding="utf-8"))
        iv_entries = [e for e in ilog if e.get("entry_type") == "image_validation"]
        assert len(iv_entries) == 1, (
            f"Expected 1 image_validation entry, got {len(iv_entries)}"
        )

        # ── c. required fields present in the image_validation entry ──────
        iv = iv_entries[0]
        for field in ("step_id", "image_paths", "image_exists", "image_error", "has_image_error"):
            assert field in iv, f"Missing field {field!r} in image_validation entry"
        assert iv["step_id"] == 0
        assert iv["has_image_error"] is True
        assert "image_missing" in iv["image_error"]

        print(f"PASS test_logger_writes_summary_and_interaction_log")
        print(f"  image_validation entry fields: {list(iv.keys())}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_posix_path_unchanged()
    test_file_triple_slash_uri()
    test_windows_unc_wsl_path()
    test_file_uri_with_spaces()
    test_existing_path_returns_true()
    test_missing_path_returns_false()
    test_empty_image_paths_requires_images_true()
    test_empty_image_paths_requires_images_false()
    test_logger_writes_summary_and_interaction_log()
    print("\nAll tests passed.")
