"""
Utilities for image path normalization and existence validation.

normalize_image_path_for_exists_check()
    Converts any path format (file:// URI, Windows UNC, POSIX absolute) to a
    local POSIX path suitable for Path.exists().  The original path is never
    modified — this function is only used for the exists check, not for
    modifying what gets passed to the VLM or written to logs.

validate_image_paths()
    Checks each path in a list and returns a structured validation record
    including per-path exists flags and a human-readable error message.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse


def normalize_image_path_for_exists_check(path: str) -> str:
    """
    Convert any image path format to a local POSIX filesystem path so
    Path.exists() can be used for validation.

    Supported input formats
    -----------------------
    1. POSIX absolute path:          /home/lly/.../step_0000.png
    2. file:/// URI:                 file:///home/lly/.../step_0000.png
    3. file://host/ URI (rare):      file://localhost/home/lly/.../step_0000.png
    4. Windows UNC (WSL gateway):    \\\\wsl.localhost\\Ubuntu\\home\\lly\\...

    The returned value is used ONLY for Path.exists().  The original path
    string is preserved in all logs and passed unchanged to the VLM.

    Parameters
    ----------
    path : str
        Raw image path in any of the above formats.

    Returns
    -------
    str
        A local POSIX path (may not exist; existence is the caller's concern).
        Returns the input unchanged if no known prefix is detected.
    """
    if not isinstance(path, str) or not path:
        return path

    # Case 1: file:// URI — use urlparse for correctness
    if path.lower().startswith("file://"):
        parsed = urlparse(path)
        # parsed.path is the POSIX path component (already starts with /).
        # unquote decodes percent-encoded chars, e.g. %20 -> space.
        return unquote(parsed.path) if parsed.path else path

    # Case 2: Windows UNC path via WSL gateway
    # Pattern: \\server\share\rest  (two leading backslashes)
    # Example: \\wsl.localhost\Ubuntu\home\lly\projects\...
    unc_match = re.match(r"^\\\\[^\\]+\\[^\\]+(.*)", path)
    if unc_match:
        linux_path = unc_match.group(1).replace("\\", "/")
        return linux_path or "/"

    # Case 3: Already a POSIX path (absolute or relative)
    return path


def validate_image_paths(
    image_paths: Optional[List[str]],
    *,
    requires_images: bool,
    step_id: int,
) -> Dict[str, Any]:
    """
    Validate that each image path in the list points to an existing file.

    Parameters
    ----------
    image_paths : list of str or None
        Paths as passed to agent.act().  May be None or empty.
    requires_images : bool
        True when the current episode condition (chart-only, text+image,
        simulation-only) expects at least one image per step.
        Used to generate a warning when image_paths is empty.
    step_id : int
        Current step index, included in the returned record for traceability.

    Returns
    -------
    dict with keys:
        step_id          int
        requires_images  bool
        image_paths      List[str]   original paths (may be empty)
        image_exists     List[bool]  one flag per path
        missing_paths    List[str]   subset that do not exist on disk
        image_error      Optional[str]  human-readable problem, or None
        has_image_error  bool
    """
    paths: List[str] = list(image_paths) if image_paths else []
    exists_flags: List[bool] = []
    missing: List[str] = []

    for p in paths:
        local = normalize_image_path_for_exists_check(p)
        try:
            exists = Path(local).exists()
        except Exception:
            exists = False
        exists_flags.append(exists)
        if not exists:
            missing.append(p)

    image_error: Optional[str] = None
    has_image_error = False

    if requires_images and not paths:
        image_error = (
            f"[image_missing] step_id={step_id}: image-required condition "
            "but image_paths is empty — VLM received no images this step"
        )
        has_image_error = True
    elif missing:
        image_error = (
            f"[image_missing] step_id={step_id}: "
            f"{len(missing)}/{len(paths)} image file(s) not found on disk: "
            f"{missing}"
        )
        has_image_error = True

    return {
        "step_id":         step_id,
        "requires_images": requires_images,
        "image_paths":     paths,
        "image_exists":    exists_flags,
        "missing_paths":   missing,
        "image_error":     image_error,
        "has_image_error": has_image_error,
    }
