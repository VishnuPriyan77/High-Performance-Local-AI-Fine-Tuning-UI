"""Shared helpers for CoreForge-UI."""

from __future__ import annotations

import os
import platform
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def normalize_instructional_record(record: Dict[str, Any]) -> Dict[str, str]:
    """
    Enforce strict Alpaca schema and sanitize inputs.

    Required keys:
        instruction, input, output
    """
    if not isinstance(record, dict):
        raise ValueError("Record must be a dict.")

    instruction = str(record.get("instruction", "")).strip()
    inp = str(record.get("input", "")).strip()
    output = str(record.get("output", "")).strip()

    if not instruction or not output:
        raise ValueError("Both instruction and output are required.")
    if "answer:" in inp.lower():
        inp = inp.strip()
    return {"instruction": instruction, "input": inp, "output": output}


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> Path:
    """
    Write a list of JSONL records to disk.
    """
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def stable_slug(value: str, max_length: int = 60) -> str:
    """
    Build a deterministic short filename slug from input text.
    """
    base = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())[:max_length]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{base or 'dataset'}-{digest}"


def detect_device(prefer_cuda: bool = True) -> str:
    """Return a valid torch device string when torch is available."""
    try:
        import torch
    except Exception as exc:
        if prefer_cuda:
            return "cpu"
        raise RuntimeError(f"Torch is required for device detection: {exc}") from exc

    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def platform_binary_hint() -> str:
    """Return the expected venv python binary for common OS families."""
    system = platform.system().lower()
    if system == "windows":
        return "Scripts/python.exe"
    return "bin/python"


def safe_read_text(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_bytes().decode("utf-8", errors="replace")
