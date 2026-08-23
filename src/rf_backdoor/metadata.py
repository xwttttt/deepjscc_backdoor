"""Run provenance and artifact hashing."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_hash(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpu_info() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return "; ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    except Exception:  # pragma: no cover - environment-specific
        pass
    return None


def collect_run_metadata(root: str | Path, *, upstream_path: str | Path | None = None, seed: int | None = None, packages: Iterable[str] = ("numpy", "torch", "torchvision")) -> dict[str, Any]:
    root_path = Path(root).resolve()
    dependencies = {}
    for package in packages:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_git_commit": _git_hash(root_path),
        "upstream_git_commit": _git_hash(Path(upstream_path).resolve()) if upstream_path else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": _gpu_info(),
        "seed": seed,
        "dependencies": dependencies,
    }


def write_run_metadata(path: str | Path, metadata: dict[str, Any], *, artifacts: Iterable[str | Path] = ()) -> None:
    payload = dict(metadata)
    payload["artifact_sha256"] = {str(Path(item)): sha256_file(item) for item in artifacts}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

