"""Provenance for every experiment run.

Every runner calls `save_run(df, out_dir, config)`; it writes `<name>.parquet` plus a
`<name>.provenance.json` sidecar so any number in the write-up traces back to the exact
code, model, dtype, batch, backend, GPU, driver and library versions that produced it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("torch", "transformers", "sglang", "sglang-kernel", "sgl-kernel", "triton", "flashinfer-python", "numpy", "pandas")


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except Exception:
        return None


def git_info(repo: Path | None = None) -> dict[str, Any]:
    cwd = str(repo) if repo else None
    sha = _run(["git", "-C", cwd or ".", "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", cwd or ".", "status", "--porcelain"])
    return {"git_sha": sha, "git_dirty": bool(dirty) if dirty is not None else None}


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if smi:
        info["gpus"] = [line.strip() for line in smi.splitlines()]
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["cuda_capability"] = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return info


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def collect(config: Any = None) -> dict[str, Any]:
    cfg = asdict(config) if is_dataclass(config) else config
    cfg_json = json.dumps(cfg, sort_keys=True, default=str)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        **git_info(),
        **gpu_info(),
        "packages": package_versions(),
        "config": cfg,
        "config_hash": hashlib.sha256(cfg_json.encode()).hexdigest()[:12],
    }


def save_run(df, out_dir: str | Path, config: Any, name: str | None = None) -> Path:
    """Write df to parquet with a provenance sidecar. Returns the parquet path."""
    prov = collect(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name or f"run-{prov['config_hash']}-{prov['timestamp_utc'][:19].replace(':', '')}"
    pq = out_dir / f"{stem}.parquet"
    df.to_parquet(pq, index=False)
    (out_dir / f"{stem}.provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    return pq
