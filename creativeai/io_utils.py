from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from creativeai.schemas import RunManifest, utc_now_iso



def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out



def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")



def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)



def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)



def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()



def resolve_git_hash(cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return "nogit"


def resolve_git_full_hash(cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except Exception:
        return "nogit"


def git_dirty(cwd: str | Path | None = None) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_snapshot() -> dict[str, Any]:
    package_names = ["creativeai", "llama-cpp-python", "numpy", "pandas", "pyarrow", "torch", "sentence-transformers"]
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {name: _package_version(name) for name in package_names},
        "embedding_backend_env": os.environ.get("CREATIVEAI_EMBEDDING_BACKEND", ""),
        "require_semantic_env": os.environ.get("CREATIVEAI_REQUIRE_SEMANTIC", ""),
    }



def build_manifest(
    run_id: str,
    quantization: str,
    backend: str,
    model_hash: str = "unknown",
    cwd: str | Path | None = None,
    session_id: str = "",
    extra: dict[str, Any] | None = None,
) -> RunManifest:
    env = environment_snapshot()
    return RunManifest(
        run_id=run_id,
        git_hash=resolve_git_hash(cwd),
        git_full_hash=resolve_git_full_hash(cwd),
        git_dirty=git_dirty(cwd),
        model_hash=model_hash,
        quantization=quantization,
        backend=backend,
        created_at_utc=utc_now_iso(),
        host=platform.node() or "unknown-host",
        session_id=session_id,
        python_version=str(env.get("python_version", "")),
        platform=str(env.get("platform", "")),
        extra={**(extra or {}), "environment": env},
    )



def snapshot_tabular(records: list[dict[str, Any]], out_path: str | Path) -> str:
    """Write a tabular snapshot. Uses parquet when pandas+pyarrow are available, JSON otherwise."""
    p = Path(out_path)
    ensure_dir(p.parent)
    try:
        import pandas as pd  # type: ignore

        df = pd.json_normalize(records)
        if p.suffix.lower() != ".parquet":
            p = p.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return str(p)
    except Exception:
        fallback = p.with_suffix(".json")
        write_json(fallback, {"records": records})
        return str(fallback)



def infer_records(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        return load_jsonl(p)
    if p.suffix.lower() == ".json":
        payload = load_json(p)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "records" in payload and isinstance(payload["records"], list):
            return payload["records"]
        return [payload]
    raise ValueError(f"Unsupported input format: {p}")



def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def token_count_text(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def token_count_from_list(items: list[str]) -> int:
    return sum(token_count_text(item) for item in items)
