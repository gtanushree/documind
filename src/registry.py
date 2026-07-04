from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_use_tmp = bool(os.getenv("VERCEL") or os.getenv("DOCUMIND_TMP_REGISTRY"))
 
REGISTRY_PATH: Path = (
    Path("/tmp/registry.json")
    if _use_tmp
    else Path(__file__).resolve().parent.parent / "data" / "registry.json"
)
 
_lock = Lock()

#REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "registry.json"
#_lock = Lock()


def _ensure_file() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text(json.dumps({}))


def _read() -> dict:
    _ensure_file()
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def _write(data: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_namespaces() -> list[str]:
    with _lock:
        return sorted(_read().keys())


def get_namespace_info(namespace: str) -> dict:
    with _lock:
        return _read().get(namespace, {"files": [], "created_at": None})


def create_namespace(namespace: str) -> None:
    with _lock:
        data = _read()
        data.setdefault(namespace, {"files": [], "created_at": _now()})
        _write(data)


def register_files(namespace: str, files: list[dict]) -> None:
    """files: list of {"name": str, "chunks": int, "pages": int}"""
    with _lock:
        data = _read()
        entry = data.setdefault(namespace, {"files": [], "created_at": _now()})
        existing_names = {f["name"] for f in entry["files"]}
        for f in files:
            if f["name"] in existing_names:
                # Re-indexed file: drop the old record, keep the new one.
                entry["files"] = [x for x in entry["files"] if x["name"] != f["name"]]
            entry["files"].append({**f, "indexed_at": _now()})
        _write(data)


def delete_namespace(namespace: str) -> None:
    with _lock:
        data = _read()
        data.pop(namespace, None)
        _write(data)
