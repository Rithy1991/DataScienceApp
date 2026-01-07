from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelRecord:
    model_id: str
    created_utc: str
    kind: str  # tabular|forecast
    name: str
    artifact_path: str
    meta: Dict[str, Any]


def _utcnow_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def init_dirs(registry_dir: str, artifacts_dir: str) -> None:
    Path(registry_dir).mkdir(parents=True, exist_ok=True)
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)


def register_model(
    registry_dir: str,
    artifacts_dir: str,
    kind: str,
    name: str,
    artifact_path: str,
    meta: Dict[str, Any],
) -> ModelRecord:
    init_dirs(registry_dir, artifacts_dir)
    model_id = _utcnow_id(kind)

    src = Path(artifact_path)
    if not src.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    dest = Path(artifacts_dir) / f"{model_id}{src.suffix}"
    shutil.copy2(src, dest)

    record = ModelRecord(
        model_id=model_id,
        created_utc=datetime.now(timezone.utc).isoformat(),
        kind=kind,
        name=name,
        artifact_path=str(dest.as_posix()),
        meta=meta,
    )

    (Path(registry_dir) / f"{model_id}.json").write_text(json.dumps(record.__dict__, indent=2), encoding="utf-8")
    return record


def list_models(registry_dir: str) -> List[ModelRecord]:
    p = Path(registry_dir)
    if not p.exists():
        return []
    records: List[ModelRecord] = []
    for f in sorted(p.glob("*.json"), reverse=True):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            records.append(
                ModelRecord(
                    model_id=str(raw.get("model_id")),
                    created_utc=str(raw.get("created_utc")),
                    kind=str(raw.get("kind")),
                    name=str(raw.get("name")),
                    artifact_path=str(raw.get("artifact_path")),
                    meta=dict(raw.get("meta") or {}),
                )
            )
        except Exception:
            continue
    return records


def get_model(registry_dir: str, model_id: str) -> Optional[ModelRecord]:
    f = Path(registry_dir) / f"{model_id}.json"
    if not f.exists():
        return None
    raw = json.loads(f.read_text(encoding="utf-8"))
    return ModelRecord(
        model_id=str(raw.get("model_id")),
        created_utc=str(raw.get("created_utc")),
        kind=str(raw.get("kind")),
        name=str(raw.get("name")),
        artifact_path=str(raw.get("artifact_path")),
        meta=dict(raw.get("meta") or {}),
    )


def delete_model(registry_dir: str, model_id: str) -> bool:
    rec = get_model(registry_dir, model_id)
    if not rec:
        return False

    try:
        Path(rec.artifact_path).unlink(missing_ok=True)  # py3.8+: exists; mac likely 3.11
    except TypeError:
        # compatibility
        if Path(rec.artifact_path).exists():
            Path(rec.artifact_path).unlink()

    meta_path = Path(registry_dir) / f"{model_id}.json"
    if meta_path.exists():
        meta_path.unlink()
    return True
