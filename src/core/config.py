from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: Dict[str, Any]

    @property
    def title(self) -> str:
        return str(self.raw.get("app", {}).get("title", "Data Science & AI Studio"))

    @property
    def refresh_rate_seconds(self) -> int:
        return int(self.raw.get("app", {}).get("refresh_rate_seconds", 30))

    @property
    def registry_dir(self) -> str:
        return str(self.raw.get("models", {}).get("registry_dir", "models"))

    @property
    def artifacts_dir(self) -> str:
        return str(self.raw.get("models", {}).get("artifacts_dir", "artifacts"))

    @property
    def max_rows_preview(self) -> int:
        return int(self.raw.get("data", {}).get("max_rows_preview", 5000))

    @property
    def logging_dir(self) -> str:
        return str(self.raw.get("logging", {}).get("log_dir", "logs"))

    @property
    def logging_level(self) -> str:
        return str(self.raw.get("logging", {}).get("level", "INFO"))

    @property
    def history_db_path(self) -> str:
        return str(self.raw.get("logging", {}).get("history_db_path", "data/history.sqlite"))

    @property
    def ai_provider(self) -> str:
        return str(self.raw.get("ai", {}).get("provider", "local"))

    @property
    def ai_model(self) -> str:
        return str(self.raw.get("ai", {}).get("model", "google/flan-t5-small"))

    @property
    def ai_max_new_tokens(self) -> int:
        return int(self.raw.get("ai", {}).get("max_new_tokens", 256))

    @property
    def allow_api_ingestion(self) -> bool:
        return bool(self.raw.get("security", {}).get("allow_api_ingestion", False))

    @property
    def api_allowlist(self) -> List[str]:
        raw = self.raw.get("security", {}).get("api_allowlist", [])
        if not isinstance(raw, list):
            return []
        return [str(x).strip() for x in raw if str(x).strip()]

    @property
    def max_api_response_bytes(self) -> int:
        return int(self.raw.get("security", {}).get("max_api_response_bytes", 10 * 1024 * 1024))

    @property
    def allow_runtime_pip_install(self) -> bool:
        return bool(self.raw.get("security", {}).get("allow_runtime_pip_install", False))


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        return AppConfig(raw={})
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raw = {}
    return AppConfig(raw=raw)


def save_config(config: Dict[str, Any], config_path: str | Path = "config.yaml") -> None:
    path = Path(config_path)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
