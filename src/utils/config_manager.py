import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ConfigManager:
    config: Dict[str, Any]
    config_path: Path

    @classmethod
    def from_yaml(cls, config_path: str = "config/config.yaml"):
        path = Path(config_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        cls._resolve_paths(config, base_dir=path.parent.parent)
        cls._create_directories(config)

        return cls(config=config, config_path=path)

    @staticmethod
    def _create_directories(config: Dict):
        for key_path in ("data.raw_path", "data.processed_path", "models.out_path"):
            path = ConfigManager._get_nested(config, key_path)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_paths(config: Dict[str, Any], base_dir: Path):
        for key_path in ("data.raw_path", "data.processed_path", "models.out_path"):
            path = ConfigManager._get_nested(config, key_path)
            if path:
                directory = Path(path)
                if not directory.is_absolute():
                    ConfigManager._set_nested(config, key_path, str(base_dir / directory))

    @staticmethod
    def _get_nested(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        value = config
        for key in key_path.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(key)
            if value is None:
                return default
        return value

    @staticmethod
    def _set_nested(config: Dict[str, Any], key_path: str, value: Any):
        keys = key_path.split(".")
        current = config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        return self._get_nested(self.config, key_path, default)

    def get_data_path(self, subpath: str = "") -> Path:
        base = Path(self.get("data.raw_path", "data/raw"))
        return base / subpath if subpath else base

    def get_results_path(self, subject_id: Optional[int] = None) -> Path:
        base = Path(self.get("models.out_path", "models"))
        if subject_id:
            base = base / f"subject_{subject_id}"
        base.mkdir(parents=True, exist_ok=True)
        return base


config_manager = ConfigManager.from_yaml()
