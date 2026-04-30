"""Configuration loading and validation."""

from pathlib import Path

import yaml

from .models import Config


def load_config(path: Path | str) -> Config:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        data = yaml.safe_load(f)

    return Config.model_validate(data)


def save_config(config: Config, path: Path | str) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    with path.open("w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
