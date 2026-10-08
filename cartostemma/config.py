"""Load and validate the pipeline configuration (see ``config.example.yaml``)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    paths: dict[str, str]
    crs: dict[str, int]
    grid: dict[str, Any]
    distortion: dict[str, Any]
    clustering: dict[str, Any]
    stemma: dict[str, Any]
    clone_groups: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        clone = raw.get("clone_groups") or {}
        return cls(
            paths=raw["paths"],
            crs=raw["crs"],
            grid=raw["grid"],
            distortion=raw["distortion"],
            clustering=raw["clustering"],
            stemma=raw["stemma"],
            clone_groups=clone,
        )
