"""Reusable settings dataclasses, validation, and generic YAML I/O."""
from dataclasses import dataclass
import math
from numbers import Rational, Real
from pathlib import Path

import yaml


def read_yaml(path: str | Path) -> dict:
    """Read a YAML file whose root value is a dictionary."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error

    if not isinstance(data, dict):
        raise ValueError(f"YAML root in {path} must be a dictionary")
    return data


def write_yaml(path: str | Path, data: dict) -> None:
    """Write a dictionary to a YAML file using safe serialization."""
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")

    path = Path(path)
    try:
        serialized = yaml.safe_dump(data, sort_keys=False)
    except yaml.YAMLError as error:
        raise TypeError(
            f"data contains values that cannot be serialized to YAML: {error}"
        ) from error

    with path.open("w", encoding="utf-8") as stream:
        stream.write(serialized)


def _require_finite_real(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    if isinstance(value, Rational):
        return
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class SubstrateSettings:
    """Settings for percentile-based substrate selection."""

    percentile: float
    distance: float

    def __post_init__(self) -> None:
        _require_finite_real("percentile", self.percentile)
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile must be between 0 and 100, inclusive")

        _require_finite_real("distance", self.distance)
        if self.distance < 0:
            raise ValueError("distance must be greater than or equal to 0")


@dataclass(frozen=True)
class PreprocessingSettings:
    """Select the preprocessing method and its configuration."""

    method: str
    substrate: SubstrateSettings | None = None

    def __post_init__(self) -> None:
        if type(self.method) is not str:
            raise TypeError("method must be a str")
        if self.method not in ("substrate", "all_pixels"):
            raise ValueError("method must be 'substrate' or 'all_pixels'")

        if self.method == "substrate":
            if not isinstance(self.substrate, SubstrateSettings):
                raise TypeError(
                    "substrate method requires a SubstrateSettings instance"
                )
        elif self.substrate is not None:
            raise ValueError("all_pixels method requires substrate to be None")
