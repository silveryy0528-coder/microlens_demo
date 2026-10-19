"""Reusable settings dataclasses, validation, and generic YAML I/O."""
from dataclasses import dataclass
import math
from numbers import Rational, Real


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
