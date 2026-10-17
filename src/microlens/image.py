"""Data model for height maps and their physical sampling metadata."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


@dataclass
class Image:
    """A 2D height map with lateral sampling and unit metadata.

    Parameters
    ----------
    values : numpy.ndarray
        Height values indexed as ``[row_y, column_x]``.
    pixel_size_x, pixel_size_y : float
        Positive finite spacing between pixel centers in the corresponding
        lateral direction.
    xy_unit, data_unit : str
        Units for the lateral pixel sizes and height values, respectively.
    """

    values: np.ndarray
    pixel_size_x: float
    pixel_size_y: float
    xy_unit: str
    data_unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a NumPy array")
        if self.values.ndim != 2:
            raise ValueError("values must be a 2D array")

        for name, pixel_size in (
            ("pixel_size_x", self.pixel_size_x),
            ("pixel_size_y", self.pixel_size_y),
        ):
            if (
                isinstance(pixel_size, bool)
                or not isinstance(pixel_size, Real)
                or not math.isfinite(pixel_size)
                or pixel_size <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
