"""Generate synthetic microlens height maps with analytical ground truth.

Geometry and conventions follow ``docs/methodology.md``.
"""

import math
import copy
from numbers import Integral, Real

import numpy as np

from microlens.image import Image


def generate_synthetic_image(
    slope_xy: tuple[float, float] = (0.0, 0.0),
    **microlens_kwargs,
) -> tuple[Image, list[dict[str, float]]]:
    """Generate a microlens image with optional substrate tilt.

    `slope_xy` is passed to `add_tilt`; the other keyword arguments are passed
    to `generate_microlens_image`. The ground truth is not affected by tilt.

    Returns
    -------
    tuple of (Image, list of dict)
        Height map and complete-lens ground truth.
    """
    image, ground_truth = generate_microlens_image(**microlens_kwargs)
    return add_tilt(image, slope_xy), ground_truth


def add_tilt(image: Image, slope: tuple[float, float]) -> Image:
    """Return a copy of `image` with the plane ``z = a * x + b * y`` added.

    `slope` is ``(a, b)`` in data unit per lateral unit (no unit conversion).
    Pixel-center coordinates are used, so the plane is zero at pixel (0, 0).
    The input image is not modified.
    """
    if not isinstance(slope, (tuple, list)) or len(slope) != 2:
        raise ValueError("slope must be a pair (a, b)")
    for value in slope:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("slope entries must be real numbers")
        if not math.isfinite(value):
            raise ValueError("slope entries must be finite")

    slope_x, slope_y = slope
    n_y, n_x = image.values.shape
    x_coordinates = np.arange(n_x) * image.pixel_size_x
    y_coordinates = np.arange(n_y) * image.pixel_size_y
    plane = slope_x * x_coordinates[np.newaxis, :] + slope_y * y_coordinates[:, np.newaxis]

    image_out = copy.deepcopy(image)
    image_out.values += plane
    return image_out


def generate_microlens_image(
    image_shape: tuple[int, int] = (1024, 1024),
    fov_nm: tuple[float, float] = (6000.0, 6000.0),
    lens_pitch_nm: tuple[float, float] = (2000.0, 2000.0),
    offset_nm: tuple[float, float] = (1000.0, 1000.0),
    footprint_radius_nm: float = 750.0,
    lens_height_nm: float = 500.0,
) -> tuple[Image, list[dict[str, float]]]:
    """Generate a periodic spherical-cap lens array on a flat substrate.

    Lattice, cap geometry, and the complete/partial lens rule are defined in
    ``docs/methodology.md``. Partial lenses are rendered but not returned in
    the ground truth.

    Parameters
    ----------
    image_shape : tuple of int
        ``(n_y, n_x)``.
    fov_nm, lens_pitch_nm, offset_nm : tuple of float
        ``(x, y)`` in nm. `offset_nm` is the center of lens ``i = j = 0``.
    footprint_radius_nm, lens_height_nm : float
        Cap footprint radius and apex height in nm.

    Returns
    -------
    tuple of (Image, list of dict)
        Height map in nm, and one ground-truth dict per complete lens
        (lattice indices, centers, sphere and footprint radii, height).
    """
    for name, pair in (
        ("image_shape", image_shape),
        ("fov_nm", fov_nm),
        ("lens_pitch_nm", lens_pitch_nm),
        ("offset_nm", offset_nm),
    ):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"{name} must be a pair of two values")

    for value in image_shape:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("image_shape entries must be integers")
        if value <= 0:
            raise ValueError("image_shape entries must be positive")

    for name, values_pair in (
        ("fov_nm", fov_nm),
        ("lens_pitch_nm", lens_pitch_nm),
        ("offset_nm", offset_nm),
        ("footprint_radius_nm", (footprint_radius_nm,)),
        ("lens_height_nm", (lens_height_nm,)),
    ):
        for value in values_pair:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} entries must be real numbers")
            if not math.isfinite(value):
                raise ValueError(f"{name} entries must be finite")
            if name != "offset_nm" and value <= 0:
                raise ValueError(f"{name} entries must be positive")

    n_y, n_x = image_shape
    fov_x_nm, fov_y_nm = fov_nm
    lens_pitch_x_nm, lens_pitch_y_nm = lens_pitch_nm
    offset_x_nm, offset_y_nm = offset_nm

    if 2 * footprint_radius_nm > min(lens_pitch_x_nm, lens_pitch_y_nm):
        raise ValueError("lens pitch must be at least twice the footprint radius")

    if lens_height_nm > footprint_radius_nm:
        raise ValueError(
            "lens height must not exceed footprint radius "
            "(taller-than-hemisphere caps are not single-valued height maps)"
        )

    pixel_size_x = fov_x_nm / n_x
    pixel_size_y = fov_y_nm / n_y
    sphere_radius_nm = (footprint_radius_nm**2 + lens_height_nm**2) / (
        2 * lens_height_nm
    )
    x_coordinates = np.arange(n_x) * pixel_size_x
    y_coordinates = np.arange(n_y) * pixel_size_y

    # A center x_i can have a pixel center strictly within the footprint only if
    # -radius < x_i < x_last + radius, where x_last is the last pixel center.
    def lattice_index_range(offset, pitch, last_coordinate):
        first = math.floor((-footprint_radius_nm - offset) / pitch) + 1
        last = math.ceil((last_coordinate + footprint_radius_nm - offset) / pitch) - 1
        return range(first, last + 1)

    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    values = np.zeros((n_y, n_x), dtype=float)
    ground_truth: list[dict[str, float]] = []

    for index_y in lattice_index_range(offset_y_nm, lens_pitch_y_nm, y_coordinates[-1]):
        center_y_nm = offset_y_nm + index_y * lens_pitch_y_nm
        for index_x in lattice_index_range(
            offset_x_nm, lens_pitch_x_nm, x_coordinates[-1]
        ):
            center_x_nm = offset_x_nm + index_x * lens_pitch_x_nm
            distance_squared = (x_grid - center_x_nm) ** 2 + (y_grid - center_y_nm) ** 2
            inside_footprint = distance_squared < footprint_radius_nm**2
            if not inside_footprint.any():
                continue
            radicand = np.maximum(
                sphere_radius_nm**2 - distance_squared[inside_footprint], 0.0
            )
            # Sphere surface with its center at z = lens_height - radius, so
            # the apex is at lens_height and the footprint edge is at z = 0.
            values[inside_footprint] = np.sqrt(radicand) - (
                sphere_radius_nm - lens_height_nm
            )
            touches_border = (
                inside_footprint[0, :].any()
                or inside_footprint[-1, :].any()
                or inside_footprint[:, 0].any()
                or inside_footprint[:, -1].any()
            )
            if touches_border:
                continue
            ground_truth.append(
                {
                    "index_x": index_x,
                    "index_y": index_y,
                    "center_x_nm": center_x_nm,
                    "center_y_nm": center_y_nm,
                    "sphere_radius_nm": sphere_radius_nm,
                    "footprint_radius_nm": footprint_radius_nm,
                    "geometric_height_nm": lens_height_nm,
                }
            )

    image = Image(
        values=values,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        xy_unit="nm",
        data_unit="nm",
    )
    return image, ground_truth
