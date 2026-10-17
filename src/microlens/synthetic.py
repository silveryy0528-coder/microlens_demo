"""Generate analytical synthetic microlens height maps."""

import math
from numbers import Integral, Real

import numpy as np

from microlens.image import Image


def generate_synthetic_image(
    image_shape: tuple[int, int] = (1024, 1024),
    fov_nm: tuple[float, float] = (6000.0, 6000.0),
    lens_pitch_nm: tuple[float, float] = (2000.0, 2000.0),
    offset_nm: tuple[float, float] = (1000.0, 1000.0),
    footprint_radius_nm: float = 750.0,
    lens_height_nm: float = 500.0,
) -> tuple[Image, list[dict[str, float]]]:
    """Generate a periodic spherical-cap microlens array seen through a FoV.

    Lens centers lie on the lattice ``x_i = offset_x + i * pitch_x`` and
    ``y_j = offset_y + j * pitch_y`` for all integers ``i, j``, in the
    pixel-center coordinates of `Image` (``x = column * pixel_size_x``,
    ``y = row * pixel_size_y``). Each lens is a spherical cap on a substrate
    at ``z = 0``, with its apex at `lens_height_nm` and footprint edge at
    ``z = 0``. Every lens with at least one pixel center strictly inside its
    footprint is rendered, including partially visible lenses whose center is
    outside the image. Only complete lenses are listed in the ground truth: a
    lens is partial, and omitted, if any pixel inside its footprint lies in
    the first or last row or column of the image.

    Parameters
    ----------
    image_shape : tuple of int
        Image shape ``(n_y, n_x)``, i.e. rows then columns.
    fov_nm, lens_pitch_nm, offset_nm : tuple of float
        Field of view, lattice pitch, and lattice position of lens
        ``i = j = 0``, each in nm and ordered ``(x, y)``. Offsets may be
        negative or exceed the FoV.
    footprint_radius_nm, lens_height_nm : float
        Cap footprint radius and geometric apex height in nm. The height must
        not exceed the radius, and each pitch must be at least twice the
        radius so lenses do not overlap.

    Returns
    -------
    image : Image
        Height map in nm with pixel sizes ``fov / n``.
    ground_truth : list of dict
        One entry per complete (non-border-touching) lens, ordered by
        ``index_y`` then ``index_x``, with keys ``index_x``, ``index_y``,
        ``center_x_nm``, ``center_y_nm``, ``sphere_radius_nm``,
        ``footprint_radius_nm`` and ``geometric_height_nm``.
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
    sphere_radius_nm = (
        footprint_radius_nm**2 + lens_height_nm**2
    ) / (2 * lens_height_nm)
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
            distance_squared = (
                (x_grid - center_x_nm) ** 2 + (y_grid - center_y_nm) ** 2
            )
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
