"""Generate analytical synthetic microlens height maps."""

import math
from numbers import Integral, Real

import numpy as np

from microlens.image import Image


def generate_synthetic_image(
    n_x: int = 1024,
    n_y: int = 1024,
    fov_x_nm: float = 6000.0,
    fov_y_nm: float = 6000.0,
    lenses_x: int = 3,
    lenses_y: int = 3,
    footprint_radius_nm: float = 750.0,
    lens_height_nm: float = 500.0,
) -> tuple[Image, list[dict[str, float]]]:
    """Generate a regular array of spherical-cap microlenses.

    Pixel-center coordinates follow ``x = column * pixel_size_x`` and
    ``y = row * pixel_size_y``. Lenses are centered in equally sized grid
    cells. The footprint diameter must fit within both cell dimensions so
    generated lenses do not overlap. Each lens is a spherical cap rising above
    a flat substrate at ``z = 0``; its footprint edge lies at ``z = 0`` and its
    apex at ``lens_height_nm``. The height must not exceed the footprint radius
    (at most a hemisphere).

    Parameters
    ----------
    n_x, n_y : int
        Image dimensions in columns and rows.
    fov_x_nm, fov_y_nm : float
        Physical field of view along each image axis, in nanometers.
    lenses_x, lenses_y : int
        Number of lenses along each physical image axis.
    footprint_radius_nm, lens_height_nm : float
        Spherical-cap footprint radius and geometric apex height, in
        nanometers.

    Returns
    -------
    image : Image
        Height map with lateral pixel sizes derived from the field of view.
    ground_truth : list of dict
        Per-lens center coordinates, sphere radius, footprint radius, and
        geometric height, all in nanometers.
    """
    for name, value in (
        ("n_x", n_x),
        ("n_y", n_y),
        ("lenses_x", lenses_x),
        ("lenses_y", lenses_y),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    for name, value in (
        ("fov_x_nm", fov_x_nm),
        ("fov_y_nm", fov_y_nm),
        ("footprint_radius_nm", footprint_radius_nm),
        ("lens_height_nm", lens_height_nm),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")

    pixel_size_x = fov_x_nm / n_x
    pixel_size_y = fov_y_nm / n_y
    cell_width_x = fov_x_nm / lenses_x
    cell_width_y = fov_y_nm / lenses_y
    if 2 * footprint_radius_nm > min(cell_width_x, cell_width_y):
        raise ValueError("footprint diameter must fit within each grid cell")

    if lens_height_nm > footprint_radius_nm:
        raise ValueError(
            "lens height must not exceed footprint radius "
            "(taller-than-hemisphere caps are not single-valued height maps)"
        )

    sphere_radius_nm = (
        footprint_radius_nm**2 + lens_height_nm**2
    ) / (2 * lens_height_nm)
    x_coordinates = np.arange(n_x) * pixel_size_x
    y_coordinates = np.arange(n_y) * pixel_size_y
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    values = np.zeros((n_y, n_x), dtype=float)
    ground_truth: list[dict[str, float]] = []

    for lens_y in range(lenses_y):
        center_y_nm = (lens_y + 0.5) * cell_width_y
        for lens_x in range(lenses_x):
            center_x_nm = (lens_x + 0.5) * cell_width_x
            distance_squared = (
                (x_grid - center_x_nm) ** 2 + (y_grid - center_y_nm) ** 2
            )
            inside_footprint = distance_squared <= footprint_radius_nm**2
            radicand = np.maximum(
                sphere_radius_nm**2 - distance_squared[inside_footprint], 0.0
            )
            # Sphere surface with its center at z = lens_height - radius, so
            # the apex is at lens_height and the footprint edge is at z = 0.
            values[inside_footprint] = np.sqrt(radicand) - (
                sphere_radius_nm - lens_height_nm
            )
            ground_truth.append(
                {
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    image, ground_truth = generate_synthetic_image(
        n_x=1000,
        n_y=400,
        lenses_x=2,
        lenses_y=4,
        fov_x_nm=6000.0,
        fov_y_nm=6000.0,
        footprint_radius_nm=650.0,
        lens_height_nm=500.0,
    )
    plt.figure()
    xx, yy = np.meshgrid(
        image.pixel_size_x * np.arange(image.values.shape[1]),
        image.pixel_size_y * np.arange(image.values.shape[0]))
    plt.pcolormesh(xx, yy, image.values, cmap="viridis")
    plt.colorbar(label=f"{image.data_unit}")
    plt.show()

    plt.figure()
    plt.plot(image.values[227, :])
    plt.xlabel("X Pixel")
    plt.ylabel(f"{image.data_unit}")
    plt.show()