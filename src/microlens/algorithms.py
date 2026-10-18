"""Numerical algorithms for microlens height maps."""

import math
from numbers import Real

import numpy as np
from scipy.ndimage import distance_transform_edt


def fit_and_subtract_plane(
    image_data: np.ndarray,
    pixel_size: tuple[float, float],
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a plane to selected pixels and subtract it from the full image.

    Pixel coordinates follow ``[row_y, column_x]`` indexing, with physical
    coordinates ``x = column * pixel_size[0]`` and
    ``y = row * pixel_size[1]``.

    Parameters
    ----------
    image_data : numpy.ndarray
        Non-empty, finite, real numeric 2D height map.
    pixel_size : tuple of float
        Positive finite pixel spacing ordered as ``(x, y)``.
    mask : numpy.ndarray of bool, optional
        Pixels used to fit the plane. It must have the same shape as
        `image_data`; ``True`` entries are included.

    Returns
    -------
    corrected_image : numpy.ndarray
        A new array containing the full image minus the fitted plane.
    coefficients : numpy.ndarray
        Fitted coefficients ``[a, b, c]`` for ``z = a*x + b*y + c``.

    Raises
    ------
    ValueError
        If an input is invalid or the selected design matrix has rank below 3.
    """
    data = np.asarray(image_data)
    if (
        data.ndim != 2
        or data.size == 0
        or not np.issubdtype(data.dtype, np.number)
        or np.issubdtype(data.dtype, np.complexfloating)
        or not np.isfinite(data).all()
    ):
        raise ValueError("image_data must be a non-empty finite real numeric 2D array")

    if not isinstance(pixel_size, (tuple, list, np.ndarray)) or len(pixel_size) != 2:
        raise ValueError("pixel_size must contain exactly two values")
    for value in pixel_size:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("pixel_size values must be positive finite real numbers")

    if mask is None:
        fit_mask = np.ones(data.shape, dtype=bool)
    else:
        fit_mask = np.asarray(mask)
        if fit_mask.dtype != np.dtype(bool) or fit_mask.shape != data.shape:
            raise ValueError(
                "mask must be a Boolean array with the same shape as image_data"
            )

    data = data.astype(float, copy=False)
    rows, columns = np.indices(data.shape)
    x = columns * float(pixel_size[0])
    y = rows * float(pixel_size[1])
    design = np.column_stack((x.ravel(), y.ravel(), np.ones(data.size)))
    coefficients, _, rank, _ = np.linalg.lstsq(
        design[fit_mask.ravel()], data.ravel()[fit_mask.ravel()], rcond=None
    )
    if rank != 3:
        raise ValueError("selected pixels must provide a plane design matrix of rank 3")

    fitted_plane = (design @ coefficients).reshape(data.shape)
    corrected_image = data - fitted_plane
    return corrected_image, coefficients


def select_substrate_mask(
    image_data: np.ndarray,
    percentile: float = 5,
    distance: float = 5,
) -> np.ndarray:
    """Select low-height substrate pixels using a percentile-based threshold.

    Parameters
    ----------
    image_data : numpy.ndarray
        Non-empty, finite, real numeric 2D height map.
    percentile : float, default=5
        Percentile in the inclusive range from 0 to 100.
    distance : float, default=5
        Non-negative offset added to the selected percentile.

    Returns
    -------
    numpy.ndarray of bool
        ``True`` where the height is strictly below
        ``percentile(image_data, percentile) + distance``.

    Raises
    ------
    ValueError
        If an input is invalid.
    """
    data = np.asarray(image_data)
    if (
        data.ndim != 2
        or data.size == 0
        or not np.issubdtype(data.dtype, np.number)
        or np.issubdtype(data.dtype, np.complexfloating)
        or not np.isfinite(data).all()
    ):
        raise ValueError("image_data must be a non-empty finite real numeric 2D array")

    if (
        isinstance(percentile, (bool, np.bool_))
        or not isinstance(percentile, Real)
        or not math.isfinite(percentile)
        or not 0 <= percentile <= 100
    ):
        raise ValueError("percentile must be a finite real number from 0 to 100")
    if (
        isinstance(distance, (bool, np.bool_))
        or not isinstance(distance, Real)
        or not math.isfinite(distance)
        or distance < 0
    ):
        raise ValueError("distance must be a finite non-negative real number")

    threshold = np.percentile(data, percentile) + distance
    return data < threshold


def dilate_mask_by_distance(
    mask: np.ndarray,
    pixel_size: tuple[float, float],
    distance: float,
) -> np.ndarray:
    """Dilate a Boolean mask by a physical Euclidean distance.

    Pixel spacing is supplied as ``(x, y)``; EDT sampling is reordered to
    ``(y, x)`` to match NumPy's ``[row_y, column_x]`` array axes.

    Parameters
    ----------
    mask : numpy.ndarray
        2D Boolean array where ``True`` indicates the region to dilate.
    pixel_size : tuple[float, float]
        Physical size of a pixel in the ``(x, y)`` directions.
    distance : float
        Physical Euclidean distance by which to dilate the mask.

    Returns
    -------
    numpy.ndarray of bool
        Dilated Boolean mask.

    Raises
    ------
    ValueError
        If an input is invalid.
    """
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2 or mask_array.dtype != np.dtype(bool):
        raise ValueError("mask must be a 2D Boolean array")
    if not isinstance(pixel_size, (tuple, list, np.ndarray)) or len(pixel_size) != 2:
        raise ValueError("pixel_size must contain exactly two values")
    for value in pixel_size:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("pixel_size values must be positive finite real numbers")
    if (
        isinstance(distance, (bool, np.bool_))
        or not isinstance(distance, Real)
        or not math.isfinite(distance)
        or distance < 0
    ):
        raise ValueError("distance must be a finite non-negative real number")

    if distance == 0:
        return mask_array.copy()
    if not mask_array.any():
        return np.zeros(mask_array.shape, dtype=bool)

    distance_map = distance_transform_edt(
        ~mask_array, sampling=(pixel_size[1], pixel_size[0])
    )
    return distance_map <= distance
