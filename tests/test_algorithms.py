import numpy as np
import pytest

from microlens.algorithms import fit_and_subtract_plane, select_substrate_mask


def test_fit_and_subtract_plane_WHEN_analytic_plane_and_no_mask_THEN_corrects_all_pixels():
    rows, columns = np.indices((6, 8))
    pixel_size = (2.5, 4.0)
    x = columns * pixel_size[0]
    y = rows * pixel_size[1]
    image_data = 0.7 * x - 1.2 * y + 8.5
    original = image_data.copy()

    corrected, coefficients = fit_and_subtract_plane(
        image_data, pixel_size, mask=None
    )

    np.testing.assert_allclose(corrected, np.zeros_like(image_data), atol=1e-12)
    np.testing.assert_allclose(coefficients, [0.7, -1.2, 8.5], atol=1e-12)
    np.testing.assert_array_equal(image_data, original)


def test_fit_and_subtract_plane_WHEN_lens_is_excluded_by_mask_THEN_preserves_lens_and_flattens_substrate():
    rows, columns = np.indices((9, 11))
    pixel_size = (1.5, 2.0)
    x = columns * pixel_size[0]
    y = rows * pixel_size[1]
    substrate = 0.4 * x + 0.9 * y - 3.0
    lens = np.zeros_like(substrate)
    lens[3:6, 4:7] = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 5.0, 2.0],
            [1.0, 2.0, 1.0],
        ]
    )
    image_data = substrate + lens
    mask = lens == 0
    original = image_data.copy()

    corrected, _ = fit_and_subtract_plane(image_data, pixel_size, mask=mask)

    np.testing.assert_allclose(corrected, lens, atol=1e-12)
    np.testing.assert_array_equal(image_data, original)


@pytest.mark.parametrize(
    ("image_data", "pixel_size", "mask"),
    [
        (np.empty((0, 3)), (1.0, 1.0), None),
        (np.zeros((2, 2, 2)), (1.0, 1.0), None),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), (1.0, 1.0), None),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), (1.0, 1.0), None),
        (np.array([[1.0 + 2.0j, 2.0], [3.0, 4.0]]), (1.0, 1.0), None),
        (np.ones((3, 3)), (1.0,), None),
        (np.ones((3, 3)), (0.0, 1.0), None),
        (np.ones((3, 3)), (1.0, -1.0), None),
        (np.ones((3, 3)), (np.nan, 1.0), None),
        (np.ones((3, 3)), (1.0, np.inf), None),
        (np.ones((3, 3)), (1.0, 1.0), np.zeros((3, 3), dtype=bool)),
        (np.ones((3, 3)), (1.0, 1.0), np.ones((2, 3), dtype=bool)),
        (np.ones((3, 3)), (1.0, 1.0), np.ones((3, 3), dtype=int)),
        (
            np.ones((3, 3)),
            (1.0, 1.0),
            np.array(
                [
                    [True, False, False],
                    [False, True, False],
                    [False, False, False],
                ]
            ),
        ),
        (
            np.ones((3, 3)),
            (1.0, 1.0),
            np.array(
                [
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                ]
            ),
        ),
    ],
)
def test_fit_and_subtract_plane_WHEN_image_spacing_or_fit_selection_is_invalid_THEN_raises(
    image_data, pixel_size, mask
):
    with pytest.raises((TypeError, ValueError)):
        fit_and_subtract_plane(image_data, pixel_size, mask=mask)


def test_select_substrate_mask_WHEN_values_equal_threshold_THEN_excludes_boundary_strictly():
    image_data = np.array([[0.0, 1.0, 2.0, 3.0]])

    mask = select_substrate_mask(image_data, percentile=50, distance=0.5)

    # The median is 1.5, so the value exactly on the threshold is not selected.
    np.testing.assert_array_equal(mask, np.array([[True, True, False, False]]))


@pytest.mark.parametrize(
    ("percentile", "distance", "expected"),
    [
        (20, 0, [[True, False, False, False, False, False]]),
        (80, 0, [[True, True, True, True, False, False]]),
        (20, 2, [[True, True, True, False, False, False]]),
    ],
)
def test_select_substrate_mask_WHEN_percentile_or_distance_changes_THEN_returns_expected_selection(
    percentile, distance, expected
):
    image_data = np.arange(6, dtype=float).reshape(1, 6)

    mask = select_substrate_mask(image_data, percentile, distance)

    np.testing.assert_array_equal(mask, np.array(expected, dtype=bool))


@pytest.mark.parametrize(
    ("percentile", "distance"),
    [
        (-0.1, 0),
        (100.1, 0),
        (float("nan"), 0),
        (float("inf"), 0),
        (50, -0.1),
        (50, float("nan")),
        (50, float("inf")),
        (50, float("-inf")),
    ],
)
def test_select_substrate_mask_WHEN_parameters_are_invalid_THEN_raises_value_error(
    percentile, distance
):
    with pytest.raises(ValueError):
        select_substrate_mask(np.ones((2, 2)), percentile, distance)


@pytest.mark.parametrize(
    "image_data",
    [
        np.empty((0, 2)),
        np.array([1.0, 2.0]),
        np.zeros((2, 2, 2)),
        np.array([[1.0, np.nan]]),
        np.array([[1.0, np.inf]]),
        np.array([[1.0 + 1.0j, 2.0]]),
    ],
)
def test_select_substrate_mask_WHEN_image_is_malformed_or_nonfinite_THEN_raises(
    image_data,
):
    with pytest.raises((TypeError, ValueError)):
        select_substrate_mask(image_data)
