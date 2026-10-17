import numpy as np
import pytest

from microlens.image import Image
from microlens.synthetic import generate_synthetic_image


def single_lens_kwargs(**overrides):
    """Return a 100 x 100 px, 1000 nm FoV (10 nm pixels) with one lens at 500 nm."""
    kwargs = dict(
        image_shape=(100, 100),
        fov_nm=(1000.0, 1000.0),
        lens_pitch_nm=(1000.0, 1000.0),
        offset_nm=(500.0, 500.0),
        footprint_radius_nm=200.0,
        lens_height_nm=50.0,
    )
    kwargs.update(overrides)
    return kwargs


def centers_of(ground_truth):
    return [(lens["center_x_nm"], lens["center_y_nm"]) for lens in ground_truth]


def test_generate_synthetic_image_WHEN_defaults_are_used_THEN_returns_default_image_and_truth():
    image, ground_truth = generate_synthetic_image()

    assert isinstance(image, Image)
    assert image.values.shape == (1024, 1024)
    assert image.pixel_size_x == pytest.approx(6000.0 / 1024)
    assert image.pixel_size_y == pytest.approx(6000.0 / 1024)
    assert image.xy_unit == "nm"
    assert image.data_unit == "nm"
    assert len(ground_truth) == 9
    assert set(centers_of(ground_truth)) == {
        (x, y) for x in (1000.0, 3000.0, 5000.0) for y in (1000.0, 3000.0, 5000.0)
    }
    assert all(lens["footprint_radius_nm"] == 750.0 for lens in ground_truth)
    assert all(lens["geometric_height_nm"] == 500.0 for lens in ground_truth)


def test_generate_synthetic_image_WHEN_shape_is_rectangular_THEN_shape_is_n_y_by_n_x_and_pixel_sizes_follow_axes():
    image, _ = generate_synthetic_image(
        **single_lens_kwargs(image_shape=(50, 100), fov_nm=(1000.0, 1000.0))
    )

    assert image.values.shape == (50, 100)
    assert image.pixel_size_x == pytest.approx(10.0)
    assert image.pixel_size_y == pytest.approx(20.0)


def test_generate_synthetic_image_WHEN_axes_differ_THEN_preserves_shape_units_and_pixel_sizes():
    image, ground_truth = generate_synthetic_image(
        image_shape=(200, 300),
        fov_nm=(3000.0, 2000.0),
        lens_pitch_nm=(600.0, 500.0),
        offset_nm=(300.0, 250.0),
        footprint_radius_nm=100.0,
        lens_height_nm=50.0,
    )

    assert image.values.shape == (200, 300)
    assert image.pixel_size_x == pytest.approx(10.0)
    assert image.pixel_size_y == pytest.approx(10.0)
    assert image.xy_unit == "nm"
    assert image.data_unit == "nm"
    assert len(ground_truth) == 5 * 4


def test_generate_synthetic_image_WHEN_pitches_differ_THEN_centers_are_periodic_with_each_pitch():
    _, ground_truth = generate_synthetic_image(
        image_shape=(200, 300),
        fov_nm=(3000.0, 2000.0),
        lens_pitch_nm=(600.0, 500.0),
        offset_nm=(300.0, 250.0),
        footprint_radius_nm=100.0,
        lens_height_nm=50.0,
    )

    x_centers = sorted({lens["center_x_nm"] for lens in ground_truth})
    y_centers = sorted({lens["center_y_nm"] for lens in ground_truth})
    assert x_centers == [300.0, 900.0, 1500.0, 2100.0, 2700.0]
    assert y_centers == [250.0, 750.0, 1250.0, 1750.0]
    assert np.diff(x_centers) == pytest.approx(600.0)
    assert np.diff(y_centers) == pytest.approx(500.0)
    for lens in ground_truth:
        assert lens["center_x_nm"] == 300.0 + lens["index_x"] * 600.0
        assert lens["center_y_nm"] == 250.0 + lens["index_y"] * 500.0


def test_generate_synthetic_image_WHEN_offset_is_shifted_by_whole_pitches_THEN_same_lattice_is_generated():
    reference_image, reference_truth = generate_synthetic_image()

    for offset in (-1000.0, 3000.0):
        image, ground_truth = generate_synthetic_image(offset_nm=(offset, offset))
        assert set(centers_of(ground_truth)) == set(centers_of(reference_truth))
        assert np.array_equal(image.values, reference_image.values)

    shifted_truth = generate_synthetic_image(offset_nm=(-1000.0, 1000.0))[1]
    assert sorted({lens["index_x"] for lens in shifted_truth}) == [1, 2, 3]


def test_generate_synthetic_image_WHEN_offset_is_not_a_pitch_multiple_THEN_lattice_is_shifted():
    _, ground_truth = generate_synthetic_image(offset_nm=(1500.0, 500.0))

    # Centers at x = -500, 5500 and y = 500, 6500 are partial lenses.
    assert {x for x, _ in centers_of(ground_truth)} == {1500.0, 3500.0}
    assert {y for _, y in centers_of(ground_truth)} == {2500.0, 4500.0}


def test_generate_synthetic_image_WHEN_offset_tuple_is_x_y_ordered_THEN_x_offset_shifts_columns():
    _, ground_truth = generate_synthetic_image(offset_nm=(1500.0, 1000.0))

    assert {y for _, y in centers_of(ground_truth)} == {1000.0, 3000.0, 5000.0}
    assert 1500.0 in {x for x, _ in centers_of(ground_truth)}


def test_generate_synthetic_image_WHEN_center_is_outside_fov_THEN_partial_lens_is_rendered_but_not_in_ground_truth():
    image, ground_truth = generate_synthetic_image(
        **single_lens_kwargs(
            lens_pitch_nm=(400.0, 1000.0),
            offset_nm=(1050.0, 500.0),
            footprint_radius_nm=100.0,
        )
    )

    assert sorted(centers_of(ground_truth)) == [(250.0, 500.0), (650.0, 500.0)]
    # Center (1050, 500) is beyond the last pixel center x = 990.
    radius = (100.0**2 + 50.0**2) / (2 * 50.0)
    expected = np.sqrt(radius**2 - 60.0**2) - (radius - 50.0)
    assert image.values[50, 99] == pytest.approx(expected)


def test_generate_synthetic_image_WHEN_center_is_negative_THEN_partial_lens_is_rendered_but_not_in_ground_truth():
    image, ground_truth = generate_synthetic_image(
        **single_lens_kwargs(
            lens_pitch_nm=(400.0, 1000.0),
            offset_nm=(-50.0, 500.0),
            footprint_radius_nm=100.0,
        )
    )

    assert sorted(x for x, _ in centers_of(ground_truth)) == [350.0, 750.0]
    assert image.values[50, 0] > 0.0


def test_generate_synthetic_image_WHEN_center_is_on_image_boundary_THEN_lens_is_not_in_ground_truth():
    image, ground_truth = generate_synthetic_image(
        **single_lens_kwargs(
            lens_pitch_nm=(400.0, 1000.0),
            offset_nm=(0.0, 500.0),
            footprint_radius_nm=100.0,
        )
    )

    assert sorted(x for x, _ in centers_of(ground_truth)) == [400.0, 800.0]
    assert image.values[50, 0] == pytest.approx(50.0)


def test_generate_synthetic_image_WHEN_footprint_ends_just_before_or_reaches_border_pixel_THEN_only_the_former_is_in_ground_truth():
    # The last column is at x = 990 nm. A footprint with radius 100 nm centered
    # at 880 nm ends before it; one centered at 900 nm contains it.
    for center_x_nm, expected_listed in ((880.0, True), (900.0, False)):
        _, ground_truth = generate_synthetic_image(
            **single_lens_kwargs(
                lens_pitch_nm=(2000.0, 1000.0),
                offset_nm=(center_x_nm, 500.0),
                footprint_radius_nm=100.0,
            )
        )
        assert (center_x_nm in [x for x, _ in centers_of(ground_truth)]) is expected_listed


def test_generate_synthetic_image_WHEN_footprint_only_touches_last_pixel_THEN_lens_is_excluded():
    # Center 1090 nm is exactly one footprint radius from the last pixel center.
    image, ground_truth = generate_synthetic_image(
        **single_lens_kwargs(
            lens_pitch_nm=(400.0, 1000.0),
            offset_nm=(1090.0, 500.0),
            footprint_radius_nm=100.0,
        )
    )

    assert sorted(x for x, _ in centers_of(ground_truth)) == [290.0, 690.0]
    assert np.all(image.values[:, 95:] == 0.0)


def test_generate_synthetic_image_WHEN_no_lens_reaches_fov_THEN_ground_truth_is_empty_and_image_is_flat():
    # Centers at -500 and 1500 nm are both farther than the radius from the image.
    image, ground_truth = generate_synthetic_image(
        **single_lens_kwargs(
            lens_pitch_nm=(2000.0, 1000.0),
            offset_nm=(1500.0, 500.0),
            footprint_radius_nm=100.0,
        )
    )

    assert ground_truth == []
    assert np.all(image.values == 0.0)


def test_generate_synthetic_image_WHEN_cap_is_generated_THEN_ground_truth_matches_spherical_geometry():
    image, ground_truth = generate_synthetic_image(
        **single_lens_kwargs(image_shape=(101, 101))
    )
    assert len(ground_truth) == 1
    lens = ground_truth[0]

    expected_radius_nm = (200.0**2 + 50.0**2) / (2 * 50.0)
    assert lens["sphere_radius_nm"] == pytest.approx(expected_radius_nm)
    assert lens["footprint_radius_nm"] == 200.0
    assert lens["geometric_height_nm"] == 50.0
    assert lens["center_x_nm"] == 500.0
    assert lens["center_y_nm"] == 500.0
    assert (lens["index_x"], lens["index_y"]) == (0, 0)

    row, column = 52, 54
    x_nm = column * image.pixel_size_x
    y_nm = row * image.pixel_size_y
    distance_squared = (
        (x_nm - lens["center_x_nm"]) ** 2
        + (y_nm - lens["center_y_nm"]) ** 2
    )
    expected_height_nm = np.sqrt(expected_radius_nm**2 - distance_squared) - (
        expected_radius_nm - 50.0
    )
    assert image.values[row, column] == pytest.approx(expected_height_nm)
    assert image.values[0, 0] == 0.0


def test_generate_synthetic_image_WHEN_apex_is_on_pixel_THEN_lens_is_a_bump_with_edge_at_substrate():
    # Pixel (50, 50) is the lens center; pixel (50, 70) is at the footprint edge.
    image, _ = generate_synthetic_image(**single_lens_kwargs())

    assert image.values[50, 50] == pytest.approx(50.0)
    assert image.values[50, 70] == pytest.approx(0.0, abs=1e-9)
    assert image.values[50, 60] < image.values[50, 50]
    assert image.values[50, 60] > image.values[50, 70]
    assert image.values.min() >= 0.0
    assert image.values.max() <= 50.0


def test_generate_synthetic_image_WHEN_height_exceeds_footprint_radius_THEN_raises_value_error():
    with pytest.raises(ValueError, match="must not exceed footprint radius"):
        generate_synthetic_image(footprint_radius_nm=400.0, lens_height_nm=401.0)


def test_generate_synthetic_image_WHEN_height_equals_footprint_radius_THEN_hemisphere_is_accepted():
    _, ground_truth = generate_synthetic_image(
        footprint_radius_nm=400.0, lens_height_nm=400.0
    )

    assert ground_truth[0]["sphere_radius_nm"] == pytest.approx(400.0)


@pytest.mark.parametrize("pitch", [(1499.0, 2000.0), (2000.0, 1499.0)])
def test_generate_synthetic_image_WHEN_pitch_is_less_than_twice_footprint_radius_THEN_raises_value_error(
    pitch,
):
    with pytest.raises(ValueError, match="at least twice the footprint radius"):
        generate_synthetic_image(lens_pitch_nm=pitch)


def test_generate_synthetic_image_WHEN_pitch_equals_twice_footprint_radius_THEN_is_accepted():
    _, ground_truth = generate_synthetic_image(lens_pitch_nm=(1500.0, 1500.0))

    assert len(ground_truth) > 0


@pytest.mark.parametrize(
    ("parameter", "value", "error"),
    [
        ("image_shape", (0, 10), ValueError),
        ("image_shape", (10, -1), ValueError),
        ("image_shape", (10, 2.5), TypeError),
        ("image_shape", (10, True), TypeError),
        ("image_shape", (10,), ValueError),
        ("image_shape", (10, 10, 10), ValueError),
        ("image_shape", 10, ValueError),
        ("fov_nm", (0.0, 6000.0), ValueError),
        ("fov_nm", (6000.0, float("inf")), ValueError),
        ("fov_nm", (6000.0,), ValueError),
        ("lens_pitch_nm", (0.0, 2000.0), ValueError),
        ("lens_pitch_nm", (2000.0, float("nan")), ValueError),
        ("offset_nm", (float("nan"), 0.0), ValueError),
        ("offset_nm", (0.0, float("inf")), ValueError),
        ("offset_nm", (True, 0.0), TypeError),
        ("offset_nm", (0.0,), ValueError),
        ("footprint_radius_nm", float("nan"), ValueError),
        ("lens_height_nm", -1.0, ValueError),
        ("lens_height_nm", True, TypeError),
    ],
)
def test_generate_synthetic_image_WHEN_configuration_is_invalid_THEN_raises_expected_error(
    parameter, value, error
):
    with pytest.raises(error):
        generate_synthetic_image(**{parameter: value})
