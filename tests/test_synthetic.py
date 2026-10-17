import numpy as np
import pytest

from microlens.image import Image
from microlens.synthetic import generate_synthetic_image


def test_generate_synthetic_image_WHEN_defaults_are_used_THEN_returns_default_image_and_truth():
    image, ground_truth = generate_synthetic_image()

    assert isinstance(image, Image)
    assert image.values.shape == (1024, 1024)
    assert image.pixel_size_x == pytest.approx(6000.0 / 1024)
    assert image.pixel_size_y == pytest.approx(6000.0 / 1024)
    assert image.xy_unit == "nm"
    assert image.data_unit == "nm"
    assert len(ground_truth) == 9
    assert all(lens["footprint_radius_nm"] == 750.0 for lens in ground_truth)
    assert all(lens["geometric_height_nm"] == 500.0 for lens in ground_truth)


def test_generate_synthetic_image_WHEN_custom_configuration_is_used_THEN_matches_shape_and_metadata():
    image, ground_truth = generate_synthetic_image(
        n_x=80,
        n_y=60,
        fov_x_nm=800.0,
        fov_y_nm=600.0,
        lenses_x=2,
        lenses_y=1,
        footprint_radius_nm=100.0,
        lens_height_nm=50.0,
    )

    assert image.values.shape == (60, 80)
    assert image.pixel_size_x == pytest.approx(10.0)
    assert image.pixel_size_y == pytest.approx(10.0)
    assert len(ground_truth) == 2


@pytest.mark.parametrize(
    ("parameter", "value", "error"),
    [
        ("n_x", 0, ValueError),
        ("n_y", -1, ValueError),
        ("lenses_x", 0, ValueError),
        ("lenses_y", 1.5, TypeError),
        ("n_x", 2.5, TypeError),
        ("fov_x_nm", 0.0, ValueError),
        ("fov_y_nm", float("inf"), ValueError),
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


def test_generate_synthetic_image_WHEN_footprint_does_not_fit_cell_THEN_raises_value_error():
    with pytest.raises(ValueError, match="footprint diameter must fit"):
        generate_synthetic_image(footprint_radius_nm=1001.0)


def test_generate_synthetic_image_WHEN_regular_array_is_requested_THEN_centers_are_cell_centered():
    _, ground_truth = generate_synthetic_image(
        n_x=100,
        n_y=80,
        fov_x_nm=1000.0,
        fov_y_nm=800.0,
        lenses_x=2,
        lenses_y=2,
        footprint_radius_nm=100.0,
        lens_height_nm=50.0,
    )

    centers = [(lens["center_x_nm"], lens["center_y_nm"]) for lens in ground_truth]
    assert centers == [
        (250.0, 200.0),
        (750.0, 200.0),
        (250.0, 600.0),
        (750.0, 600.0),
    ]


def test_generate_synthetic_image_WHEN_cap_is_generated_THEN_ground_truth_matches_spherical_geometry():
    image, ground_truth = generate_synthetic_image(
        n_x=101,
        n_y=101,
        fov_x_nm=1000.0,
        fov_y_nm=1000.0,
        lenses_x=1,
        lenses_y=1,
        footprint_radius_nm=200.0,
        lens_height_nm=50.0,
    )
    lens = ground_truth[0]

    expected_radius_nm = (200.0**2 + 50.0**2) / (2 * 50.0)
    assert lens["sphere_radius_nm"] == pytest.approx(expected_radius_nm)
    assert lens["center_x_nm"] == 500.0
    assert lens["center_y_nm"] == 500.0

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
    image, _ = generate_synthetic_image(
        n_x=100,
        n_y=100,
        fov_x_nm=1000.0,
        fov_y_nm=1000.0,
        lenses_x=1,
        lenses_y=1,
        footprint_radius_nm=200.0,
        lens_height_nm=50.0,
    )

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


def test_generate_synthetic_image_WHEN_anisotropic_sampling_is_used_THEN_preserves_physical_scales():
    image, _ = generate_synthetic_image(
        n_x=100,
        n_y=50,
        fov_x_nm=1000.0,
        fov_y_nm=1000.0,
        lenses_x=1,
        lenses_y=1,
        footprint_radius_nm=100.0,
        lens_height_nm=50.0,
    )

    assert image.pixel_size_x == pytest.approx(10.0)
    assert image.pixel_size_y == pytest.approx(20.0)
