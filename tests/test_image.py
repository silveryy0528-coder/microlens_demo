import numpy as np
import pytest

from microlens.image import Image


def test_image_WHEN_valid_inputs_THEN_stores_values_and_metadata():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    image = Image(
        values=values,
        pixel_size_x=5.0,
        pixel_size_y=6.0,
        xy_unit="nm",
        data_unit="nm",
    )

    assert image.values is values
    assert image.pixel_size_x == 5.0
    assert image.pixel_size_y == 6.0
    assert image.xy_unit == "nm"
    assert image.data_unit == "nm"


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, 2.0]),
        np.zeros((2, 2, 2)),
    ],
)
def test_image_WHEN_values_are_not_2d_THEN_raises_value_error(values):
    with pytest.raises(ValueError, match="values must be a 2D array"):
        Image(values, 1.0, 1.0, "nm", "nm")


def test_image_WHEN_values_are_not_a_numpy_array_THEN_raises_type_error():
    with pytest.raises(TypeError, match="values must be a NumPy array"):
        Image([[1.0, 2.0]], 1.0, 1.0, "nm", "nm")


@pytest.mark.parametrize(
    ("pixel_size_x", "pixel_size_y"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, float("nan")),
        (1.0, float("-inf")),
        (True, 1.0),
    ],
)
def test_image_WHEN_pixel_size_is_not_positive_and_finite_THEN_raises_value_error(
    pixel_size_x, pixel_size_y
):
    with pytest.raises(ValueError, match="must be a positive finite number"):
        Image(np.zeros((2, 2)), pixel_size_x, pixel_size_y, "nm", "nm")
