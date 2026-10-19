from pathlib import Path

import numpy as np
import pytest

from microlens.algorithms import fit_and_subtract_plane
from microlens.image import Image
from microlens.settings import read_yaml, write_yaml
from microlens.synthetic import generate_synthetic_image
from microlens.usecasemicrolens import (
    MicrolensSettings,
    UseCaseMicrolens,
)


DEFAULT_SETTINGS_PATH = Path(__file__).parents[1] / "configs" / "settings.yaml"


def test_use_case_microlens_WHEN_initialized_THEN_references_original_with_incomplete_states():
    image = Image(np.array([[1.0, 2.0], [3.0, 4.0]]), 2.0, 3.0, "um", "nm")

    use_case = UseCaseMicrolens(image)

    assert use_case.image is image
    assert use_case.preprocessed_image is None
    assert use_case.preprocessing_completed is False
    assert use_case.processing_completed is False
    assert use_case.settings is None


def test_preprocess_image_WHEN_all_pixels_method_is_selected_THEN_subtracts_fitted_plane_and_preserves_metadata():
    rows, columns = np.indices((7, 9))
    pixel_size_x, pixel_size_y = 2.0, 3.0
    x = columns * pixel_size_x
    y = rows * pixel_size_y
    features = np.zeros((7, 9))
    features[2:5, 3:6] = np.array(
        [[2.0, 4.0, 2.0], [4.0, 8.0, 4.0], [2.0, 4.0, 2.0]]
    )
    values = 0.7 * x - 1.2 * y + 8.5 + features
    image = Image(values.copy(), pixel_size_x, pixel_size_y, "um", "nm")
    use_case = UseCaseMicrolens(image)
    use_case.settings = MicrolensSettings.from_dict(
        {"preprocessing": {"method": "all_pixels"}}
    )
    expected, _ = fit_and_subtract_plane(
        values, (pixel_size_x, pixel_size_y)
    )

    use_case.preprocess_image()

    preprocessed_image = use_case.preprocessed_image
    assert isinstance(preprocessed_image, Image)
    assert preprocessed_image is not image
    assert preprocessed_image.values is not image.values
    np.testing.assert_allclose(preprocessed_image.values, expected, atol=1e-12)
    assert preprocessed_image.pixel_size_x == image.pixel_size_x
    assert preprocessed_image.pixel_size_y == image.pixel_size_y
    assert preprocessed_image.xy_unit == image.xy_unit
    assert preprocessed_image.data_unit == image.data_unit
    np.testing.assert_array_equal(image.values, values)
    assert use_case.preprocessing_completed is True


def test_preprocess_image_WHEN_substrate_method_is_selected_THEN_flattens_substrate_and_preserves_lens_geometry():
    image, _ = generate_synthetic_image(
        image_shape=(81, 83),
        fov_nm=(830.0, 810.0),
        lens_pitch_nm=(900.0, 900.0),
        offset_nm=(415.0, 405.0),
        footprint_radius_nm=120.0,
        lens_height_nm=35.0,
    )
    rows, columns = np.indices(image.values.shape)
    tilted_values = (
        image.values
        + 0.013 * columns * image.pixel_size_x
        - 0.021 * rows * image.pixel_size_y
        + 7.0
    )
    tilted_image = Image(
        tilted_values.copy(),
        image.pixel_size_x,
        image.pixel_size_y,
        image.xy_unit,
        image.data_unit,
    )
    use_case = UseCaseMicrolens(tilted_image)
    use_case.settings = MicrolensSettings.from_dict(
        {
            "preprocessing": {
                "method": "substrate",
                "substrate": {"percentile": 15.0, "distance": 0.01},
            }
        }
    )

    use_case.preprocess_image()

    preprocessed_image = use_case.preprocessed_image
    assert isinstance(preprocessed_image, Image)
    assert preprocessed_image is not tilted_image
    assert preprocessed_image.pixel_size_x == tilted_image.pixel_size_x
    assert preprocessed_image.pixel_size_y == tilted_image.pixel_size_y
    assert preprocessed_image.xy_unit == tilted_image.xy_unit
    assert preprocessed_image.data_unit == tilted_image.data_unit
    substrate = image.values == 0
    lens = ~substrate
    np.testing.assert_allclose(preprocessed_image.values[substrate], 0, atol=1e-8)
    np.testing.assert_allclose(
        preprocessed_image.values[lens], image.values[lens], atol=1e-8
    )
    np.testing.assert_array_equal(tilted_image.values, tilted_values)
    assert use_case.preprocessing_completed is True


def test_process_image_WHEN_not_preprocessed_THEN_preprocesses_before_raising():
    image = Image(np.array([[1.0, 2.0], [3.0, 4.0]]), 2.0, 3.0, "um", "nm")
    use_case = UseCaseMicrolens(image)
    use_case.load_settings(DEFAULT_SETTINGS_PATH)

    with pytest.raises(NotImplementedError):
        use_case.process_image()

    assert use_case.preprocessing_completed is True
    assert use_case.preprocessed_image is not None
    assert use_case.preprocessed_image is not image
    assert use_case.processing_completed is False


def test_preprocess_image_WHEN_rerun_after_processing_flag_set_THEN_resets_processing_and_processing_stays_incomplete():
    image = Image(np.array([[1.0, 2.0], [3.0, 4.0]]), 2.0, 3.0, "um", "nm")
    use_case = UseCaseMicrolens(image)
    use_case.load_settings(DEFAULT_SETTINGS_PATH)
    use_case.preprocess_image()
    use_case.processing_completed = True

    use_case.preprocess_image()

    assert use_case.processing_completed is False
    with pytest.raises(NotImplementedError):
        use_case.process_image()
    assert use_case.processing_completed is False


def test_load_settings_WHEN_valid_yaml_is_loaded_THEN_stores_settings_and_resets_state():
    image = Image(np.array([[1.0]]), 2.0, 3.0, "um", "nm")
    use_case = UseCaseMicrolens(image)
    use_case.preprocessed_image = image
    use_case.preprocessing_completed = True
    use_case.processing_completed = True
    expected = MicrolensSettings.from_dict(read_yaml(DEFAULT_SETTINGS_PATH))

    use_case.load_settings(DEFAULT_SETTINGS_PATH)

    assert use_case.settings == expected
    assert use_case.preprocessed_image is None
    assert use_case.preprocessing_completed is False
    assert use_case.processing_completed is False


def test_load_settings_WHEN_yaml_or_settings_are_invalid_THEN_preserves_state(
    tmp_path,
):
    image = Image(np.array([[1.0]]), 2.0, 3.0, "um", "nm")
    use_case = UseCaseMicrolens(image)
    use_case.load_settings(DEFAULT_SETTINGS_PATH)
    existing_settings = use_case.settings
    use_case.preprocessed_image = image
    use_case.preprocessing_completed = True
    use_case.processing_completed = True

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("preprocessing: [", encoding="utf-8")
    invalid_settings = tmp_path / "invalid_settings.yaml"
    write_yaml(invalid_settings, {"preprocessing": {"method": "substrate"}})

    for path in (invalid_yaml, invalid_settings):
        with pytest.raises((ValueError, TypeError)):
            use_case.load_settings(path)
        assert use_case.settings is existing_settings
        assert use_case.preprocessed_image is image
        assert use_case.preprocessing_completed is True
        assert use_case.processing_completed is True


def test_save_settings_WHEN_settings_are_loaded_THEN_round_trips_without_changing_state(
    tmp_path,
):
    image = Image(np.array([[1.0]]), 2.0, 3.0, "um", "nm")
    use_case = UseCaseMicrolens(image)
    with pytest.raises(RuntimeError, match="loaded"):
        use_case.preprocess_image()
    with pytest.raises(RuntimeError, match="loaded"):
        use_case.save_settings(tmp_path / "not-loaded.yaml")

    use_case.load_settings(DEFAULT_SETTINGS_PATH)
    original_settings = use_case.settings
    use_case.preprocessed_image = image
    use_case.preprocessing_completed = True
    use_case.processing_completed = True
    saved_path = tmp_path / "saved.yaml"

    use_case.save_settings(saved_path)

    assert use_case.settings is original_settings
    assert use_case.preprocessed_image is image
    assert use_case.preprocessing_completed is True
    assert use_case.processing_completed is True
    assert MicrolensSettings.from_dict(read_yaml(saved_path)) == original_settings
