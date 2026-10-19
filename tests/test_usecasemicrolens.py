from pathlib import Path

import numpy as np
import pytest

from microlens.image import Image
from microlens.settings import read_yaml, write_yaml
from microlens.usecasemicrolens import MicrolensSettings, UseCaseMicrolens


DEFAULT_SETTINGS_PATH = Path(__file__).parents[1] / "configs" / "settings.yaml"


def test_use_case_microlens_WHEN_initialized_THEN_references_original_with_incomplete_states():
    image = Image(np.array([[1.0, 2.0], [3.0, 4.0]]), 2.0, 3.0, "um", "nm")

    use_case = UseCaseMicrolens(image)

    assert use_case.image is image
    assert use_case.preprocessed_image is None
    assert use_case.preprocessing_completed is False
    assert use_case.processing_completed is False
    assert use_case.settings is None


def test_preprocess_image_WHEN_called_THEN_copies_values_and_preserves_metadata():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    image = Image(values, 2.0, 3.0, "um", "nm")
    use_case = UseCaseMicrolens(image)
    use_case.load_settings(DEFAULT_SETTINGS_PATH)

    use_case.preprocess_image()

    preprocessed_image = use_case.preprocessed_image
    assert isinstance(preprocessed_image, Image)
    assert preprocessed_image is not image
    assert preprocessed_image.values is not image.values
    np.testing.assert_array_equal(preprocessed_image.values, image.values)
    assert preprocessed_image.pixel_size_x == image.pixel_size_x
    assert preprocessed_image.pixel_size_y == image.pixel_size_y
    assert preprocessed_image.xy_unit == image.xy_unit
    assert preprocessed_image.data_unit == image.data_unit
    np.testing.assert_array_equal(image.values, values)
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
