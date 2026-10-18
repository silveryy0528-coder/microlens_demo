import numpy as np
import pytest

from microlens.image import Image
from microlens.usecasemicrolens import UseCaseMicrolens


def test_use_case_microlens_WHEN_initialized_THEN_references_original_with_incomplete_states():
    image = Image(np.array([[1.0, 2.0], [3.0, 4.0]]), 2.0, 3.0, "um", "nm")

    use_case = UseCaseMicrolens(image)

    assert use_case.image is image
    assert use_case.preprocessed_image is None
    assert use_case.preprocessing_completed is False
    assert use_case.processing_completed is False


def test_preprocess_image_WHEN_called_THEN_copies_values_and_preserves_metadata():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    image = Image(values, 2.0, 3.0, "um", "nm")
    use_case = UseCaseMicrolens(image)

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

    with pytest.raises(NotImplementedError):
        use_case.process_image()

    assert use_case.preprocessing_completed is True
    assert use_case.preprocessed_image is not None
    assert use_case.preprocessed_image is not image
    assert use_case.processing_completed is False


def test_preprocess_image_WHEN_rerun_after_processing_flag_set_THEN_resets_processing_and_processing_stays_incomplete():
    image = Image(np.array([[1.0, 2.0], [3.0, 4.0]]), 2.0, 3.0, "um", "nm")
    use_case = UseCaseMicrolens(image)
    use_case.preprocess_image()
    use_case.processing_completed = True

    use_case.preprocess_image()

    assert use_case.processing_completed is False
    with pytest.raises(NotImplementedError):
        use_case.process_image()
    assert use_case.processing_completed is False
