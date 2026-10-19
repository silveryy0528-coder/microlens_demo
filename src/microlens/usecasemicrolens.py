"""Use case orchestration for microlens image processing."""
import copy
from dataclasses import asdict, dataclass

from microlens.algorithms import fit_and_subtract_plane, select_substrate_mask
from microlens.image import Image
from microlens.settings import (
    PreprocessingSettings,
    SubstrateSettings,
    read_yaml,
    write_yaml,
)


@dataclass(frozen=True)
class MicrolensSettings:
    """Settings for a microlens characterization run."""

    preprocessing: PreprocessingSettings

    @classmethod
    def from_dict(cls, data: dict) -> "MicrolensSettings":
        """Construct microlens settings from a nested dictionary."""
        if not isinstance(data, dict):
            raise TypeError("settings data must be a dictionary")

        settings_data = data.copy()
        preprocessing_data = settings_data.get("preprocessing")
        if isinstance(preprocessing_data, dict):
            preprocessing_data = preprocessing_data.copy()
            substrate_data = preprocessing_data.get("substrate")
            if isinstance(substrate_data, dict):
                preprocessing_data["substrate"] = SubstrateSettings(**substrate_data)
            settings_data["preprocessing"] = PreprocessingSettings(
                **preprocessing_data
            )

        return cls(**settings_data)

    def __post_init__(self) -> None:
        if not isinstance(self.preprocessing, PreprocessingSettings):
            raise TypeError("preprocessing must be a PreprocessingSettings instance")


class UseCaseMicrolens:
    """Preprocess and process a microlens height map.

    Parameters
    ----------
    image : Image
        Original height map.
    """

    def __init__(self, image: Image) -> None:
        self.image = image
        self.preprocessed_image: Image | None = None
        self.preprocessing_completed = False
        self.processing_completed = False
        self.settings = None

    def load_settings(self, path) -> None:
        """Load validated settings and invalidate previous processing results."""
        settings = MicrolensSettings.from_dict(read_yaml(path))
        self.settings = settings
        self.preprocessed_image = None
        self.preprocessing_completed = False
        self.processing_completed = False

    def save_settings(self, path) -> None:
        """Save the currently loaded settings."""
        if self.settings is None:
            raise RuntimeError("settings must be loaded before they can be saved")
        write_yaml(path, asdict(self.settings))

    def preprocess_image(self) -> None:
        if self.settings is None:
            raise RuntimeError("settings must be loaded before preprocessing")
        self.processing_completed = False
        self.preprocessing_completed = False
        self.preprocessed_image = None
        preprocessed_image = self.background_correction(
            self.image, self.settings.preprocessing
        )
        self.preprocessed_image = preprocessed_image
        self.preprocessing_completed = True

    @staticmethod
    def background_correction(
        image: Image, settings: PreprocessingSettings
    ) -> Image:
        """Subtract a fitted background plane from a height map."""
        image_out = copy.deepcopy(image)
        pixel_size = (image.pixel_size_x, image.pixel_size_y)
        if settings.method == "substrate":
            preliminary, _ = fit_and_subtract_plane(image.values, pixel_size)
            substrate_mask = select_substrate_mask(
                preliminary,
                percentile=settings.substrate.percentile,
                distance=settings.substrate.distance,
            )
            corrected, _ = fit_and_subtract_plane(
                image.values, pixel_size, mask=substrate_mask
            )
        else:
            corrected, _ = fit_and_subtract_plane(image.values, pixel_size)
        image_out.values = corrected

        return image_out

    def process_image(self) -> None:
        """Prepare the image and indicate that processing is not implemented."""
        if not self.preprocessing_completed:
            self.preprocess_image()
        raise NotImplementedError("Image processing steps are not implemented.")


if __name__ == "__main__":
    import numpy as np
    image_data = np.zeros((9, 9))
    image_data[:, 2:4] = 10
    # make image_data tilted
    rows, columns = np.indices(image_data.shape)
    image_data = image_data + 0.1 * columns - 0.2 * rows
    from microlens.image import Image
    test_image = Image(image_data, 1.0, 1.0, "um", "nm")
    use_case = UseCaseMicrolens(test_image)
    use_case.load_settings(r'C:\Users\guoya\Documents\Git_repo\microlens\configs\settings.yaml')
    use_case.preprocess_image()
    preprocessed_data = use_case.preprocessed_image.values
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(image_data, cmap='gray')
    fig.add_subplot(1,2,2)
    plt.imshow(preprocessed_data, cmap='gray')
    plt.show()