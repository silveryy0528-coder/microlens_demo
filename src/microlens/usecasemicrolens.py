"""Use case orchestration for microlens image processing."""
import copy
from dataclasses import dataclass

from microlens.image import Image
from microlens.settings import PreprocessingSettings, SubstrateSettings


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
        print(data, settings_data)

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

    def preprocess_image(self) -> None:
        self.processing_completed = False
        self.preprocessed_image = self.background_correction(self.image)
        self.preprocessing_completed = True

    @staticmethod
    def background_correction(image: Image) -> Image:
        image_out = copy.deepcopy(image)
        return image_out

    def process_image(self) -> None:
        """Prepare the image and indicate that processing is not implemented."""
        if not self.preprocessing_completed:
            self.preprocess_image()
        raise NotImplementedError("Image processing steps are not implemented.")
