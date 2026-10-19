"""Use case orchestration for microlens image processing."""
import copy
from dataclasses import dataclass

from .image import Image
from .settings import PreprocessingSettings


@dataclass(frozen=True)
class MicrolensSettings:
    """Settings for a microlens characterization run."""

    preprocessing: PreprocessingSettings

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
