"""Use case orchestration for microlens image processing."""
import copy

from .image import Image


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

    def preprocess_image(self) -> None:
        """Create an independent copy of the original image."""
        self.processing_completed = False
        self.preprocessed_image = copy.deepcopy(self.image)
        self.preprocessing_completed = True

    def process_image(self) -> None:
        """Prepare the image and indicate that processing is not implemented."""
        if not self.preprocessing_completed:
            self.preprocess_image()
        raise NotImplementedError("Image processing steps are not implemented.")
