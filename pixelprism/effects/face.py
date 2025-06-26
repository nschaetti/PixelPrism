

# Imports
import cv2

# Local
from pixelprism.effects import EffectBase, DrawPointsEffect
from pixelprism.primitives import Point
import pixelprism.effects.functional as F


class FacePoint(Point):

    def __init__(
            self,
            x,
            y,
            size: int = 1
    ):
        """
        Initialize the face point with its coordinates and size

        Args:
            x (int): X-coordinate of the point
            y (int): Y-coordinate of the point
            size (int): Size of the point
        """
        super().__init__(x, y, size)
    # end __init__

    def __repr__(self):
        """
        String representation of the face point
        """
        return f"FacePoint(x={self.x}, y={self.y}, size={self.size})"
    # end FacePoint

# end FacePoint


class FacePointsEffect(EffectBase):
    """
    Face points effect to detect facial landmarks on an image
    """

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.landmark_model = cv2.face.createFacemarkLBF()
        self.landmark_model.loadModel(cv2.data.haarcascades + "lbfmodel.yaml")
    # end __init__

    def apply(
            self,
            image,
            **kwargs
    ):
        """
        Apply the face points effect to the image

        Args:
            image_obj (Image): Image object to apply the effect to
            input_layers (list): List of input layer names
            output_layers (list): List of output layer names
            kwargs: Additional keyword arguments
        """
        faces = F.face_detection(
            image,
            self.face_cascade,
            self.landmark_model,
        )
    # end apply

# end FacePointsEffect

