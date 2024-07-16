
# Import necessary packages
import cv2
import numpy as np

import pixel_prism.effects.functional as F
from .effect_base import EffectBase
from pixel_prism.base.image import Image


class DrawPointsEffect(EffectBase):
    """
    Effect to draw points on an image
    """

    def __init__(
            self,
            color=(255, 255, 0),
            thickness=1
    ):
        """
        Initialize the draw points effect with the points to draw

        Args:
            color (tuple): Color of the points to draw
            thickness (int): Thickness of the points to draw
        """
        self.color = color
        self.thickness = thickness
    # end __init__

    def apply(
            self,
            image: Image,
            **kwargs
    ):
        """
        Apply the draw points effect to the image

        Args:
            image (Image): Image to apply the effect to
            kwargs: Additional keyword arguments
        """
        # Assert points are provided
        if 'points' not in kwargs:
            raise ValueError("Points must be provided to draw on the image")
        # end if

        # Points
        points = kwargs['points']

        return F.draw_points(image, points, self.color, self.thickness)
    # end apply

# end DrawPointsEffect

