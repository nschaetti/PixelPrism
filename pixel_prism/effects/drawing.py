
# Import necessary packages
import cv2

from .effect_base import EffectBase


class DrawPointsEffect(EffectBase):
    """
    Effect to draw points on an image
    """

    def __init__(
            self,
            points,
            color=(0, 255, 0),
            thickness=1
    ):
        """
        Initialize the draw points effect with the points to draw

        Args:
            points (list): List of points to draw
            color (tuple): Color of the points to draw
            thickness (int): Thickness of the points to draw
        """
        self.points = points
        self.color = color
        self.thickness = thickness
    # end __init__

    def apply(self, image, **kwargs):
        for point in self.points:
            cv2.circle(image, (int(point.x), int(point.y)), int(point.size / 2), self.color, self.thickness)
        return image

# end DrawPointsEffect

