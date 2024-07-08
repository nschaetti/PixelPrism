
# Imports
import cv2

# Local imports
from pixel_prism.effects.effects import EffectBase
from pixel_prism.primitives import Point


class SIFTPoint(Point):
    """
    SIFT point class that extends the Point class with additional SIFT attributes.
    """

    def __init__(
            self,
            x,
            y,
            size,
            angle,
            response,
            octave,
            class_id
    ):
        """
        Initialize the SIFT point with its coordinates, size, angle, response, octave, and class ID.

        Args:
            x (int): X-coordinate of the point
            y (int): Y-coordinate of the point
            size (int): Size of the point
            angle (float): Angle of the point
            response (float): Response of the point
            octave (int): Octave of the point
            class_id (int): Class ID of the point
        """
        super().__init__(x, y, size)
        self.angle = angle
        self.response = response
        self.octave = octave
        self.class_id = class_id
    # end __init__

    def __repr__(self):
        return f"SIFTPoint(x={self.x}, y={self.y}, size={self.size}, angle={self.angle}, response={self.response}, octave={self.octave}, class_id={self.class_id})"
    # end __repr__

# end SIFTPoint


class SIFTPointsEffect(EffectBase):

    def __init__(
            self,
            num_octaves=4,
            num_scales=3
    ):
        """
        Initialize the SIFT effect with the number of octaves and scales to use

        Args:
            num_octaves (int): Number of octaves to use in SIFT
            num_scales (int): Number of scales to use in SIFT
        """
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sift = cv2.SIFT_create(nOctaveLayers=num_scales)
    # end __init__

    def apply(
            self,
            image,
            **kwargs
    ):
        """
        Apply the SIFT effect to the image
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = self.sift.detect(gray_image, None)

        sift_points = []
        for kp in keypoints:
            point = SIFTPoint(
                x=kp.pt[0],
                y=kp.pt[1],
                size=kp.size,
                angle=kp.angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id
            )
            sift_points.append(point)
        # end for

        return sift_points
    # end apply

# end SIFTPointsEffect

