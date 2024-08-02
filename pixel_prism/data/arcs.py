#
#
#

# Imports
import numpy as np
from pixel_prism.animate.able import MovAble

from .data import Data
from .points import Point2D, Point
from .scalar import Scalar


# An arc
class ArcData(Data, MovAble):
    """
    A class to represent a cubic Bezier curve in 2D space.
    """

    def __init__(
            self,
            center: Point2D,
            radius: Scalar,
            start_angle: Scalar,
            end_angle: Scalar
    ):
        """
        Initialize the arc with its center, radius, start angle, and end angle.

        Args:
            center (Point2D): Center of the arc
            radius (Scalar): Radius of the arc
            start_angle (Scalar): Start angle of the arc
            end_angle (Scalar): End angle of the arc
        """
        super().__init__()
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
    # end __init__

    # Get
    def get(self):
        """
       Get the center, radius, start angle, and end angle of the arc.

        Returns:
            tuple: Start, control1, control2, and end points of the curve
        """
        return self.center, self.radius, self.start_angle, self.end_angle
    # end get

    # Set
    def set(self, center: Point2D, radius: Scalar, start_angle: Scalar, end_angle: Scalar):
        """
        Set the center, radius, start angle, and end angle of the arc.

        Args:
            center (Point2D): Center of the arc
            radius (Scalar): Radius of the arc
            start_angle (Scalar): Start angle of the arc
            end_angle (Scalar): End angle of the arc
        """
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
    # end set

    # Set center
    def set_center(self, x: float, y: float):
        """
        Set the center of the arc.

        Args:
            x (float): X-coordinate of the center
            y (float): Y-coordinate of the center
        """
        self.center.x = x
        self.center.y = y
    # end set_center

    # Set radius
    def set_radius(self, radius: float):
        """
        Set the radius of the arc.

        Args:
            radius (float): Radius of the arc
        """
        self.radius.set(radius)
    # end set_radius

    # Set start angle
    def set_start_angle(self, angle: float):
        """
        Set the start angle of the arc.

        Args:
            angle (float): Start angle of the arc
        """
        self.start_angle.set(angle)
    # end set_start_angle

    # Set end angle
    def set_end_angle(self, angle: float):
        """
        Set the end angle of the arc.

        Args:
            angle (float): End angle of the arc
        """
        self.end_angle.set(angle)
    # end set_end_angle

    @classmethod
    def from_scalar(
            cls,
            center_x: float,
            center_y: float,
            radius: float,
            start_angle: float,
            end_angle: float
    ):
        """
        Create an arc from scalar values.

        Args:
            center_x (float): X-coordinate of the center
            center_y (float): Y-coordinate of the center
            radius (float): Radius of the arc
            start_angle (float): Start angle of the arc
            end_angle (float): End angle of the arc

        Returns:
            ArcData: Arc created from scalar values
        """
        return cls(
            Point2D(center_x, center_y),
            Scalar(radius),
            Scalar(start_angle),
            Scalar(end_angle)
        )
    # end from_scalar

# end ArcData


