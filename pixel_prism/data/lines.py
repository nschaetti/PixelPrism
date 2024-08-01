
# Imports
import numpy as np
from pixel_prism.animate.able import MovAble

from .data import Data
from .points import Point


# A line
# Line(start=(4.313823-6.127024j), end=(3.716065-3.765878j))
class Line(Data, MovAble):
    """
    A class to represent a line in 2D space.
    """

    def __init__(
            self,
            start: Point,
            end: Point
    ):
        """
        Initialize the line with its start and end points.

        Args:
            start (Point2D): Start point of the line
            end (Point2D): End point of the line
        """
        super().__init__()
        self.start = start
        self.end = end
    # end __init__

    def length(self):
        """
        Get the length of the line.

        Returns:
            float: Length of the line
        """
        return np.linalg.norm(self.end.pos - self.start.pos)
    # end length

    # Get
    def get(self):
        """
        Get the start and end points of the line.

        Returns:
            tuple: Start and end points of the line
        """
        return self.start, self.end
    # end get

    # Set
    def set(self, start: Point, end: Point):
        """
        Set the start and end points of the line.

        Args:
            start (Point2D): Start point of the line
            end (Point2D): End point of the line
        """
        self.start = start
        self.end = end
    # end set

    # str
    def __str__(self):
        """
        Get the string representation of the line.

        Returns:
            str: String representation of the line
        """
        return f"Line(start={self.start.pos}, end={self.end.pos})"
    # end __str__

    # repr
    def __repr__(self):
        """
        Get the string representation of the line.

        Returns:
            str: String representation of the line
        """
        return f"Line(start={self.start.pos}, end={self.end.pos})"
    # end __repr__

# end Line

