
# Imports
import numpy as np
from pixel_prism.animate.able import MovAble
from pixel_prism.data import Point2D

from .drawablemixin import DrawableMixin


# A line
# Line(start=(4.313823-6.127024j), end=(3.716065-3.765878j))
class Line(DrawableMixin, MovAble):
    """
    A class to represent a line in 2D space.
    """

    def __init__(
            self,
            start: Point2D,
            end: Point2D,
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

    # Draw the element
    def draw(self, context):
        """
        Draw the line to the context.
        """
        # context.move_to(self.start.x, self.start.y)
        context.line_to(self.end.x, self.end.y)
    # end draw

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

# end LineData

