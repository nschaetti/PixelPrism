
# Imports
import numpy as np
from pixel_prism.animate.able import MovableMixin
from pixel_prism.data import Point2D
from pixel_prism.utils import random_color

from .rectangles import Rectangle
from .drawablemixin import DrawableMixin


# A line
# Line(start=(4.313823-6.127024j), end=(3.716065-3.765878j))
class Line(DrawableMixin, MovableMixin):
    """
    A class to represent a line in 2D space.
    """

    def __init__(
            self,
            start: Point2D,
            end: Point2D,
            bbox: Rectangle = None
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
        self.bbox = bbox
    # end __init__

    def length(self):
        """
        Get the length of the line.

        Returns:
            float: Length of the line
        """
        return np.linalg.norm(self.end.pos - self.start.pos)
    # end length

    # Move
    def translate(self, dx: float, dy: float):
        """
        Move the path by a given displacement.

        Args:
            dx (float): Displacement in the X-direction
            dy (float): Displacement in the Y-direction
        """
        # Translate the start and end points
        self.start.x += dx
        self.start.y += dy
        self.end.x += dx
        self.end.y += dy

        # Translate the bounding box
        if self.bbox is not None:
            self.bbox.translate(dx, dy)
        # end if
    # end translate

    # Draw path (for debugging)
    def draw_path(self, context):
        """
        Draw the path to the context.

        Args:
            context (cairo.Context): Context to draw the path to
        """
        # Select a random int
        color = random_color()

        # Save the context
        context.save()

        # Set the color
        context.set_source_rgb(color.red, color.green, color.blue)

        # Draw the path
        context.move_to(self.start.x, self.start.y)
        context.line_to(self.end.x, self.end.y)

        # Set the line width
        context.set_line_width(0.1)

        # Stroke the path
        context.stroke()

        # Restore the context
        context.restore()
    # end draw_path

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
        if self.bbox is None:
            return f"Line(start={self.start.pos}, end={self.end.pos})"
        else:
            return f"Line(start={self.start.pos}, end={self.end.pos}, bbox={self.bbox})"
        # end if
    # end __str__

    # repr
    def __repr__(self):
        """
        Get the string representation of the line.

        Returns:
            str: String representation of the line
        """
        return Line.__str__(self)
    # end __repr__

# end Line

