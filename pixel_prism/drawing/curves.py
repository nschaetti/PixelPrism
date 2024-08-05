

# Imports
import numpy as np

from pixel_prism.animate.able import MovableMixin
from pixel_prism.data import Point2D
from pixel_prism.utils import random_color

from .rectangles import Rectangle
from .drawablemixin import DrawableMixin


# A Cubic Bezier curve
# CubicBezier(start=(3.716065-3.765878j), control1=(3.536737-4.134496j), control2=(3.247821-4.403487j), end=(2.799502-4.403487j))
class CubicBezierCurve(DrawableMixin, MovableMixin):
    """
    A class to represent a cubic Bezier curve in 2D space.
    """

    def __init__(
            self,
            start: Point2D,
            control1: Point2D,
            control2: Point2D,
            end: Point2D,
            bbox: Rectangle = None
    ):
        """
        Initialize the curve with its start, control1, control2, and end points.

        Args:
            start (Point2D): Start point of the curve
            control1 (Point2D): First control point of the curve
            control2 (Point2D): Second control point of the curve
            end (Point2D): End point of the curve
        """
        super().__init__()
        self.start = start
        self.control1 = control1
        self.control2 = control2
        self.end = end
        self.bbox = bbox
    # end __init__

    @property
    def width(self):
        """
        Get the width of the curve.
        """
        if self.bbox is None:
            return None
        # end if
        return self.bbox.width
    # end width

    @property
    def height(self):
        """
        Get the height of the curve.
        """
        if self.bbox is None:
            return None
        # end
        return self.bbox.height
    # end height

    # Move
    def translate(self, dx: float, dy: float):
        """
        Move the path by a given displacement.

        Args:
            dx (float): Displacement in the X-direction
            dy (float): Displacement in the Y-direction
        """
        # Translate the points
        self.start.x += dx
        self.start.y += dy
        self.control1.x += dx
        self.control1.y += dy
        self.control2.x += dx
        self.control2.y += dy
        self.end.x += dx
        self.end.y += dy

        # Translate the bounding box
        if self.bbox is not None:
            self.bbox.translate(dx, dy)
        # end
    # end translate

    # Draw path (for debugging)
    def draw_path(self, context):
        """
        Draw the path to the context.
        """
        # Select a random int
        color = random_color()

        # Save the context
        context.save()

        # Set the color
        context.set_source_rgb(color.red, color.green, color.blue)

        # Draw the path
        context.move_to(self.start.x, self.start.y)
        context.curve_to(
            self.control1.x,
            self.control1.y,
            self.control2.x,
            self.control2.y,
            self.end.x,
            self.end.y
        )

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
        Draw the curve to the context.
        """
        # context.move_to(self.start.x, self.start.y)
        context.curve_to(
            self.control1.x,
            self.control1.y,
            self.control2.x,
            self.control2.y,
            self.end.x,
            self.end.y
        )
    # end draw

    # str
    def __str__(self):
        """
        Get the string representation of the curve.
        """
        return (
            f"CubicBezier(start={self.start},control1={self.control1},control2={self.control2},end={self.end})"
        )
    # end __str__

    # repr
    def __repr__(self):
        """
        Get the string representation of the curve.
        """
        return (
            f"CubicBezier(start={self.start},control1={self.control1},control2={self.control2},end={self.end})"
        )
    # end __repr__

    @classmethod
    def from_2d(
            cls,
            start_x: float,
            start_y: float,
            control1_x: float,
            control1_y: float,
            control2_x: float,
            control2_y: float,
            end_x: float,
            end_y: float
    ):
        """
        Create a cubic Bezier curve from scalar values.
        """
        return cls(
            Point2D(start_x, start_y),
            Point2D(control1_x, control1_y),
            Point2D(control2_x, control2_y),
            Point2D(end_x, end_y)
        )
    # end from_2d

# end CubicBezierCurve


# A quadratic Bezier curve
# QuadraticBezier(start=(4.313823-6.127024j), control=(3.536737-4.134496j), end=(2.799502-4.403487j))
class QuadraticBezierCurve(DrawableMixin, MovableMixin):
    """
    A class to represent a quadratic Bezier curve in 2D space.
    """

    def __init__(
            self,
            start: Point2D,
            control: Point2D,
            end: Point2D,
            bbox: Rectangle = None
    ):
        """
        Initialize the curve with its start, control1, control2, and end points.

        Args:
            start (Point2D): Start point of the curve
            control (Point2D): First control point of the curve
            end (Point2D): End point of the curve
            bbox (Rectangle): Bounding box of the curve
        """
        super().__init__()
        self.start = start
        self.control = control
        self.end = end
        self.bbox = bbox
    # end __init__

    @property
    def width(self):
        """
        Get the width of the curve.
        """
        if self.bbox is None:
            return None
        # end if
        return self.bbox.width
    # end width

    @property
    def height(self):
        """
        Get the height of the curve.
        """
        if self.bbox is None:
            return None
        # end
        return self.bbox.height
    # end height

    # Move
    def translate(self, dx: float, dy: float):
        """
        Move the path by a given displacement.

        Args:
            dx (float): Displacement in the X-direction
            dy (float): Displacement in the Y-direction
        """
        # Translate the points
        self.start.x += dx
        self.start.y += dy
        self.control.x += dx
        self.control.y += dy
        self.end.x += dx
        self.end.y += dy

        # Translate the bounding box
        if self.bbox is not None:
            self.bbox.translate(dx, dy)
        # end
    # end translate

    # Draw path (for debugging)
    def draw_path(self, context):
        """
        Draw the path to the context.
        """
        # Select a random int
        color = random_color()

        # Save the context
        context.save()

        # Set the color
        context.set_source_rgb(color.red, color.green, color.blue)

        # Draw the path
        context.move_to(self.start.x, self.start.y)
        context.curve_to(
            self.control.x,
            self.control.y,
            self.control.x,
            self.control.y,
            self.end.x,
            self.end.y
        )

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
        Draw the curve to the context.
        """
        # context.move_to(self.start.x, self.start.y)
        context.curve_to(
            self.control.x,
            self.control.y,
            self.control.x,
            self.control.y,
            self.end.x,
            self.end.y
        )
    # end draw

    @classmethod
    def from_2d(
            cls,
            start_x: float,
            start_y: float,
            control_x: float,
            control_y: float,
            end_x: float,
            end_y: float
    ):
        """
        Create a cubic Bezier curve from scalar values.
        """
        return cls(
            Point2D(start_x, start_y),
            Point2D(control_x, control_y),
            Point2D(end_x, end_y)
        )
    # end from_2d

# end QuadraticBezierCurve
