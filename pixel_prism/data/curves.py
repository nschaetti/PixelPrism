

# Imports
import numpy as np
from pixel_prism.animate.able import MovAble
from .data import Data
from .points import Point, Point2D


# A Cubic Bezier curve
# CubicBezier(start=(3.716065-3.765878j), control1=(3.536737-4.134496j), control2=(3.247821-4.403487j), end=(2.799502-4.403487j))
class CubicBezierCurve(Data, MovAble):
    """
    A class to represent a cubic Bezier curve in 2D space.
    """

    def __init__(
            self,
            start: Point,
            control1: Point,
            control2: Point,
            end: Point
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
    # end __init__

    # Get
    def get(self):
        """
        Get the start, control1, control2, and end points of the curve.

        Returns:
            tuple: Start, control1, control2, and end points of the curve
        """
        return self.start, self.control1, self.control2, self.end
    # end get

    # Set
    def set(self, start: Point, control1: Point, control2: Point, end: Point):
        """
        Set the start, control1, control2, and end points of the curve.

        Args:
            start (Point2D): Start point of the curve
            control1 (Point2D): First control point of the curve
            control2 (Point2D): Second control point of the curve
            end (Point2D): End point of the curve
        """
        self.start = start
        self.control1 = control1
        self.control2 = control2
        self.end = end
    # end set

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
class QuadraticBezierCurve(Data, MovAble):
    """
    A class to represent a quadratic Bezier curve in 2D space.
    """

    def __init__(
            self,
            start: Point,
            control: Point,
            end: Point
    ):
        """
        Initialize the curve with its start, control1, control2, and end points.

        Args:
            start (Point2D): Start point of the curve
            control (Point2D): First control point of the curve
            end (Point2D): End point of the curve
        """
        super().__init__()
        self.start = start
        self.control = control
        self.end = end
    # end __init__

    # Get
    def get(self):
        """
        Get the start, control, and end points of the curve.

        Returns:
            tuple: Start, control, and end points of the curve
        """
        return self.start, self.control, self.end
    # end get

    # Set
    def set(self, start: Point, control: Point, end: Point):
        """
        Set the start, control1, control2, and end points of the curve.

        Args:
            start (Point2D): Start point of the curve
            control (Point2D): First control point of the curve
            end (Point2D): End point of the curve
        """
        self.start = start
        self.control = control
        self.end = end
    # end set

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
