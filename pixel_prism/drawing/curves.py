

# Imports
import numpy as np

from pixel_prism.animate.able import MovableMixin
from pixel_prism.data import Point2D
from pixel_prism.utils import random_color

from .rectangles import Rectangle
from .drawablemixin import DrawableMixin


# A Cubic Bezier curve
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
            bounding_box: Rectangle = None
    ):
        """
        Initialize the curve with its start, control1, control2, and end points.

        Args:
            start (Point2D): Start point of the curve
            control1 (Point2D): First control point of the curve
            control2 (Point2D): Second control point of the curve
            end (Point2D): End point of the curve
        """
        # Constructor
        DrawableMixin.__init__(self, False)

        # Properties
        self.start = start
        self.control1 = control1
        self.control2 = control2
        self.end = end

        # Bounding box
        if bounding_box is not None:
            self.bbox = bounding_box
            self.has_bbox = True
        else:
            self.bbox = None
            self.has_bbox = False
        # end if

        # Calculate the length of the curve
        self.length = self._recursive_bezier_length(
            np.array([start.x, start.y]),
            np.array([control1.x, control1.y]),
            np.array([control2.x, control2.y]),
            np.array([end.x, end.y]),
            0
        )
    # end __init__

    # region PROPERTIES

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

    # endregion PROPERTIES

    # region PUBLIC

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

    def draw_partial_bezier_cubic(
            self,
            context,
            t
    ):
        """
        Draw a partial cubic Bezier curve from 0 to t.

        Args:
            context (cairo.Context): Context to draw the curve to
            t (float): Parameter
        """
        # Subdivide the curve at t
        p0, p01, p012, p0123, p123, p23, p3 = self._bezier_subdivide(t)

        # Draw the first part of the subdivided curve
        context.curve_to(
            p01.x,
            p01.y,
            p012.x,
            p012.y,
            p0123.x,
            p0123.y
        )
    # end draw_partial_bezier_cubic

    # Draw the element
    def draw(
            self,
            context,
            move_to: bool = False,
            build_ratio: float = 1.0
    ):
        """
        Draw the curve to the context.
        """
        # Move to the start point
        if move_to:
            context.move_to(self.start.x, self.start.y)
        # end if

        # Draw the curve
        if build_ratio == 1.0:
            context.curve_to(
                self.control1.x,
                self.control1.y,
                self.control2.x,
                self.control2.y,
                self.end.x,
                self.end.y
            )
        else:
            self.draw_partial_bezier_cubic(context, build_ratio)
        # end if
    # end draw

    # endregion PUBLIC

    # region PRIVATE

    def _recursive_bezier_length(self, p0, p1, p2, p3, depth, tolerance=1e-5):
        """
        Calculate the length of a cubic Bezier curve defined by points p0, p1, p2, p3.
        """
        # Calculate midpoints
        p01 = (p0 + p1) / 2
        p12 = (p1 + p2) / 2
        p23 = (p2 + p3) / 2
        p012 = (p01 + p12) / 2
        p123 = (p12 + p23) / 2
        p0123 = (p012 + p123) / 2

        # Calculate the lengths of the line segments
        l1 = np.linalg.norm(p0 - p3)
        l2 = np.linalg.norm(p0 - p1) + np.linalg.norm(p1 - p2) + np.linalg.norm(p2 - p3)

        # Subdivide if necessary
        if np.abs(l1 - l2) > tolerance:
            return (
                    self._recursive_bezier_length(p0, p01, p012, p0123, depth + 1, tolerance) +
                    self._recursive_bezier_length(p0123, p123, p23, p3, depth + 1, tolerance)
            )
        # end if

        return l2
    # end _recursive_bezier_length

    def _bezier_subdivide(
            self,
            t
    ):
        """
        Subdivide a cubic Bezier curve at parameter t.

        Args:

            t (float): Parameter
        """
        # Points
        p0 = np.array([self.start.x, self.start.y])
        p1 = np.array([self.control1.x, self.control1.y])
        p2 = np.array([self.control2.x, self.control2.y])
        p3 = np.array([self.end.x, self.end.y])

        # Calculate the points
        p01 = (1 - t) * p0 + t * p1
        p12 = (1 - t) * p1 + t * p2
        p23 = (1 - t) * p2 + t * p3
        p012 = (1 - t) * p01 + t * p12
        p123 = (1 - t) * p12 + t * p23
        p0123 = (1 - t) * p012 + t * p123

        return (
            Point2D(float(p0[0]), float(p0[1])),
            Point2D(float(p01[0]), float(p01[1])),
            Point2D(float(p012[0]), float(p012[1])),
            Point2D(float(p0123[0]), float(p0123[1])),
            Point2D(float(p123[0]), float(p123[1])),
            Point2D(float(p23[0]), float(p23[1])),
            Point2D(float(p3[0]), float(p3[1]))
        )
    # end _bezier_subdivide

    # endregion PRIVATE

    # region OVERRIDE

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

    # endregion OVERRIDE

    # region CLASS_METHODS

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

    # endregion CLASS_METHODS

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

        # Compute the length of the curve
        self.length = self.recursive_bezier_length(
            np.array([start.x, start.y]),
            np.array([control.x, control.y]),
            np.array([end.x, end.y]),
            0
        )
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

    def recursive_bezier_length(
            self,
            p0,
            p1,
            p2,
            depth,
            tolerance=1e-5
    ):
        """
        Calculate the length of a quadratic Bezier curve defined by points p0, p1, p2.

        Args:
            p0 (np.ndarray): Start point of the curve
            p1 (np.ndarray): Control point of the curve
            p2 (np.ndarray): End point of the curve
            depth (int): Depth of the recursion
            tolerance (float): Tolerance for the recursion
        """
        # Calculate midpoints
        p01 = (p0 + p1) / 2
        p12 = (p1 + p2) / 2
        p012 = (p01 + p12) / 2

        # Calculate the lengths of the line segments
        l1 = np.linalg.norm(p0 - p2)
        l2 = np.linalg.norm(p0 - p1) + np.linalg.norm(p1 - p2)

        # Subdivide if necessary
        if np.abs(l1 - l2) > tolerance:
            return self.recursive_bezier_length(p0, p01, p012, depth + 1, tolerance) + self.recursive_bezier_length(p012, p12, p2, depth + 1)
        # end if

        return l2
    # end recursive_bezier_length

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
    def draw(
            self,
            context
    ):
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
