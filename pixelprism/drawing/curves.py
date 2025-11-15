# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2024 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

# Imports
import math
from typing import Any
import numpy as np
from pixelprism import p2, s
from pixelprism.animate import MovableMixin, CallableMixin, animeattr
from pixelprism.math import Point2D, Scalar, Color
from pixelprism.utils import random_color

# Local imports
from . import BoundingBox
from .boundingboxmixin import BoundingBoxMixin
from .rectangles import Rectangle
from .drawablemixin import DrawableMixin
from .. import utils
from ..base import Context


# A Cubic Bezier curve
@animeattr("start")
@animeattr("control1")
@animeattr("control2")
@animeattr("end")
@animeattr("position")
@animeattr("length")
@animeattr("line_width")
class CubicBezierCurve(
    DrawableMixin,
    BoundingBoxMixin,
    MovableMixin,
    CallableMixin
):
    """
    A class to represent a cubic Bezier curve in 2D space.
    """

    def __init__(
            self,
            start: Point2D,
            control1: Point2D,
            control2: Point2D,
            end: Point2D,
            position: Scalar = Scalar(0.0),
            path_length: Scalar = Scalar(1.0),
            line_width: Scalar = Scalar(0.0),
            line_color: Color = utils.WHITE,
            on_change=None
    ):
        """
        Initialize the curve with its start, control1, control2, and end points.

        Args:
            start (Point2D): Start point of the curve
            control1 (Point2D): First control point of the curve
            control2 (Point2D): Second control point of the curve
            end (Point2D): End point of the curve
            position (Scalar): Position of the curve
            path_length (Scalar): Length of the curve
            line_width (Scalar): Width of the line
            line_color (Color): Color of the line
            on_change (callable): Function to call when the arc
        """
        # Constructor
        DrawableMixin.__init__(self)
        MovableMixin.__init__(self)

        # Properties
        self._start = start
        self._control1 = control1
        self._control2 = control2
        self._end = end
        self._position = position
        self._path_length = path_length

        # Display properties
        self._line_width = line_width
        self._line_color = line_color

        # Beginning, end, middle, q1 and q2
        self._center = Point2D(0, 0)
        self._middle_point = Point2D(0, 0)
        self._q1_point = Point2D(0, 0)
        self._q3_point = Point2D(0, 0)
        self._x_minima = Point2D(0, 0)
        self._x_maxima = Point2D(0, 0)
        self._y_minima = Point2D(0, 0)
        self._y_maxima = Point2D(0, 0)

        # Curve points
        self._curve_start = Point2D(0, 0)
        self._curve_end = Point2D(0, 0)
        self._curve_center = Point2D(0, 0)
        self._curve_middle_point = Point2D(0, 0)
        self._curve_q1_point = Point2D(0, 0)
        self._curve_q3_point = Point2D(0, 0)
        self._curve_x_minima = Point2D(0, 0)
        self._curve_x_maxima = Point2D(0, 0)
        self._curve_y_minima = Point2D(0, 0)
        self._curve_y_maxima = Point2D(0, 0)
        self._curve_control1 = Point2D(0, 0)
        self._curve_control2 = Point2D(0, 0)

        # Bounding box mixin
        BoundingBoxMixin.__init__(self)

        # Update points and length
        self.update_data()

        # Movable
        self.start_control1 = None
        self.start_control2 = None
        self.start_end = None

        # Set event listeners
        self._start.add_event_listener("on_change", self._on_start_changed)
        self._control1.add_event_listener("on_change", self._on_control1_changed)
        self._control2.add_event_listener("on_change", self._on_control2_changed)
        self._end.add_event_listener("on_change", self._on_end_changed)
        self._position.add_event_listener("on_change", self._on_position_changed)
        self._path_length.add_event_listener("on_change", self._on_path_length_changed)

        # Update math
        self.update_data()

        # List of event listeners (per events)
        self.event_listeners = {
            "on_change": [] if on_change is None else [on_change]
        }
    # end __init__

    # region PROPERTIES

    @property
    def start(self) -> Point2D:
        """
        Get the start point of the curve.
        """
        return self._start
    # end start

    @start.setter
    def start(self, value: Point2D):
        """
        Set the start point of the curve.
        """
        self._start.x = value.x
        self._start.y = value.y
    # end start

    @property
    def control1(self) -> Point2D:
        """
        Get the first control point of the curve.
        """
        return self._control1
    # end control1

    @control1.setter
    def control1(self, value: Point2D):
        """
        Set the first control point of the curve.
        """
        self._control1.x = value.x
        self._control1.y = value.y
    # end control1

    # Absolute position of controle1
    @property
    def abs_control1(self) -> Point2D:
        return self.start + self.control1
    # end abs_control1

    @property
    def control2(self) -> Point2D:
        """
        Get the second control point of the curve.
        """
        return self._control2
    # end control2

    @control2.setter
    def control2(self, value: Point2D):
        """
        Set the second control point of the curve.
        """
        self._control2.x = value.x
        self._control2.y = value.y
    # end control2

    # Absolute position of control2
    @property
    def abs_control2(self) -> Point2D:
        return self.end + self.control2
    # end abs_control2

    @property
    def end(self) -> Point2D:
        """
        Get the end point of the curve.
        """
        return self._end
    # end end

    @end.setter
    def end(self, value: Point2D):
        """
        Set the end point of the curve.
        """
        self._end.x = value.x
        self._end.y = value.y
    # end end

    @property
    def length(self) -> float:
        """
        Get the length of the curve.
        """
        return self._length
    # end length

    @length.setter
    def length(self, value: float):
        """
        Set the length of the curve.
        """
        raise ValueError(f"{self.__class__.__name__} is cannot be set !")
    # end length

    @property
    def position(self) -> Scalar:
        """
        Get the position of the curve.
        """
        if self._position.value < 0.0:
            return Scalar(0.0)
        elif self._position.value > 1.0:
            return Scalar(1.0)
        # end if
        return self._position
    # end position

    @position.setter
    def position(self, value: Scalar):
        """
        Set the position of the curve.
        """
        self._position.value = value
    # end position

    @ property
    def path_length(self) -> Scalar:
        """
        Get the length of the curve.
        """
        if self._path_length.value < 0.0:
            return Scalar(0.0)
        elif self._path_length.value > 1.0:
            return Scalar(1.0)
        # end if
        return self._path_length
    # end path_length

    @path_length.setter
    def path_length(self, value: Scalar):
        """
        Set the length of the curve.
        """
        self._path_length.value = value
    # end path_length

    @property
    def line_width(self) -> Scalar:
        """
        Get the width of the line.
        """
        return self._line_width
    # end line_width

    @line_width.setter
    def line_width(self, value: Scalar):
        """
        Set the width of the line.
        """
        self._line_width.value = value
    # end line_width

    @property
    def line_color(self) -> Color:
        """
        Get the color of the line.
        """
        return self._line_color
    # end line_color

    @line_color.setter
    def line_color(self, value: Color):
        """
        Set the color of the line.
        """
        self._line_color.red = value.red
        self._line_color.green = value.green
        self._line_color.blue = value.blue
    # end line_color

    @property
    def curve_start(self) -> Point2D:
        """
        Get the start point of the curve.
        """
        return self._curve_start
    # end start_point

    @property
    def curve_end(self) -> Point2D:
        """
        Get the end point of the curve.
        """
        return self._curve_end
    # end curve_end

    @property
    def curve_control1(self) -> Point2D:
        """
        Get the first control point of the curve.
        """
        return self._curve_control1
    # end curve_control1

    @property
    def curve_abs_control1(self) -> Point2D:
        """
        Get the first control point of the curve.
        """
        # return self._curve_abs_control1
        return self._curve_start + self._curve_control1
    # end curve_abs_control1

    @property
    def curve_control2(self) -> Point2D:
        """
        Get the second control point of the curve.
        """
        return self._curve_control2
    # end curve_control2

    @property
    def curve_abs_control2(self) -> Point2D:
        """
        Get the second control point of the curve.
        """
        # return self._curve_abs_control2
        return self._curve_end + self._curve_control2
    # end curve_abs_control2

    @property
    def curve_center(self) -> Point2D:
        """
        Get the center point of the curve.
        """
        return self._curve_center
    # end curve_center

    @property
    def curve_middle_point(self) -> Point2D:
        """
        Get the middle point of the curve.
        """
        return self._curve_middle_point
    # end curve_middle_point

    @property
    def curve_q1_point(self) -> Point2D:
        """
        Get the Q1 point of the curve.
        """
        return self._curve_q1_point
    # end curve_q1_point

    @property
    def curve_q3_point(self) -> Point2D:
        """
        Get the Q3 point of the curve.
        """
        return self._curve_q3_point
    # end curve_q3_point

    @property
    def curve_x_minima(self) -> Point2D:
        """
        Get the X minima point of the curve.
        """
        return self._curve_x_minima
    # end curve_x_minima

    @property
    def curve_x_maxima(self) -> Point2D:
        """
        Get the X maxima point of the curve.
        """
        return self._curve_x_maxima
    # end curve_x_maxima

    @property
    def curve_y_minima(self) -> Point2D:
        """
        Get the Y minima point of the curve.
        """
        return self._curve_y_minima
    # end curve_y_minima

    @property
    def curve_y_maxima(self) -> Point2D:
        """
        Get the Y maxima point of the curve.
        """
        return self._curve_y_maxima
    # end curve_y_maxima

    @property
    def center(self) -> Point2D:
        """
        Get the center point of the curve.
        """
        return self._center
    # end center

    @property
    def middle_point(self) -> Point2D:
        """
        Get the middle point of the curve.
        """
        return self._middle_point
    # end middle_point

    @property
    def q1_point(self) -> Point2D:
        """
        Get the Q1 point of the curve.
        """
        return self._q1_point
    # end q1_point

    @property
    def q3_point(self) -> Point2D:
        """
        Get the Q3 point of the curve.
        """
        return self._q3_point
    # end q3_point

    # endregion PROPERTIES

    # region PUBLIC

    # Get a point on the curve
    def bezier(
            self,
            t,
            start=None,
            control1=None,
            control2=None,
            end=None
    ) -> Point2D:
        """
        Get a point on the curve at parameter t.

        Args:
            t (float): Parameter
            start (Point2D): Start point of the curve
            control1 (Point2D): First control point of the curve
            control2 (Point2D): Second control point of the curve
            end (Point2D): End point of the curve
        """
        start = self.start if start is None else start
        control1 = self.control1 if control1 is None else control1
        control2 = self.control2 if control2 is None else control2
        end = self.end if end is None else end

        P0 = np.array([start.x, start.y])
        P1 = P0 + np.array([control1.x, control1.y])
        P3 = np.array([end.x, end.y])
        P2 = P3 + np.array([control2.x, control2.y])
        t_point = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
        return Point2D(
            float(t_point[0]),
            float(t_point[1])
        )
    # end bezier

    # Update points
    def update_points(
            self
    ):
        """
        Update the points of the curve.
        """
        # Compute extreme points
        x_min, x_max, y_min, y_max = self._compute_extreme_points()

        # X minima
        self._x_minima.set(x_min.x, x_min.y)
        self._x_maxima.set(x_max.x, x_max.y)
        self._y_minima.set(y_min.x, y_min.y)
        self._y_maxima.set(y_max.x, y_max.y)

        # Center
        self._center.x = (self._x_maxima.x + self._x_minima.x) / 2.0
        self._center.y = (self._y_maxima.y + self._y_minima.y) / 2.0

        # Middle point
        self._middle_point.x = self.bezier(0.5).x
        self._middle_point.y = self.bezier(0.5).y

        # Q1 point
        self._q1_point.x = self.bezier(0.25).x
        self._q1_point.y = self.bezier(0.25).y

        # Q2 point
        self._q3_point.x = self.bezier(0.75).x
        self._q3_point.y = self.bezier(0.75).y
    # end update_points

    # Update curve points
    def update_curve_points(
            self
    ):
        """
        Update the curve points of the curve.
        """
        # Get partial curve
        curve_start, curve_control1, curve_control2, curve_end = self.get_partial_curve(
            t1=self.position.value,
            t2=min(self.position.value + self.path_length.value, 1.0)
        )

        # Start point
        self._curve_start.set(curve_start.x, curve_start.y)

        # End point
        self._curve_end.x = curve_end.x
        self._curve_end.y = curve_end.y

        # Control 1
        self._curve_control1.x = curve_control1.x - self._curve_start.x
        self._curve_control1.y = curve_control1.y - self._curve_start.y

        # Control 2
        self._curve_control2.x = curve_control2.x - self._curve_end.x
        self._curve_control2.y = curve_control2.y - self._curve_end.y

        # Middle point
        self._curve_middle_point.x = self.bezier(self.position.value + self.path_length.value * 0.5).x
        self._curve_middle_point.y = self.bezier(self.position.value + self.path_length.value * 0.5).y

        # Q1 point
        self._curve_q1_point.x = self.bezier(self.position.value + self.path_length.value * 0.25).x
        self._curve_q1_point.y = self.bezier(self.position.value + self.path_length.value * 0.25).y

        # Q3 point
        self._curve_q3_point.x = self.bezier(self.position.value + self.path_length.value * 0.75).x
        self._curve_q3_point.y = self.bezier(self.position.value + self.path_length.value * 0.75).y

        # Compute extreme points
        x_min, x_max, y_min, y_max = self._compute_curve_extreme_points()

        # X minima
        self._curve_x_minima.set(x_min.x, x_min.y)
        self._curve_x_maxima.set(x_max.x, x_max.y)
        self._curve_y_minima.set(y_min.x, y_min.y)
        self._curve_y_maxima.set(y_max.x, y_max.y)

        # Update bounding box
        self._bounding_box.upper_left.set(x_min.x, y_min.y)
        self._bounding_box.width.set(x_max.x - x_min.x)
        self._bounding_box.height.set(y_max.y - y_min.y)

        # Center
        self._curve_center.x = self._bounding_box.upper_left.x + self._bounding_box.width / 2.0
        self._curve_center.y = self._bounding_box.upper_left.y + self._bounding_box.height / 2.0
    # end update_curve_points

    # Update length
    def update_length(
            self
    ):
        """
        Update the length of the curve.
        """
        # Calculate the length of the curve
        self._length = self._recursive_bezier_length(
            np.array([self.start.x, self.start.y]),
            np.array([self.abs_control1.x, self.abs_control1.y]),
            np.array([self.abs_control2.x, self.abs_control2.y]),
            np.array([self.end.x, self.end.y]),
            0
        )
    # end update_length

    # Update math
    def update_data(self):
        """
        Update the math of the curve.
        """
        # Update points
        self.update_points()

        # Update curve points
        self.update_curve_points()

        # Update length
        self.update_length()

        # Update bounding box
        # self.update_bbox()
    # end update_data

    # Update bounding box
    def update_bbox(
            self
    ):
        """
        Update the bounding box of the drawable.
        """
        # Get the bounding box
        bbox = self._create_bbox()

        # Update the bounding box
        self._bounding_box.upper_left.x = bbox.upper_left.x
        self._bounding_box.upper_left.y = bbox.upper_left.y
        self._bounding_box.width = bbox.width
        self._bounding_box.height = bbox.height
    # end update_bbox

    # Get partial curve
    def get_partial_curve(
            self,
            t1: float,
            t2: float
    ):
        """
        Update the partial curve from t1 to t2.

        Args:
            t1 (float): Start parameter (0 <= t1 <= 1)
            t2 (float): End parameter (t1 <= t2 <= 1)

        Returns:
            Tuple[Point2D, Point2D, Point2D, Point2D]: The start, control1, control2, and end points of the partial curve
        """
        # Get control points
        p0 = np.array([self.start.x, self.start.y])
        p1 = np.array([self.abs_control1.x, self.abs_control1.y])
        p2x = np.array([self.abs_control2.x, self.abs_control2.y])
        p3 = np.array([self.end.x, self.end.y])

        # Subdivide at t1 to get the new control points from t1 to 1
        _, p01, p012, p0123, _, p123, p23, p3 = self._bezier_subdivide(t1, p0, p1, p2x, p3)

        # Calculate relative t for t2
        relative_t = (t2 - t1) / (1 - t1)

        # Subdivide again at relative t
        p0123_start, p01_start, p012_end, p0123_end, test2, test3, test4, test5 = self._bezier_subdivide(
            relative_t,
            p0123,
            p123,
            p23,
            p3
        )

        return (
            p2(p0123_start[0], p0123_start[1]),
            p2(p01_start[0], p01_start[1]),
            p2(p012_end[0], p012_end[1]),
            p2(p0123_end[0], p0123_end[1])
        )
    # end get_partial_curve

    # endregion PUBLIC

    # region DRAW

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
            x1=self.abs_control1.x,
            y1=self.abs_control1.y,
            x2=self.abs_control2.x,
            y2=self.abs_control2.y,
            x3=self.end.x,
            y3=self.end.y
        )

        # Set the line width
        context.set_line_width(0.1)

        # Stroke the path
        context.stroke()

        # Restore the context
        context.restore()
    # end draw_path

    def draw_sub_control_point(self, context, point, label=None):
        """
        Draw a control point on the context.

        Args:
            context (Context): The drawing context
            point (np.array): The point to draw
            label (str): Optional label for the point
        """
        # Set the style for control points
        context.save()
        context.set_source_rgb(utils.YELLOW)  # Yellow color for control points
        context.set_line_width(0.05)
        context.arc(point[0], point[1], 0.01, 0, 2 * np.pi)  # Small circle at the control point
        context.fill()

        # Draw the label if provided
        if label:
            context.move_to(point[0] + 0.1, point[1])
            context.set_font_size(0.1)
            context.show_text(label)
        # end if
        context.restore()
    # end draw_sub_control_point

    # Draw control points
    def draw_control_points(
            self,
            context: Context,
            color: Color = utils.YELLOW.copy(),
            point_size: float = 0.05,
            line_width: float = 0.01
    ):
        """
        Draw the control points of the curve.

        Args:
            context (cairo.Context): Context to draw the control points to
            color (Color): Color of the control points
            point_size (float): Size of the control points
            line_width (float): Width of the control points
        """
        # Save the context
        context.save()

        # Set the color
        context.set_source_rgb(color)
        context.set_line_width(line_width)

        # Draw start
        context.rectangle(
            self.start.x - point_size / 2,
            self.start.y - point_size / 2,
            point_size,
            point_size
        )
        context.stroke()

        # Draw control 1
        context.rectangle(
            self.abs_control1.x - point_size / 2,
            self.abs_control1.y - point_size / 2,
            point_size,
            point_size
        )
        context.stroke()

        # Draw line between start and control 1
        context.move_to(self.start.x, self.start.y)
        context.line_to(self.abs_control1.x, self.abs_control1.y)
        context.stroke()

        # Draw control 2
        context.rectangle(
            self.abs_control2.x - point_size / 2,
            self.abs_control2.y - point_size / 2,
            point_size,
            point_size
        )
        context.stroke()

        # Draw end
        context.rectangle(
            self.end.x - point_size / 2,
            self.end.y - point_size / 2,
            point_size,
            point_size
        )
        context.stroke()

        # Draw line between control 2 and end
        context.move_to(self.abs_control2.x, self.abs_control2.y)
        context.line_to(self.end.x, self.end.y)
        context.stroke()

        # Restore the context
        context.restore()
    # end draw_control_points

    # Draw points
    def draw_points(
            self,
            context: Context,
            font_size: float = 0.075,
            line_width: float = 0.02,
            radius: float = 0.05
    ):
        """
        Draw the points of the curve.

        Args:
            context (cairo.Context): Context to draw the points to
            font_size (float): Size of the font
            line_width (float): Width of the points
            radius (float): Radius of the points
        """
        # Save the context
        context.save()

        # Line width
        context.set_line_width(line_width)

        # Points to display
        points_to_display = [
            # "Start",
            # "End",
            "Center",
            # "Middle",
            # "Q1",
            # "Q3",
            # "X minima",
            # "X maxima",
            # "Y minima",
            # "Y maxima",
            "Curve start",
            "Curve end",
            # "Curve control 1",
            "Curve abs. control 1",
            # "Curve control 2",
            "Curve abs. control 2",
            "Curve center",
            # "Curve middle point",
            # "Curve Q1",
            # "Curve Q3",
            "Curve X minima",
            "Curve X maxima",
            "Curve Y minima",
            "Curve Y maxima"
        ]

        # Points to show
        points = [
            self._start,
            self._end,
            self._center,
            self._middle_point,
            self._q1_point,
            self._q3_point,
            self._x_minima,
            self._x_maxima,
            self._y_minima,
            self._y_maxima,
            self._curve_start,
            self._curve_end,
            self._curve_control1,
            self.curve_abs_control1,
            self._curve_control2,
            self.curve_abs_control2,
            self._curve_center,
            self._curve_middle_point,
            self._curve_q1_point,
            self._curve_q3_point,
            self._curve_x_minima,
            self._curve_x_maxima,
            self._curve_y_minima,
            self._curve_y_maxima
        ]

        # Point names
        point_names = [
            "Start",
            "End",
            "Center",
            "Middle",
            "Q1",
            "Q3",
            "X minima",
            "X maxima",
            "Y minima",
            "Y maxima",
            "Curve start",
            "Curve end",
            "Curve control 1",
            "Curve abs. control 1",
            "Curve control 2",
            "Curve abs. control 2",
            "Curve center",
            "Curve middle point",
            "Curve Q1",
            "Curve Q3",
            "Curve X minima",
            "Curve X maxima",
            "Curve Y minima",
            "Curve Y maxima"
        ]

        # Draw the points
        for point, point_name in zip(points, point_names):
            if point_name in points_to_display:
                # Draw point
                context.set_source_rgb(utils.RED)
                context.arc(
                    point.x,
                    point.y,
                    radius,
                    0,
                    2 * math.pi
                )
                context.stroke()

                # Draw text upper left
                context.set_font_size(font_size)

                # point_position = f"({point.x:0.01f}, {point.y:0.01f})"
                context.set_source_rgb(utils.WHITE)
                extents = context.text_extents(point_name)
                context.move_to(point.x - extents.width / 2, point.y - extents.height * 2)
                context.show_text(point_name)
                context.fill()
            # end if
        # end if

        # Restore the context
        context.restore()
    # end draw_points

    # Draw the element
    def draw(
            self,
            context: Context,
            draw_bboxes: bool = False,
            draw_reference_point: bool = False,
            draw_control_points: bool = False,
            draw_points: bool = False,
            *args,
            **kwargs
    ):
        """
        Draw the curve to the context.

        Args:
            context (cairo.Context): Context to draw the curve to
            draw_bboxes (bool): Flag to draw the bounding box
            draw_reference_point (bool): Flag to draw the reference point
            draw_points (bool): Flag to draw the points
            draw_control_points (bool): Flag to draw the control points
        """
        # Save context
        context.save()

        # Move to the start point
        context.move_to(self.start.x, self.start.y)

        # Line width and color
        context.set_source_rgba(self._line_color)
        context.set_line_width(self._line_width.value)

        # Draw the curve
        if self.position == 0.0 and self.path_length == 1.0:
            context.curve_to(
                x1=self.abs_control1.x,
                y1=self.abs_control1.y,
                x2=self.abs_control2.x,
                y2=self.abs_control2.y,
                x3=self.end.x,
                y3=self.end.y
            )
        else:
            context.move_to(self.curve_start.x, self.curve_start.y)
            context.curve_to(
                x1=self.curve_abs_control1.x,
                y1=self.curve_abs_control1.y,
                x2=self.curve_abs_control2.x,
                y2=self.curve_abs_control2.y,
                x3=self.curve_end.x,
                y3=self.curve_end.y
            )
        # end if
        context.stroke()

        # Draw bounding box
        if draw_bboxes:
            self.draw_bbox(
                context=context,
                border_color=utils.BLUE.copy(),
                border_width=self.line_width.value / 2.0
            )
        # end if

        # Draw reference point
        if draw_reference_point:
            self.draw_bbox_anchors(context)
        # end if

        # Draw points
        if draw_points:
            self.draw_points(context)
        # end if

        # Draw the control points
        if draw_control_points:
            self.draw_control_points(
                context,
                color=utils.YELLOW.copy(),
                line_width=self.line_width.value/2.0
            )
        # end if

        # Restore
        context.restore()
    # end draw

    # endregion DRAW

    # region PRIVATE

    # Compute extreme points
    def _compute_curve_extreme_points(
            self
    ):
        """
        Compute the extreme points of the curve.
        """
        # X Coefficients
        ax = -3 * self.curve_start.x + 9 * self.curve_abs_control1.x - 9 * self.curve_abs_control2.x + 3 * self.curve_end.x
        bx = 6 * self.curve_start.x - 12 * self.curve_abs_control1.x + 6 * self.curve_abs_control2.x
        cx = -3 * self.curve_start.x + 3 * self.curve_abs_control1.x

        # Y Coefficients
        ay = -3 * self.curve_start.y + 9 * self.curve_abs_control1.y - 9 * self.curve_abs_control2.y + 3 * self.curve_end.y
        by = 6 * self.curve_start.y - 12 * self.curve_abs_control1.y + 6 * self.curve_abs_control2.y
        cy = -3 * self.curve_start.y + 3 * self.curve_abs_control1.y

        # Solve for x and y
        tx = self._solve_quadratic(ax, bx, cx)
        ty = self._solve_quadratic(ay, by, cy)

        # Filter the values
        tx = [t for t in tx if 0 <= t <= 1]
        ty = [t for t in ty if 0 <= t <= 1]

        # Maximum and minimum values
        def bezier(t, P0, P1, P2, P3):
            return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
        # end bezier

        # Points
        P0 = np.array([self.curve_start.x, self.curve_start.y])
        P1 = np.array([self.curve_abs_control1.x, self.curve_abs_control1.y])
        P2 = np.array([self.curve_abs_control2.x, self.curve_abs_control2.y])
        P3 = np.array([self.curve_end.x, self.curve_end.y])

        # Compute the points
        points = [P0, P3] + [bezier(t, P0, P1, P2, P3) for t in tx] + [bezier(t, P0, P1, P2, P3) for t in ty]

        # Get points for xmin, xmax, ymin, ymax
        xmin = min(points, key=lambda p: p[0])
        xmax = max(points, key=lambda p: p[0])
        ymin = min(points, key=lambda p: p[1])
        ymax = max(points, key=lambda p: p[1])

        return (
            p2(float(xmin[0]), float(xmin[1])),
            p2(float(xmax[0]), float(xmax[1])),
            p2(float(ymin[0]), float(ymin[1])),
            p2(float(ymax[0]), float(ymax[1]))
        )
    # end _compute_curve_extreme_points

    def _solve_quadratic(self, a, b, c):
        """
        Solve the quadratic equation ax^2 + bx + c = 0.

        Args:
            a (float): Coefficient of x^2
            b (float): Coefficient of x
            c (float): Constant term

        Returns:
            list: Real solutions of the equation
        """
        if abs(a) < 1e-5:  # a is approximately 0, so the equation is linear, not quadratic
            if abs(b) < 1e-5:  # a and b are both approximately 0
                return []  # No solution or infinite solutions (here we return no solution)
            else:
                return [-c / b]  # Linear solution for bx + c = 0
            # end if
        else:
            discriminant = b ** 2 - 4 * a * c
            if discriminant < 0:
                return []  # No real solutions
            elif discriminant == 0:
                return [-b / (2 * a)]  # One real solution
            else:
                sqrt_disc = np.sqrt(discriminant)
                return [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]  # Two real solutions
            # end if
        # end if
    # end _solve_quadratic

    # Compute extreme points
    def _compute_extreme_points(
            self
    ):
        """
        Compute the extreme points of the curve.
        """
        # Coefficients
        ax = -3 * self.start.x + 9 * self.abs_control1.x - 9 * self.abs_control2.x + 3 * self.end.x
        bx = 6 * self.start.x - 12 * self.abs_control1.x + 6 * self.abs_control2.x
        cx = -3 * self.start.x + 3 * self.abs_control1.x
        ay = -3 * self.start.y + 9 * self.abs_control1.y - 9 * self.abs_control2.y + 3 * self.end.y
        by = 6 * self.start.y - 12 * self.abs_control1.y + 6 * self.abs_control2.y
        cy = -3 * self.start.y + 3 * self.abs_control1.y

        # Solve for x and y
        tx = self._solve_quadratic(ax, bx, cx)
        ty = self._solve_quadratic(ay, by, cy)

        # Filter the values
        tx = [t for t in tx if 0 <= t <= 1]
        ty = [t for t in ty if 0 <= t <= 1]

        # Maximum and minimum values
        def bezier(t, P0, P1, P2, P3):
            return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
        # end bezier

        # Points
        P0 = np.array([self.start.x, self.start.y])
        P1 = np.array([self.abs_control1.x, self.abs_control1.y])
        P2 = np.array([self.abs_control2.x, self.abs_control2.y])
        P3 = np.array([self.end.x, self.end.y])

        # Compute the points
        points = [P0, P3] + [bezier(t, P0, P1, P2, P3) for t in tx] + [bezier(t, P0, P1, P2, P3) for t in ty]

        # Get points for xmin, xmax, ymin, ymax
        xmin = min(points, key=lambda p: p[0])
        xmax = max(points, key=lambda p: p[0])
        ymin = min(points, key=lambda p: p[1])
        ymax = max(points, key=lambda p: p[1])

        return (
            p2(float(xmin[0]), float(xmin[1])),
            p2(float(xmax[0]), float(xmax[1])),
            p2(float(ymin[0]), float(ymin[1])),
            p2(float(ymax[0]), float(ymax[1]))
        )
    # end _compute_extreme_points

    def _create_bbox(
            self
    ):
        """
        Create the bounding box.
        """
        # Get extreme points
        xmin, xmax, ymin, ymax = self._compute_curve_extreme_points()

        return BoundingBox.from_objects(
            upper_left=Point2D(xmin.x, ymin.y),
            width=Scalar(xmax.x - xmin.x),
            height=Scalar(ymax.y - ymin.y)
        )
    # end _create_bbox

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
            t,
            p0,
            p1,
            p2,
            p3
    ):
        """
        Subdivide a cubic Bezier curve at parameter t.

        Args:
            t (float): Parameter for subdivision
            p0, p1, p2, p3 (Point2D): Optional points for subdivision. If not provided, use the curve's own points.

        Returns:
            Tuple[Point2D]: Seven points of the subdivided curve
        """
        # Points
        p01 = (1 - t) * p0 + t * p1
        p12 = (1 - t) * p1 + t * p2
        p23 = (1 - t) * p2 + t * p3
        p012 = (1 - t) * p01 + t * p12
        p123 = (1 - t) * p12 + t * p23
        p0123 = (1 - t) * p012 + t * p123

        return (
            p0,
            p01,
            p012,
            p0123,
            p0123,
            p123,
            p23,
            p3
        )
    # end _bezier_subdivide

    def _bezier_subdivide_at_t(
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
        p1 = np.array([self.abs_control1.x, self.abs_control1.y])
        p3 = np.array([self.end.x, self.end.y])
        p2 = np.array([self.abs_control2.x, self.abs_control2.y])

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
    # end _bezier_subdivide_at_t

    # endregion PRIVATE

    # region MOVABLE

    def init_move(
            self,
            *args,
            **kwargs
    ):
        """
        Initialize the move.
        """
        self.start_position = None
        self.start_control1 = None
        self.start_control2 = None
        self.start_end = None
    # end init_move

    def start_move(
            self,
            start_value: Any,
            *args,
            **kwargs
    ):
        """
        Start the move.
        """
        self.start_position = self.start.copy()
        self.start_control1 = self.abs_control1.copy()
        self.start_control2 = self.abs_control2.copy()
        self.start_end = self.end.copy()
    # end start_move

    def animate_move(
            self,
            t,
            duration,
            interpolated_t,
            end_value,
            *args,
            **kwargs
    ):
        """
        Animate the move.
        """
        # Compute target value for control1, control2 and end using start
        end_value_end = (self.end - self.start) + end_value

        self.start.x = self.start_position.x * (1 - interpolated_t) + end_value.x * interpolated_t
        self.start.y = self.start_position.y * (1 - interpolated_t) + end_value.y * interpolated_t
        self.end.x = self.start_end.x * (1 - interpolated_t) + end_value_end.x * interpolated_t
        self.end.y = self.start_end.y * (1 - interpolated_t) + end_value_end.y * interpolated_t
    # end animate_move

    def end_move(
            self,
            end_value: Any,
            *args,
            **kwargs
    ):
        """
        End the move.
        """
        pass
    # end end_move

    # Finish move
    def finish_move(
            self,
            *args,
            **kwargs
    ):
        """
        Finish the move.

        Args:
            *args: Variable arguments
            **kwargs: Arbitrary keyword arguments
        """
        pass
    # end finish_move

    # endregion MOVABLE

    # region EVENTS

    def _on_start_changed(
            self,
            event
    ):
        """
        Handle the start point changing.

        Args:
            event (Event): Event that triggered the change
        """
        self.update_data()
        # self.dispatch_event("on_change", ObjectChangedEvent(self, property="start", value=self.start))
    # end _on_start_changed

    def _on_control1_changed(
            self,
            event
    ):
        """
        Handle the control1 point changing.

        Args:
            event (Event): Event that triggered the change
        """
        self.update_data()
        # self.dispatch_event("on_change", ObjectChangedEvent(self, property="control1", value=self.control1))
    # end _on_control1_changed

    def _on_control2_changed(
            self,
            event
    ):
        """
        Handle the control2 point changing.

        Args:
            event (Event): Event that triggered the change
        """
        self.update_data()
        # self.dispatch_event("on_change", ObjectChangedEvent(self, property="control2", value=self.control2))
    # end _on_control2_changed

    def _on_end_changed(
            self,
            event
    ):
        """
        Handle the end point changing.

        Args:
            event (Event): Event that triggered the change
        """
        self.update_data()
        # self.dispatch_event("on_change", ObjectChangedEvent(self, property="end", value=self.end))
    # end _on_end_changed

    def _on_position_changed(
            self,
            event
    ):
        """
        Handle the position changing.

        Args:
            event (Event): Event that triggered the change
        """
        self.update_data()
        # self.dispatch_event("on_change", ObjectChangedEvent(self, property="position", value=self.position))
    # end _on_position_changed

    def _on_path_length_changed(
            self,
            event
    ):
        """
        Handle the length changing.

        Args:
            event (Event): Event that triggered the change
        """
        self.update_data()
        # self.dispatch_event("on_change", ObjectChangedEvent(self, property="path_length", value=self.path_length))
    # end _on_path_length_changed

    # endregion EVENTS

    # region OVERRIDE

    # Translate object (to override)
    def _translate_object(
            self,
            dp
    ):
        """
        Translate the object.

        Args:
            dp (Point2D): Displacement vector
        """
        # Translate the points
        self.start.x += dp.x
        self.start.y += dp.y
        self.end.x += dp.x
        self.end.y += dp.y
        self.update_data()
    # _translate_object

    # Rotate object (to override)
    def _rotate_object(
            self,
            angle,
            center: Point2D
    ):
        """
        Rotate the object.

        Args:
            angle (float): Angle to rotate
        """
        # Get the angle
        angle = angle.value if isinstance(angle, Scalar) else angle

        # Copy center
        center = center.copy()

        # Rotate start
        new_x = center.x + (self.start.x - center.x) * math.cos(angle) - (self.start.y - center.y) * math.sin(angle)
        new_y = center.y + (self.start.x - center.x) * math.sin(angle) + (self.start.y - center.y) * math.cos(angle)
        self._start.set(new_x, new_y)

        # Rotate end
        new_x = center.x + (self.end.x - center.x) * math.cos(angle) - (self.end.y - center.y) * math.sin(angle)
        new_y = center.y + (self.end.x - center.x) * math.sin(angle) + (self.end.y - center.y) * math.cos(angle)
        self._end.set(new_x, new_y)

        # Control 1
        new_x = self.control1.x * math.cos(angle) - self.control1.y * math.sin(angle)
        new_y = self.control1.x * math.sin(angle) + self.control1.y * math.cos(angle)
        self._control1.set(new_x, new_y)

        # Control 2
        new_x = self.control2.x * math.cos(angle) - self.control2.y * math.sin(angle)
        new_y = self.control2.x * math.sin(angle) + self.control2.y * math.cos(angle)
        self._control2.set(new_x, new_y)
    # end _rotate_object

    # Scale object (to override)
    def _scale_object(
            self,
            scale,
            center: Point2D
    ):
        """
        Scale the object.

        Args:
            scale (float): Scale factor
            center (Point2D): Center of scaling
        """
        # Get scale
        scale = scale.value if isinstance(scale, Scalar) else scale

        # Copy center
        center = center.copy()

        # Scale start
        new_x = center.x + scale * (self.start.x - center.x)
        new_y = center.y + scale * (self.start.y - center.y)
        self._start.set(new_x, new_y)

        # Scale end
        new_x = center.x + scale * (self.end.x - center.x)
        new_y = center.y + scale * (self.end.y - center.y)
        self._end.set(new_x, new_y)

        # Scale control 1
        new_x = scale * self.control1.x
        new_y = scale * self.control1.y
        self._control1.set(new_x, new_y)

        # Scale control 2
        new_x = scale * self.control2.x
        new_y = scale * self.control2.y
        self._control2.set(new_x, new_y)
    # end _scale_object

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
            end_y: float,
            position: float = 0.0,
            path_length: float = 1.0,
            line_width: Scalar = Scalar(0.0),
            line_color: Color = utils.WHITE,
            on_change=None
    ):
        """
        Create a cubic Bezier curve from scalar values.
        """
        return cls(
            Point2D(start_x, start_y),
            Point2D(control1_x, control1_y),
            Point2D(control2_x, control2_y),
            Point2D(end_x, end_y),
            position=Scalar(position),
            path_length=Scalar(path_length),
            line_width=line_width,
            line_color=line_color,
            on_change=on_change
        )
    # end from_2d

    @classmethod
    def from_objects(
            cls,
            start: Point2D,
            control1: Point2D,
            control2: Point2D,
            end: Point2D,
            position: Scalar = Scalar(0.0),
            path_length: Scalar = Scalar(1.0),
            line_width: Scalar = Scalar(0.0),
            line_color: Color = utils.WHITE,
            on_change=None
    ):
        """
        Create a cubic Bezier curve from objects.
        """
        return CubicBezierCurve(
            start,
            control1,
            control2,
            end,
            position=position,
            path_length=path_length,
            line_width=line_width,
            line_color=line_color,
            on_change=on_change
        )
    # end from_objects

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

    # endregion PUBLIC

    # region DRAW

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

    # endregion DRAW

    # region CLASS_METHODS

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

    # endregion CLASS_METHODS

# end QuadraticBezierCurve
