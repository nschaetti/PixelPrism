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

#
# Path element
#

# Imports
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple, Union
import cairo
import numpy as np
from pixelprism.data import Point2D, Color, Scalar, Event
import pixelprism.utils as utils
from pixelprism.drawing import BoundingBoxMixin, BoundingBox
from pixelprism.animate.able import (
    BuildableMixin,
    DestroyableMixin,
    RotableMixin
)
from pixelprism.animate import (
    FadeableMixin,
    MovableMixin,
    animeattr
)
from .rectangles import Rectangle
from .arcs import Arc
from .lines import Line
from .curves import CubicBezierCurve
from .drawablemixin import DrawableMixin
from .transforms import (
    Translate2D,
    Rotate2D,
    Scale2D,
    SkewX2D,
    SkewY2D,
    Matrix2D
)
from ..animate.able.movablemixin import ScalableMixin
from ..base import Context


OnChangeCallback = Optional[Callable[[Event], None]]
PathElement = Union["PathArc", "PathLine", "PathBezierCubic", "PathBezierQuadratic"]


# Path line
class PathLine(Line):
    """
    A simple line class as an element of a path.
    """

    # Constructor
    def __init__(
            self,
            start: Point2D,
            end: Point2D
    ):
        """
        Initialize the line.

        Args:
            start (Point2D): Start point of the line
            end (Point2D): End point of the line
        """
        super().__init__(start, end)
    # end __init__

    # Draw path
    def draw(
            self,
            context,
            *args,
            **kwargs
    ):
        """
        Draw the path of the line.

        Args:
            context (cairo.Context): Context to draw the path to
        """
        # Line to end
        context.line_to(self.end.x, self.end.y)
    # end draw

    # region CLASS_METHODS

    @classmethod
    def from_objects(
            cls,
            start: Point2D,
            end: Point2D,
            *args,
            **kwargs
    ):
        """
        Create a line from objects.

        Args:
            start (Point2D): Start point of the line
            end (Point2D): End point of the line
        """
        return PathLine(
            start=start,
            end=end
        )
    # end from_objects

# end PathLine


# Path bezier cubic curve
class PathBezierCubic(CubicBezierCurve):
    """
    A simple cubic Bezier curve class as an element of a path.
    """

    # Constructor
    def __init__(
            self,
            start: Point2D,
            control1: Point2D,
            control2: Point2D,
            end: Point2D,
            position: Scalar = Scalar(0.0),
            path_length: Scalar = Scalar(1.0),
            on_change: OnChangeCallback = None
    ):
        """
        Initialize the cubic Bezier curve.

        Args:
            start (Point2D): Start point of the curve
            control1 (Point2D): First control point of the curve
            control2 (Point2D): Second control point of the curve
            end (Point2D): End point of the curve
            position (Scalar): Position of the curve
            path_length (Scalar): Length of the curve
            on_change: On change event
        """
        super().__init__(
            start=start,
            control1=control1,
            control2=control2,
            end=end,
            position=position,
            path_length=path_length,
            on_change=on_change
        )
    # end __init__

    # region PUBLIC

    # Draw path
    def draw(
            self,
            context,
            *args,
            **kwargs
    ):
        """
        Draw the path of the curve.

        Args:
            context (cairo.Context): Context to draw the path to
        """
        if self.path_length == 1.0:
            context.curve_to(
                x1=self.abs_control1.x,
                y1=self.abs_control1.y,
                x2=self.abs_control2.x,
                y2=self.abs_control2.y,
                x3=self.end.x,
                y3=self.end.y
            )
        else:
            context.curve_to(
                x1=self.curve_abs_control1.x,
                y1=self.curve_abs_control1.y,
                x2=self.curve_abs_control2.x,
                y2=self.curve_abs_control2.y,
                x3=self.curve_end.x,
                y3=self.curve_end.y
            )
        # end if
    # end draw

    # endregion PUBLIC

    # region EVENTS

    # Start changed
    def _on_point_changed(self, x, y):
        """
        Handle the start point changing.

        Args:
            x (float): X-coordinate of the start point
            y (float): Y-coordinate of the start point
        """
        # Point cannot be changed, exception
        raise ValueError("Start/end point of a bezier curve cannot be changed.")
    # end on_start_changed

    # endregion EVENTS

    # region CLASS_METHODS

    @classmethod
    def from_objects(
            cls,
            start: Point2D,
            control1: Point2D,
            control2: Point2D,
            end: Point2D,
            position: Scalar = Scalar(0.0),
            path_length: Scalar = Scalar(1.0),
            on_change: OnChangeCallback = None,
            *args,
            **kwargs
    ):
        """
        Create a cubic Bezier curve from objects.

        Args:
            start (Point2D): Start point of the curve
            control1 (Point2D): First control point of the curve
            control2 (Point2D): Second control point of the curve
            end (Point2D): End point of the curve
            position (Scalar): Position of the curve
            path_length (Scalar): Length of the curve
            on_change: On change event
        """
        return PathBezierCubic(
            start=start,
            control1=control1,
            control2=control2,
            end=end,
            position=position,
            path_length=path_length,
            on_change=on_change
        )
    # end from_objects

    # endregion CLASS_METHODS

# end PathBezierCubic


# Path bezier quadratic curve
class PathBezierQuadratic(
    MovableMixin,
    BuildableMixin,
    DestroyableMixin
):
    """
    A simple quadratic bezier curve class as an element of a path.
    """

    # Constructor
    def __init__(
            self,
            start: Point2D,
            control: Point2D,
            end: Point2D,
            bounding_box: Rectangle,
            is_built: bool = True,
            build_ratio: float = 1.0
    ):
        """
        Initialize the quadratic bezier curve.

        Args:
            start (Point2D): Start point of the curve
            control (Point2D): Control point of the curve
            end (Point2D): End point of the curve
            bounding_box (Rectangle): Bounding box of the curve
            is_built (bool): Is the curve built
            build_ratio (float): Build ratio of the curve
        """
        # Constructors
        MovableMixin.__init__(self)
        BuildableMixin.__init__(self, is_built, build_ratio)
        DestroyableMixin.__init__(self)

        # Initialize the curve
        self._start = start
        self._control = control
        self._end = end
        self._bounding_box = bounding_box

        # Listen to start, control and end points
        self._start.add_event_listener("on_change", self._on_point_changed)
        self._control.add_event_listener("on_change", self._on_point_changed)
        self._end.add_event_listener("on_change", self._on_point_changed)
    # end __init__

    # Start point
    @property
    def start(self):
        """
        Get the start point of the curve.
        """
        return self._start
    # end start

    # Control point
    @property
    def control(self):
        """
        Get the control point of the curve.
        """
        return self._control
    # end control

    # End point
    @property
    def end(self):
        """
        Get the end point of the curve.
        """
        return self._end
    # end end

    # Bounding box
    @property
    def bounding_box(self):
        """
        Get the bounding box of the curve.
        """
        return self._bounding_box
    # end bounding_box

    # Draw path
    def draw(self, context):
        """
        Draw the path of the curve.

        Args:
            context (cairo.Context): Context to draw the path to
        """
        # Curve to end
        context.curve_to(
            self.control.x, self.control.y,
            self.control.x, self.control.y,
            self.end.x, self.end.y
        )
    # end draw

    # Start changed
    def _on_point_changed(self, x, y):
        """
        Handle the start point changing.

        Args:
            x (float): X
            y (float): Y
        """
        # Point cannot be changed, exception
        raise ValueError("Start/end point of a bezier curve cannot be changed.")
    # end on_start_changed

    # region CLASS_METHODS

    @classmethod
    def from_objects(
            cls,
            start: Point2D,
            control: Point2D,
            end: Point2D,
            bounding_box: Rectangle
    ):
        """
        Create a quadratic bezier curve from objects.

        Args:
            start (Point2D): Start point of the curve
            control (Point2D): Control point of the curve
            end (Point2D): End point of the curve
            bounding_box (Tuple[Point2D, Point2D]): Bounding box of the curve
        """
        return PathBezierQuadratic(
            start=start,
            control=control,
            end=end,
            bounding_box=bounding_box
        )
    # end from_objects

    # endregion CLASS_METHODS

# end PathBezierQuadratic


# Path arc
class PathArc(Arc):
    """
    A simple arc class as an element of a path.
    """

    def __init__(
            self,
            center: Point2D,
            radius: Scalar,
            start_angle: Scalar,
            end_angle: Scalar,
            on_change: OnChangeCallback = None
    ):
        """
        Initialize the arc.

        Args:
            center (Point2D): Center of the arc
            radius (Scalar): Radius of the arc
            start_angle (Scalar): Start angle of the arc
            end_angle (Scalar): End angle of the arc
        """
        super().__init__(
            center=center,
            radius=radius,
            start_angle=start_angle,
            end_angle=end_angle,
            on_change=on_change
        )
    # end __init__

    # Draw path
    def draw(
            self,
            context,
            *args,
            **kwargs
    ):
        """
        Draw the path of the arc.

        Args:
            context (cairo.Context): Context to draw the path to
        """
        # Arc to end
        context.arc(
            self.center.x,
            self.center.y,
            self.radius.value,
            self.start_angle.value,
            self.end_angle.value
        )
    # end draw

    # region CLASS_METHODS

    @classmethod
    def from_objects(
            cls,
            center: Point2D,
            radius: Scalar,
            start_angle: Scalar,
            end_angle: Scalar,
            on_change: OnChangeCallback = None,
            *args,
            **kwargs
    ):
        """
        Create an arc from objects.

        Args:
            center (Point2D): Center of the arc
            radius (Scalar): Radius of the arc
            start_angle (Scalar): Start angle of the arc
            end_angle (Scalar): End angle of the arc
            on_change: On change event
        """
        return PathArc(
            center=center,
            radius=radius,
            start_angle=start_angle,
            end_angle=end_angle,
            on_change=on_change
        )
    # end from_objects

    # endregion CLASS_METHODS

# end PathArc


# Path segment
@animeattr("start")
@animeattr("elements")
class PathSegment(
    DrawableMixin,
    BoundingBoxMixin,
    MovableMixin,
    FadeableMixin,
    RotableMixin,
    BuildableMixin,
    DestroyableMixin
):
    """
    A class to represent a path segment.
    """

    def __init__(
            self,
            start: Point2D,
            elements
    ):
        """
        Initialize the path segment with no elements.
        """
        # Initialize the elements
        self._start = start
        self._elements = elements
        self._length = 0

        # Constructors
        DrawableMixin.__init__(self)
        MovableMixin.__init__(self)
        FadeableMixin.__init__(self)
        RotableMixin.__init__(self)
        BuildableMixin.__init__(self)
        DestroyableMixin.__init__(self)
        BoundingBoxMixin.__init__(self)

        # Update points
        self.update_data()

        # Register to elements
        self.add_event("on_change")
        for element in self.elements:
            element.add_event_listener("on_change", self._on_element_updated)
        # end for
    # end __init__

    # region PROPERTIES

    @property
    def start(self):
        """
        Get the start point of the path segment.
        """
        return self._start
    # end start

    @property
    def elements(self):
        """
        Get the elements of the path segment.
        """
        return self._elements
    # end elements

    @property
    def length(self):
        """
        Get the length of the path segment.
        """
        return self._length
    # end length

    @property
    def movable_position(self) -> Any:
        """
        Get the position of the object.

        Returns:
            any: Position of the object
        """
        return self._start
    # end movable_position

    @movable_position.setter
    def movable_position(self, value: Any):
        """
        Set the position of the object.

        Args:
            value (any): Position of the object
        """
        raise NotImplementedError("Property 'movable_position' must be implemented in the derived class.")
    # end movable_position

    # endregion PROPERTIES

    # region PUBLIC

    # Update data
    def update_data(self):
        """
        Update the data of the path segment.
        """
        # Update length
        self._length = sum([element.length for element in self._elements]) if len(self._elements) > 0 else 0

        # Update bounding box
        self.update_bbox()
    # end update_data

    # Update bounding box
    def update_bbox(self):
        """
        Update the bounding box of the path segment.
        """
        # Create bounding box
        bbox = self._create_bbox()
        self._bounding_box.upper_left.x = bbox.upper_left.x
        self._bounding_box.upper_left.y = bbox.upper_left.y
        self._bounding_box.width = bbox.width
        self._bounding_box.height = bbox.height
    # end update_bbox

    # Add
    def add(self, element: PathElement) -> None:
        """
        Add an element to the path.

        Args:
            element: Element to add to the path
        """
        self.elements.append(element)
        self._length += element.length
        self.update_bbox()
        element.add_event_listener("on_change", self._on_element_updated)
    # end add

    # endregion PUBLIC

    # region DRAW

    # Draw bounding box
    def draw_bounding_box(self, context):
        """
        Draw the bounding box of the path segment.

        Args:
            context (cairo.Context): Context to draw the bounding box to
        """
        # Draw the bounding box of the elements
        for element in self.elements:
            # Draw path bounding box
            element.bounding_box.draw(context)
        # end for
    # end draw_bounding_box

    # Draw path
    def draw_path(self, context):
        """
        Draw the path of the path segment.

        Args:
            context (cairo.Context): Context to draw the path to
        """
        # For each element in the segment
        for element in self.elements:
            element.draw(context)
        # end for
    # end draw_path

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
        # Draw control points
        for element in self.elements:
            if isinstance(element, PathBezierCubic):
                element.draw_control_points(
                    context=context,
                    color=color,
                    point_size=point_size,
                    line_width=line_width
                )
            # end if
        # end for
    # end draw_control_points

    # Draw points
    def draw_points(
            self,
            context: Context,
    ):
        """
        Draw the points of the path segment.

        Args:
            context (Context): Context to draw the points to (Cairo context).
        """
        # Draw points
        for element in self.elements:
            element.draw_points(context)
        # end for
    # end draw_points

    def draw(self, context):
        """
        Draw the path segment.
        """
        # Get first element
        # first_element = self.elements[0]

        # Move to the first element
        context.move_to(self._start.x, self._start.y)

        # if self.is_built:
        # For each element in the segment
        for element in self.elements:
            element.draw(context)
        # end for
    # end draw

    # endregion DRAW

    # region PRIVATE

    # Get bounding box
    def _create_bbox(
            self,
            border_width: float = 1.0,
            border_color: Color = utils.WHITE.copy()
    ):
        """
        Get the bounding box of the path segment.

        Args:
            border_width (float): Width of the border
            border_color (Color): Color of the border
        """
        # Get the bounding box of the path segment
        if len(self.elements) > 0:
            # First bounding box
            bbox = self.elements[0].bounding_box

            # For each element in the path segment
            for element in self.elements[1:]:
                # Update the bounding box
                bbox = BoundingBox.union(bbox, element.bounding_box)
            # end for
        else:
            # Create a dummy bounding box
            bbox = BoundingBox.from_objects(
                upper_left=Point2D(0, 0),
                width=Scalar(0),
                height=Scalar(0)
            )
        # end if

        # Return the bounding box
        return bbox
    # end _create_bbox

    # endregion PRIVATE

    # region EVENTS

    def _on_element_updated(self, event):
        """
        Handle the element updating.

        Args:
            event: Event that triggered the update.
        """
        self.update_data()
        # self.dispatch_event("on_change", event=ObjectChangedEvent(self, property="elements", value=self.elements))
    # end _on_element_updated

    # endregion EVENTS

    # region MOVABLE

    # Initialize position
    def init_move(
            self,
            relative: bool = False
    ):
        """
        Initialize the move animation.
        """
        pass
    # end init_move

    # Start animation
    def start_move(
            self,
            start_value: Any,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Start the move animation.

        Args:
            start_value (any): The start position of the object
            relative (bool): Is the move relative
        """
        if relative:
            self.start_position = Point2D.null()
        else:
            self.start_position = self.movable_position.copy()
        # end if
    # end start_move

    def animate_move(
            self,
            t,
            duration,
            interpolated_t,
            end_value,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Perform the move animation.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            end_value (any): The end position of the object
            relative (bool): Is the move relative
        """
        # New position
        new_position = self.start_position * (1 - interpolated_t) + end_value * interpolated_t

        # Difference
        dp = new_position - self.movable_position

        # Translate object
        self.translate(dp)
    # end animate_move

    # Stop animation
    def end_move(
            self,
            end_value: Any,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Stop the move animation.

        Args:
            end_value (any): The end position of the object
            relative (bool): Is the move relative
        """
        raise NotImplementedError("Method 'end_move' must be implemented in the derived class.")
    # end end_move

    # Finish animation
    def finish_move(
            self,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Finish the move animation.
        """
        raise NotImplementedError("Method 'finish_move' must be implemented in the derived class.")
    # end finish_move

    # endregion MOVABLE

    # region ROTABLE

    # Initialize rotation
    def init_rotate(self, *args, **kwargs):
        """
        Initialize the rotation.
        """
        pass
    # end init_rotate

    # Start rotation
    def start_rotate(self, *args, **kwargs):
        """
        Start the rotation animation.
        """
        self._rotable_angle = 0
    # end start_rotate

    # Start animation
    def animate_rotate(
            self,
            t,
            duration,
            interpolated_t,
            end_value,
            center: Point2D = None,
            *args,
            **kwargs
    ):
        """
        Animate the rotation.
        """
        # Get the angle
        angle = interpolated_t * end_value

        # Difference
        da = angle - self._rotable_angle

        # Translate object
        self.rotate(angle=da, center=center)

        # Set current angle
        self._rotable_angle = angle
    # end animate_rotate

    # End rotate
    def end_rotate(self, *args, **kwargs):
        """
        End the rotation.
        """
        pass
    # end end_rotate

    # Finish rotate
    def finish_rotate(self, *args, **kwargs):
        """
        Finish the rotation.
        """
        pass
    # end finish_rotate

    # endregion ROTABLE

    # region BUILD

    # Start building
    def start_build(self, start_value: Any):
        """
        Start building the vector graphic.
        """
        super().start_build(start_value)
    # end start_build

    # End building
    def end_build(self, end_value: Any):
        """
        End building the vector graphic.
        """
        super().end_build(end_value)
    # end end_build

    # Animate building
    def animate_build(self, t, duration, interpolated_t, env_value):
        """
        Animate building the vector graphic.
        """
        super().animate_build(t, duration, interpolated_t, env_value)
    # end animate_build

    # endregion BUILD

    # region DESTROY

    # Start building
    def start_destroy(self, start_value: Any):
        """
        Start building the vector graphic.
        """
        super().start_destroy(start_value)
    # end start_destroy

    # End building
    def end_destroy(self, end_value: Any):
        """
        End building the vector graphic.
        """
        super().end_destroy(end_value)
    # end end_destroy

    # Animate building
    def animate_destroy(self, t, duration, interpolated_t, env_value):
        """
        Animate building the vector graphic.
        """
        super().animate_destroy(t, duration, interpolated_t, env_value)
    # end animate_destroy

    # endregion DESTROY

    # region OVERRIDE

    # Rotate object (to override)
    def _rotate_object(
            self,
            center: Point2D,
            angle: Union[Scalar, float]
    ):
        """
        Rotate the object.

        Args:
            origin (Point2D): Origin of the rotation
            angle (Union[Scalar, float]): Angle of the rotation
        """
        # Angle
        angle = angle.value if isinstance(angle, Scalar) else angle

        # Copy center
        center = center.copy()

        # Rotate the start point
        self.start.rotate_(center=center, angle=angle)

        # Rotate each segments
        for element in self.elements:
            element.rotate(center=center, angle=angle)
        # end for
    # end _rotate_object

    # Translate object (to override)
    def _translate_object(
            self,
            dp: Point2D
    ):
        """
        Translate the object.
        """
        # Move the start point
        self.start.translate_(dp)

        # Move the path segment
        for element in self.elements:
            element.translate(dp)
        # end for
    # end _translate_object

    # Scale object (to override)
    def _scale_object(
            self,
            center: Point2D,
            scale: Scalar
    ):
        """
        Scale the object.

        Args:
            center (Point2D): Center of the scaling
            scale (Point2D): Scale factor
        """
        # Scale the start point
        self.start.scale_(center=center, scale=scale)

        # Scale the path segment
        for element in self.elements:
            element.scale(center=center, scale=scale)
        # end for
    # end _scale_object

    def __len__(self) -> int:
        """
        Get the number of elements in the path.
        """
        return len(self.elements)
    # end __len__

    def __getitem__(self, index: int) -> PathElement:
        """
        Get the element at the given index in the path.

        Args:
            index (int): Index of the element to get
        """
        return self.elements[index]
    # end __getitem__

    def __setitem__(self, index: int, value: PathElement) -> None:
        """
        Set the element at the given index in the path."

        Args:
            index (int): Index of the element to set
            value: Value to set the element
        """
        self.elements[index] = value
    # end __setitem__

    def __delitem__(self, index: int) -> None:
        """
        Delete the element at the given index in the path.

        Args:
            index (int): Index of the element to delete
        """
        del self.elements[index]
    # end __delitem__

    def __str__(self):
        """
        Get the string representation of the path.
        """
        return f"PathSegment(elements={self.elements})"
    # end __str__

    def __repr__(self):
        """
        Get the string representation of the path.
        """
        return self.__str__()
    # end __repr__

    # endregion OVERRIDE

    # region CLASS_METHODS

    @classmethod
    def from_objects(
            cls,
            start: Point2D,
            elements: List[PathElement]
    ) -> 'PathSegment':
        """
        Create a path segment from data.

        Args:
            start (Point2D): Start point of the path segment
            elements (List[Union[PathArc, PathList, ParthBezierCubic]]): Elements of the path segment

        Returns:
            PathSegment: Path segment created from the list of elements
        """
        return PathSegment(
            start=start,
            elements=elements
        )
    # end from_objects

    @classmethod
    def rectangle(
            cls,
            lower_left: Point2D,
            width: Scalar,
            height: Scalar
    ) -> 'PathSegment':
        """
        Create a path segment for a rectangle.

        Args:
            lower_left (Point2D): Lower left corner of the rectangle (is copied)
            width (Scalar): Width of the rectangle
            height (Scalar): Height of the rectangle

        Returns:
            PathSegment: Path segment for the rectangle
        """
        # Get width and height
        width = width.value if isinstance(width, Scalar) else width
        height = height.value if isinstance(height, Scalar) else height

        # Create the path segment
        return cls.from_objects(
            start=lower_left.copy(),
            elements=[
                PathLine(
                    start=lower_left.copy(),
                    end=Point2D(lower_left.x + width, lower_left.y)
                ),
                PathLine(
                    start=Point2D(lower_left.x + width, lower_left.y),
                    end=Point2D(lower_left.x + width, lower_left.y + height)
                ),
                PathLine(
                    start=Point2D(lower_left.x + width, lower_left.y + height),
                    end=Point2D(lower_left.x, lower_left.y + height)
                ),
                PathLine(
                    start=Point2D(lower_left.x, lower_left.y + height),
                    end=lower_left.copy()
                )
            ]
        )
    # end rectangle

    # endregion CLASS_METHODS

# end PathSegment


@animeattr("line_width")
@animeattr("line_color")
@animeattr("fill_color")
@animeattr("path")
@animeattr("subpaths")
class Path(
    DrawableMixin,
    BoundingBoxMixin,
    MovableMixin,
    RotableMixin,
    ScalableMixin,
    FadeableMixin,
    BuildableMixin,
    DestroyableMixin
):
    """
    A simple path class that can be drawn to a cairo context.
    """

    def __init__(
            self,
            path: PathSegment,
            subpaths: List[PathSegment],
            line_width: Scalar,
            line_color: Color = utils.WHITE,
            fill_color: Color = None,
            closed_path: bool = True,
            transform: Optional[Transform] = None
    ):
        """
        Initialize the path.

        Args:
            path (PathSegment): Path segment of the path
            subpaths (List[PathSegment]): Subpaths of the path
            line_width (Scalar): Width of the line
            line_color (Color): Color of the line
            fill_color (Color): Color to fill the path with
            closed_path (bool): Is the path closed
            transform: Transformation to apply to the path
        """
        # Initialize the elements
        self._line_width = line_width
        self._line_color = line_color
        self._fill_color = fill_color
        self._path = path
        self._subpaths = [] if subpaths is None else subpaths
        self._transform = transform
        self._closed_path = closed_path
        self._length = None

        # Constructors
        DrawableMixin.__init__(self)
        MovableMixin.__init__(self)
        RotableMixin.__init__(self)
        ScalableMixin.__init__(self)
        FadeableMixin.__init__(self)
        BuildableMixin.__init__(self)
        BoundingBoxMixin.__init__(self)

        # Update points
        self.update_data()

        # Events
        self.add_event("on_change")
        self._path.add_event_listener("on_change", self._on_path_changed)
        for subpath in self._subpaths:
            subpath.add_event_listener("on_change", self._on_subpath_element)
        # end for
    # end __init__

    # region PROPERTIES

    @property
    def path(self):
        """
        Get the path segment of the path.
        """
        return self._path
    # end path

    @property
    def subpaths(self):
        """
        Get the subpaths of the path.
        """
        return self._subpaths
    # end subpaths

    @property
    def line_width(self):
        """
        Get the line width of the path.
        """
        return self._line_width
    # end line_width

    @property
    def line_color(self):
        """
        Get the line color of the path.
        """
        return self._line_color
    # end line_color

    @property
    def fill_color(self):
        """
        Get the fill color of the path.
        """
        return self._fill_color
    # end fill_color

    @property
    def closed_path(self):
        """
        Get if the path is closed.
        """
        return self._closed_path
    # end closed_path

    @property
    def transform(self):
        """
        Get the transformation of the path.
        """
        return self._transform
    # end transform

    @property
    def length(self):
        """
        Get the length of the path.
        """
        return self._length
    # end length

    @property
    def movable_position(self) -> Any:
        """
        Get the position of the object.

        Returns:
            any: Position of the object
        """
        return self._path.start
    # end movable_position

    # endregion PROPERTIES

    # region PUBLIC

    # Update data
    def update_data(self):
        """
        Update the data of the path.
        """
        # Update length
        self._length = self.compute_length()

        # Update bounding box
        self.update_bbox()
    # end update_data

    # Update bounding box
    def update_bbox(self):
        """
        Update the bounding box of the path.
        """
        # Create bounding box
        bbox = self._create_bbox()
        self._bounding_box.upper_left.x = bbox.upper_left.x
        self._bounding_box.upper_left.y = bbox.upper_left.y
        self._bounding_box.width = bbox.width
        self._bounding_box.height = bbox.height
    # end update_bbox

    # Compute length
    def compute_length(self):
        """
        Compute the length of the path.
        """
        # Add path length
        length = self._path.length if self._path is not None else 0

        # Add length of subpaths
        for subpath in self._subpaths:
            length += subpath.length
        # end for

        return length
    # end compute_length

    # Set alpha
    def set_alpha(
            self,
            alpha: float
    ):
        """
        Set the alpha value of the path.

        Args:
            alpha (float): Alpha value to set
        """
        self._line_color.alpha = alpha
        self._fill_color.alpha = alpha
    # end set_alpha

    # Add
    def add(self, element: PathElement) -> None:
        """
        Add an element to the path.

        Args:
            element: Element to add to the path
        """
        self._path.add(element)

        # Recompute length
        self._length = self.compute_length()

        # Update bounding box
        self.update_bbox()

        # Add event
        element.add_event_listener("on_change", self._on_path_changed)
    # end add

    # Add subpath
    def add_subpath(self, subpath: PathSegment):
        """
        Add a subpath to the path.

        Args:
            subpath (PathSegmentList): Subpath to add to the path
        """
        self._subpaths.append(subpath)

        # Recompute length
        self._length = self.compute_length()

        # Update bounding box
        self.update_bbox()

        # Add event
        subpath.add_event_listener("on_change", self._on_subpath_element)
    # end add_subpath

    # Get subpaths
    def get_subpaths(self) -> List[PathSegment]:
        """
        Get the subpaths of the path.

        Returns:
            list: Subpaths of the path
        """
        return self._subpaths
    # end get_subpaths

    # Set subpaths
    def set_subpaths(self, subpaths: List[PathSegment]) -> None:
        """
        Set the subpaths of the path.

        Args:
            subpaths (list): Subpaths of the path
        """
        self._subpaths = subpaths
        self._length = self.compute_length()
    # end set_subpaths

    # # Draw bounding box anchors
    # def draw_bbox_anchors(
    #         self,
    #         context
    # ):
    #     """
    #     Draw the bounding box anchors of the path.
    #
    #     Args:
    #         context (cairo.Context): Context to draw the bounding box anchors to
    #     """
    #     # Bounding box
    #     path_bbox = self.bounding_box
    #
    #     # Draw upper left position
    #     upper_left = path_bbox.upper_left
    #     context.rectangle(
    #         upper_left.x - 0.25,
    #         upper_left.y - 0.25,
    #         0.5,
    #         0.5
    #     )
    #     context.set_source_rgba(255, 255, 255, 1)
    #     context.fill()
    #
    #     # Draw upper left position
    #     context.rectangle(
    #         path_bbox.x2 - 0.25,
    #         path_bbox.y2 - 0.25,
    #         0.5,
    #         0.5
    #     )
    #     context.set_source_rgba(255, 255, 255, 1)
    #     context.fill()
    #
    #     # Draw text upper left
    #     context.set_font_size(0.6)
    #     point_position = f"({path_bbox.x1:0.02f}, {path_bbox.y1:0.02f})"
    #     extents = context.text_extents(point_position)
    #     context.move_to(path_bbox.x1 - extents.width / 2, path_bbox.y1 - extents.height)
    #     context.show_text(point_position)
    #     context.fill()
    #
    #     # Draw text bottom right
    #     context.set_font_size(0.6)
    #     point_position = f"({path_bbox.x2:0.02f}, {path_bbox.y2:0.02f})"
    #     extents = context.text_extents(point_position)
    #     context.move_to(path_bbox.x2 - extents.width / 2, path_bbox.y2 + extents.height * 2)
    #     context.show_text(point_position)
    #     context.fill()
    # # end draw_bbox_anchors

    # Draw bounding boxes
    # def draw_bbox(
    #         self,
    #         context
    # ):
    #     """
    #     Draw the bounding box of the path.
    #
    #     Args:
    #         context (cairo.Context): Context to draw the bounding box to
    #     """
    #     # Save context
    #     context.save()
    #
    #     # Draw bb of segments
    #     for subpath in self._subpaths:
    #         # Draw the bounding box of the subpath
    #         subpath.draw_bounding_box(context)
    #     # end for
    #
    #     # Draw segments bb of path
    #     self._path.draw_bounding_box(context)
    #
    #     # Draw subpathsbounding box
    #     for subpath in self._subpaths:
    #         # Get the bounding box
    #         path_bbox = subpath.bounding_box
    #
    #         # Draw the bounding box
    #         path_bbox.draw(context)
    #     # end for
    #
    #     # Draw path bounding box
    #     path_bbox = self.bounding_box
    #     path_bbox.draw(context)
    #
    #     # Restore context
    #     context.restore()
    # # end draw_bbox

    def apply_transform(
            self,
            context: cairo.Context,
            transform: Transform
    ) -> None:
        """
        Apply an SVG transform to a Cairo context.

        Args:
            context (cairo.Context): Context to apply the transform to
            transform (str): SVG transform
        """
        # Translate
        if isinstance(transform, Translate2D):
            context.translate(transform.translate.x, transform.translate.y)
        # Scale
        elif isinstance(transform, Scale2D):
            context.scale(transform.scale.x, transform.scale.y)
        # Rotate
        elif isinstance(transform, Rotate2D):
            context.translate(transform.center.x, transform.center.y)
            context.rotate(np.radians(transform.angle.value))
            context.translate(-transform.center.x, -transform.center.y)
        elif isinstance(transform, SkewX2D):
            angle = np.radians(float(transform.angle.value))
            matrix = cairo.Matrix(1, np.tan(angle), 0, 1, 0, 0)
            context.transform(matrix)
        elif isinstance(transform, SkewY2D):
            angle = np.radians(float(transform.angle.value))
            matrix = cairo.Matrix(1, 0, np.tan(angle), 1, 0, 0)
            context.transform(matrix)
        elif isinstance(transform, Matrix2D):
            matrix = cairo.Matrix(
                float(transform.xx.value),
                float(transform.yx.value),
                float(transform.xy.value),
                float(transform.yy.value),
                float(transform.x0.value),
                float(transform.y0.value)
            )
            context.transform(matrix)
        else:
            raise ValueError(f"Unknown transform: {transform}")
        # end if
    # end apply_transform

    # endregion PUBLIC

    # region DRAW

    # Draw path components
    def draw_paths(
            self,
            context
    ):
        """
        Draw the path components to the context.

        Args:
            context (cairo.Context): Context to draw the path components to
        """
        # Save context
        context.save()

        # Draw path
        self._path.draw_path(context)

        # Draw subpaths
        for subpath in self._subpaths:
            subpath.draw_path(context)
        # end for

        # Restore context
        context.restore()
    # end draw_paths

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
        Draw the path to the context.

        Args:
            context (cairo.Context): Context to draw the path to
            draw_bboxes (bool): Whether to draw the bounding boxes
            draw_reference_point (bool): Whether to draw the debug information
            draw_control_points (bool): Whether to draw the control points
            draw_points (bool): Whether to draw the points
        """
        # Save context
        context.save()

        # Reference point
        """if draw_reference_point:
            self.debug_circle.draw(context)
        # end if"""

        # Apply transform
        # if self._transform is not None:
        #     self.apply_transform(context, self._transform)
        # end if

        # Draw path
        context.new_path()
        self._path.draw(context)

        # For each path segments
        for segment in self._subpaths:
            # New sub path
            context.new_sub_path()

            # Draw the sub path
            segment.draw(context)
        # end for

        # Fill if color is set
        if self._fill_color is not None:
            # Fill rule
            context.set_fill_rule(cairo.FillRule.EVEN_ODD)

            # Set color
            context.set_source_rgba(self._fill_color)

            # If stroke or not
            if self._line_width.value == 0:
                context.fill()
            else:
                context.fill_preserve()
            # end if
        # end if

        # Stroke
        if self._line_width.value > 0:
            # Closed path ?
            if self._closed_path:
                context.close_path()
            # end if

            context.set_line_width(self._line_width.value)
            context.set_source_rgba(self._line_color)
            context.stroke()
        # end if

        # Draw the bounding box and anchors
        if draw_bboxes:
            self.bounding_box.draw(context)
            # self._path.draw_bounding_box(context)
            self._path.bounding_box.draw(context)
            for subpath in self._subpaths:
                # subpath.draw_bounding_box(context)
                subpath.bounding_box.draw(context)
            # end for
        # end if

        # Draw control points
        if draw_control_points:
            self._path.draw_control_points(context)
            for subpath in self._subpaths:
                subpath.draw_control_points(context)
            # end for
        # end if

        # Draw reference points
        if draw_reference_point:
            self.bounding_box.draw_anchors(context)
            for subpath in self._subpaths:
                subpath.bounding_box.draw_anchors(context)
            # end for
        # end if

        # Draw points
        if draw_points:
            self._path.draw_points(context)
            for subpath in self._subpaths:
                subpath.draw_points(context)
            # end for
        # end if

        # Restore the context
        context.restore()
    # end draw

    # endregion DRAW

    # region EVENTS

    # On path changed
    def _on_path_changed(self, event):
        """
        Handle the element changing.

        Args:
            event: Event that triggered the change.
        """
        self.update_data()
        # self.dispatch_event("on_change", event=ObjectChangedEvent(self, property="path", value=event))
    # end _on_path_changed

    # On subpath element
    def _on_subpath_element(self, event):
        """
        Handle the element changing.

        Args:
            event: Event that triggered the change.
        """
        self.update_data()
        # self.dispatch_event("on_change", event=ObjectChangedEvent(self, property="subpath", value=event))
    # end _on_subpath_element

    # endregion EVENTS

    # region PRIVATE

    # Translate object
    def _translate_object(
            self,
            dp
    ):
        """
        Translate the object.
        """
        # Move path
        self._path.translate(dp)

        # Translate the subpaths
        for subpath in self._subpaths:
            # Translate the subpath
            subpath.translate(dp)
        # end for
    # end _translate_object

    # Rotate object
    def _rotate_object(
            self,
            center: Point2D,
            angle: Union[Scalar, float]
    ):
        """
        Rotate the object.

        Args:
            origin (Point2D): Origin of the rotation
            angle (Union[Scalar, float]): Angle of the rotation
        """
        # Angle
        angle = angle.value if isinstance(angle, Scalar) else angle

        # Copy center
        center = center.copy()

        # Rotate the path
        self._path.rotate(center=center, angle=angle)

        # Rotate the subpaths
        for subpath in self._subpaths:
            # Rotate the subpath
            subpath.rotate(center=center, angle=angle)
        # end for
    # end _rotate_object

    # Scale object
    def _scale_object(self, center: Point2D, scale: Union[Scalar, float]):
        """
        Scale the object.

        Args:
            center (Point2D): Center of the scaling
            scale (Union[Scalar, float]): Scale factor
        """
        # Scale path
        self._path.scale(center=center, scale=scale)

        # Scale subpaths
        for subpath in self._subpaths:
            # Scale the subpath
            subpath.scale(center=center, scale=scale)
        # end for
    # end _scale_object

    def _create_bbox(
            self,
            border_width: float = 1.0,
            border_color: Color = utils.WHITE.copy()
    ):
        """
        Get the bounding box of the path.

        Args:
            border_width (float): Width of the border
            border_color (Color): Color of the border
        """
        # Get the bounding box of the path
        bbox = self._path.bounding_box

        # For each subpath
        for subpath in self._subpaths:
            # Update the bounding box
            bbox = BoundingBox.union(bbox, subpath.bounding_box)
        # end for

        # Return the bounding box
        return bbox
    # end _create_bbox

    # endregion PRIVATE

    # region MOVABLE

    # Initialize position
    def init_move(
            self,
            relative: bool = False
    ):
        """
        Initialize the move animation.
        """
        pass

    # end init_move

    # Start animation
    def start_move(
            self,
            start_value: Any,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Start the move animation.

        Args:
            start_value (any): The start position of the object
            relative (bool): Is the move relative
        """
        if relative:
            self.movablemixin_state.start_position = Point2D.null()
            self.movablemixin_state.last_position = Point2D.null()
        else:
            self.movablemixin_state.start_position = self.movable_position.copy()
        # end if
    # end start_move

    def animate_move(
            self,
            t,
            duration,
            interpolated_t,
            end_value,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Perform the move animation.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            end_value (any): The end position of the object
            relative (bool): Is the move relative
        """
        # New position
        new_position = self.movablemixin_state.start_position * (1 - interpolated_t) + end_value * interpolated_t

        # Get difference
        if relative:
            dp = new_position - self.movablemixin_state.last_position
            self.movablemixin_state.last_position = new_position
        else:
            dp = new_position - self.movable_position
        # end if

        # Translate object
        self.translate(dp)
    # end animate_move

    # Stop animation
    def end_move(
            self,
            end_value: Any,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Stop the move animation.

        Args:
            end_value (any): The end position of the object
            relative (bool): Is the move relative
        """
        pass
    # end end_move

    # Finish animation
    def finish_move(
            self,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Finish the move animation.
        """
        pass
    # end finish_move

    # endregion MOVABLE

    # region ROTABLE

    # Initialize rotation
    def init_rotate(self, *args, **kwargs):
        """
        Initialize the rotation.
        """
        pass
    # end init_rotate

    # Start rotation
    def start_rotate(self, *args, **kwargs):
        """
        Start the rotation animation.
        """
        self.rotablemixin_state.angle = 0
    # end start_rotate

    # Start animation
    def animate_rotate(
            self,
            t,
            duration,
            interpolated_t,
            end_value,
            center: Point2D = None,
            *args,
            **kwargs
    ):
        """
        Animate the rotation.
        """
        # Get the angle
        angle = interpolated_t * end_value

        # Difference
        da = angle - self.rotablemixin_state.angle

        # Translate object
        self.rotate(angle=da, center=center)

        # Set current angle
        self.rotablemixin_state.angle = angle
    # end animate_rotate

    # End rotate
    def end_rotate(self, *args, **kwargs):
        """
        End the rotation.
        """
        pass
    # end end_rotate

    # Finish rotate
    def finish_rotate(self, *args, **kwargs):
        """
        Finish the rotation.
        """
        pass
    # end finish_rotate

    # endregion ROTABLE

    # region SCALABLE

    # Initialize animation
    def init_scale(
            self,
            *args,
            **kwargs
    ):
        """
        Initialize the range animation.
        """
        pass
    # end init_scale

    # Start animation
    def start_scale(
            self,
            start_value: Any,
            center: Point2D,
            *args,
            **kwargs
    ):
        """
        Start the range animation.

        Args:
            start_value (any): The start position of the object
            center (Point2D): Center of the scaling
        """
        self.scalablemixin_state.center = center.copy()
        self.scalablemixin_state.scale = 1.0
    # end start_scale

    def animate_scale(
            self,
            t,
            duration,
            interpolated_t,
            end_value,
            center: Point2D,
            *args,
            **kwargs
    ):
        """
        Perform the move animation.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            end_value (any): The end position of the object
            center (Point2D): Center of the scaling
        """
        # Get the angle
        scale = interpolated_t * (end_value - 1) + 1

        # print(f"data = {scale} / {self.scalablemixin_state.scale}")

        # Difference
        da = scale / self.scalablemixin_state.scale

        # Translate object
        self.scale(scale=da, center=self.scalablemixin_state.center)

        # Set current angle
        self.scalablemixin_state.scale *= da
    # end animate_scale

    # Stop animation
    def end_scale(
            self,
            end_value: Any,
            *args,
            **kwargs
    ):
        """
        Stop the range animation.

        Args:
            end_value (any): The end value of the object
        """
        pass
    # end end_scale

    # Finish animation
    def finish_scale(
            self,
            *args,
            **kwargs
    ):
        """
        Finish the range animation.
        """
        pass
    # end finish_scale

    # endregion SCALABLE

    # region FADE_IN

    def start_fadein(
            self,
            start_value: Any,
            *args,
            **kwargs
    ):
        """
        Start fading in the path segment.

        Args:
            start_value (any): The start value of the path segment
        """
        self.set_alpha(0)
    # end start_fadein

    def end_fadein(
            self,
            end_value: Any,
            *args,
            **kwargs
    ):
        """
        End fading in the path segment.
        """
        self.set_alpha(1)
    # end end

    def animate_fadein(
            self,
            t,
            duration,
            interpolated_t,
            env_value,
            *args,
            **kwargs
    ):
        """
        Animate fading in the path segment.
        """
        self.set_alpha(interpolated_t)
    # end animate_fadein

    # endregion FADE_IN

    # region FADE_OUT

    def start_fadeout(self, start_value: Any):
        """
        Start fading out the path segment.
        """
        self.set_alpha(1)
    # end start_fadeout

    def end_fadeout(self, end_value: Any):
        """
        End fading out the path segment.
        """
        self.set_alpha(0)
    # end end_fadeout

    def animate_fadeout(self, t, duration, interpolated_t, target_value):
        """
        Animate fading out the path segment.
        """
        self.set_alpha(1 - interpolated_t)
    # end animate_fadeout

    # endregion FADE_OUT

    # region BUILD

    # Initialize building
    def init_build(self):
        """
        Initialize building the object.
        """
        super().init_build()
        self._path.init_build()
        for subpath in self._subpaths:
            subpath.init_build()
        # end for
    # end init_build

    # Start building
    def start_build(self, start_value: Any):
        """
        Start building the object.
        """
        super().start_build(start_value)
        self._path.start_build(start_value)
        for subpath in self._subpaths:
            subpath.start_build(start_value)
        # end for
    # end start_build

    # End building
    def end_build(self, end_value: Any):
        """
        End building the object.
        """
        super().end_build(end_value)

        # End elements
        self._path.end_build(end_value)
        for subpath in self._subpaths:
            subpath.end_build(end_value)
        # end for
    # end end_build

    def animate_build(self, t, duration, interpolated_t, env_value):
        """
        Animate building the object.
        """
        # Animate building
        super().animate_build(t, duration, interpolated_t, env_value)

        # Animate build for path
        self._path.animate_build(t, duration, interpolated_t, env_value)

        # Animate build for each subpath
        for subpath in self._subpaths:
            subpath.animate_build(t, duration, interpolated_t, env_value)
        # end for
    # end animate_build

    # Finish building
    def finish_build(self):
        """
        Finish building the object.
        """
        super().finish_build()
        self._path.finish_build()
        for subpath in self._subpaths:
            subpath.finish_build()
        # end for
    # end finish_build

    # endregion BUILD

    # region DESTROY

    # Initialize destroying
    def init_destroy(self):
        """
        Initialize destroying the object.
        """
        super().init_destroy()
        self._path.init_destroy()
        for subpath in self._subpaths:
            subpath.init_destroy()
        # end for
    # end init_destroy

    # Start destroying
    def start_destroy(self, start_value: Any):
        """
        Start building the object.
        """
        super().start_destroy(start_value)
        self._path.start_destroy(start_value)
        for subpath in self._subpaths:
            subpath.start_destroy(start_value)
        # end for
    # end start_destroy

    def animate_destroy(self, t, duration, interpolated_t, env_value):
        """
        Animate building the object.
        """
        # Animate building
        super().animate_destroy(t, duration, interpolated_t, env_value)

        # Animate build for path
        self._path.animate_destroy(t, duration, interpolated_t, env_value)

        # Animate build for each subpath
        for subpath in self._subpaths:
            subpath.animate_destroy(t, duration, interpolated_t, env_value)
        # end for
    # end animate_destroy

    # End destroying
    def end_destroy(self, end_value: Any):
        """
        End building the object.
        """
        super().end_destroy(end_value)

        # End elements
        self._path.end_destroy(end_value)
        for subpath in self._subpaths:
            subpath.end_destroy(end_value)
        # end for
    # end end_destroy

    # Finish destroying
    def finish_destroy(self):
        """
        Finish destroying the object.
        """
        super().finish_destroy()
        self._path.finish_destroy()
        for subpath in self._subpaths:
            subpath.finish_destroy()
        # end for
    # end finish_destroy

    # endregion DESTROY

    # region OVERRIDE

    def __len__(self):
        return len(self._path)
    # end __len__

    def __getitem__(self, index):
        return self._path[index]
    # end __getitem__

    def __setitem__(self, index, value):
        self._path[index] = value
    # end __setitem__

    def __delitem__(self, index):
        del self._path[index]
    # end __delitem__

    # str
    def __str__(self):
        """
        Get the string representation of the path.
        """
        return (
            f"Path("
            f"path={self._path},"
            f"subpaths={self._subpaths},"
            f"transform={self._transform.__str__() if self._transform is not None else 'None'}"
            f")"
        )
    # end __str__

    # repr
    def __repr__(self):
        """
        Get the string representation of the path.
        """
        return self.__str__()
    # end __repr__

    # endregion OVERRIDE

    # region METHODS

    # Create path from objects
    @classmethod
    def from_objects(
            cls,
            path: PathSegment,
            subpaths: List[PathSegment],
            line_width: Scalar,
            line_color: Color = utils.WHITE,
            fill_color: Color = None,
            transform: Optional[Transform] = None,
            closed_path: bool = True
    ) -> 'Path':
        """
        Create a path from objects.

        Args:
            path (PathSegment): Path segment of the path
            subpaths (List[PathSegment]): Subpaths of the path
            line_width (Scalar): Width of the line
            line_color (Color): Color of the line
            fill_color (Color): Fill color of the path
            transform: Transformation to apply to the path
            closed_path (bool): Is the path closed
        """
        return Path(
            path=path,
            subpaths=subpaths,
            line_width=line_width,
            line_color=line_color,
            fill_color=fill_color,
            transform=transform,
            closed_path=closed_path
        )
    # end from_objects

    # endregion METHODS

# end Path
