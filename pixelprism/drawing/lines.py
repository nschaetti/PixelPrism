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
import numpy as np
from typing import Union, List
from pixelprism.base import Context
import pixelprism.utils as utils
from pixelprism.animate import MovableMixin, FadeableMixin, animeattr, CallableMixin
from pixelprism.data import Point2D, Color, Scalar, Event, call_after, EventType, Style, Transform, TPoint2D
from . import BoundingBoxMixin, BoundingBox
from .drawablemixin import DrawableMixin


# A line
@animeattr("start")
@animeattr("end")
@animeattr("style")
@animeattr("transform")
class Line(
    DrawableMixin,
    BoundingBoxMixin,
    MovableMixin,
    FadeableMixin,
    CallableMixin
):
    """
    A class to represent a line in 2D space.
    """

    def __init__(
            self,
            end: Point2D,
            start: Point2D,
            style: Style,
            transform: Transform = None,
            relative: bool = False,
            on_change=None
    ):
        """
        Initialize the line with its start and end points.

        Args:
            start (Point2D): Start point of the line
            end (Point2D): End point of the line
            style (Style): Style of the line
            transform (Transform): Transform of the line
            relative (bool): Is the end position relative or absolute?
            on_change (callable): On change event
        """
        # Init drawable
        DrawableMixin.__init__(
            self,
            style=style,
            transform=transform
        )

        # Init other
        MovableMixin.__init__(self)
        FadeableMixin.__init__(self)
        CallableMixin.__init__(self)

        # Start and end points
        self._start = start
        self._end = end
        self._relative = relative

        # Middle point
        self._middle_point = (TPoint2D.tpoint2d(self._start) + TPoint2D.tpoint2d(self._end)) / 2.0

        # Bounding box
        BoundingBoxMixin.__init__(self)

        # Update points
        self.update_points()

        # Events
        self._on_start_change = Event()
        self._on_end_change = Event()

        # Set events
        self._start.on_change.subscribe(self._on_start_changed)
        self._end.on_change.subscribe(self._on_end_changed)

        # List of event listeners (per events)
        if on_change: self._on_change += on_change
    # end __init__

    # region PROPERTIES

    @property
    def start(self):
        """
        Get the start point of the line.

        Returns:
            Point2D: Start point of the line
        """
        return self._start
    # end start

    @property
    def end(self):
        """
        Get the end point of the line.

        Returns:
            Point2D: End point of the line
        """
        return self._end
    # end end

    @property
    def sx(self):
        """
        Get the X-coordinate of the start point.
        """
        return self._start.x
    # end sx

    @sx.setter
    def sx(self, value):
        """
        Set the X-coordinate of the start point.
        """
        self._start.x = value
    # end sx

    @property
    def sy(self):
        """
        Get the Y-coordinate of the start point.
        """
        return self._start.y
    # end sy

    @sy.setter
    def sy(self, value):
        """
        Set the Y-coordinate of the start point.
        """
        self._start.y = value
    # end sy

    @property
    def ex(self):
        """
        Get the X-coordinate of the end point.
        """
        return self._end.x
    # end ex

    @ex.setter
    def ex(self, value):
        """
        Set the X-coordinate of the end point.
        """
        self._end.x = value
    # end ex

    @property
    def middle_point(self):
        """
        Get the middle point of the line.

        Returns:
            Point2D: Middle point of the line
        """
        return self._middle_point
    # end middle_point

    @property
    def length(self):
        """
        Get the length of the line.

        Returns:
            float: Length of the line
        """
        return np.linalg.norm(self._end.pos - self._start.pos)
    # end length

    # Line width
    @property
    def line_width(self):
        """
        Get the line width of the arc.
        """
        return self.style.line_width
    # end line_width

    # Line color
    @property
    def line_color(self):
        """
        Get the line color of the arc.
        """
        return self.style.line_color
    # end line_color

    # Line dash
    @property
    def line_dash(self):
        """
        Get the line dash of the arc.
        """
        return self.style.line_dash
    # end line_dash

    @property
    def line_cap(self):
        """
        Get the line cap of the arc.
        """
        return self.style.line_cap
    # end line_cap

    @property
    def line_join(self):
        """
        Get the line join of the arc.
        """
        return self.style.line_join
    # end line_join

    # endregion PROPERTIES

    # region PUBLIC

    # Update data
    def update_data(
            self
    ):
        """
        Update the data of the line.
        """
        self.update_points()
        self.update_bbox()
    # end update_data

    # Update points
    def update_points(
            self
    ):
        """
        Update the points of the line.
        """
        pass
    # end update_points

    # Update bounding box
    def update_bbox(
            self
    ):
        """
        Update the bounding box of the line.
        """
        # Get bounding box
        bbox = self._create_bbox()

        # Update bounding box
        self._bounding_box.upper_left.x = bbox.upper_left.x
        self._bounding_box.upper_left.y = bbox.upper_left.y
        self._bounding_box.width = bbox.width
        self._bounding_box.height = bbox.height
    # end update_bbox

    # endregion PUBLIC

    # region DRAW

    # Draw points
    def draw_points(
            self,
            context: Context,
            point_radius: float = 0.05
    ):
        """
        Draw the points of the line.

        Args:
            context (cairo.Context): Context to draw the points to
            point_radius (float): Radius of the points
        """
        # Save the context
        context.save()

        # Draw the start point
        context.set_source_rgba(utils.GREEN)
        context.arc(
            self.start,
            point_radius,
            0.0,
            2 * np.pi
        )
        context.stroke()

        # Draw the start point
        context.set_source_rgba(utils.YELLOW)
        context.arc(
            self.end,
            point_radius,
            0.0,
            2 * np.pi
        )
        context.stroke()

        # Draw the middle point
        context.set_source_rgba(utils.RED)
        context.arc(
            self.middle_point,
            point_radius,
            0.0,
            2 * np.pi
        )
        context.stroke()

        # Restore the context
        context.restore()
    # end draw_points

    # Draw the element
    def draw(
            self,
            context: Context,
            move_to: bool = False,
            build_ratio: float = 1.0,
            draw_bboxes: bool = False,
            draw_reference_point: bool = False,
            draw_points: bool = False,
            *args,
            **kwargs
    ):
        """
        Draw the line to the context.

        Args:
            context (Context): Context to draw the line to
            move_to (bool): Move to the start point
            build_ratio (float): Build ratio
            draw_bboxes (bool): Draw bounding boxes
            draw_reference_point (bool): Draw reference point
            draw_points (bool): Draw points
        """
        # Save context
        context.save()

        # Transform
        context.set_transform(self.transform)

        # Realize the line
        context.move_to(self.start)
        context.line_to(self.end)

        # Set line color
        if self.fadablemixin_state.opacity:
            context.set_source_rgb_alpha(
                color=self.style.line_color,
                alpha=self.fadablemixin_state.opacity
            )
        else:
            context.set_source_rgba(self.style.line_color)
        # end if

        # If dash
        if self.line_dash is not None:
            context.set_dash(self.style.line_dash)
        # end if

        # Set line width
        context.set_line_width(self.style.line_width.value)
        context.stroke()

        # Draw points
        if draw_points:
            self.draw_points(context)
        # end if

        # Restore
        context.restore()

        # Draw bounding box
        if draw_bboxes:
            self.bounding_box.draw(context)
        # end if

        # Draw reference point
        if draw_reference_point:
            self.draw_bbox_anchors(context)
        # end if
    # end draw

    # endregion DRAW

    # region EVENTS

    # Start changed
    @call_after('update_data')
    def _on_start_changed(
            self,
            sender,
            event_type,
            x,
            y
    ):
        """
        Start point changed event.

        Args:
            sender (object): Sender of the event
            event_type (EventType): Type of the event
            x (float): X-coordinate of the point
            y (float): Y-coordinate of the point
        """
        self._on_start_change.trigger(self, event_type=event_type, x=x, y=y)
    # end _start_changed

    # End changed
    @call_after('update_data')
    def _on_end_changed(
            self,
            sender,
            event_type,
            x,
            y
    ):
        """
        End point changed event.

        Args:
            sender (object): Sender of the event
            event_type (EventType): Type of the event
            x (float): X-coordinate of the point
            y (float): Y-coordinate of the point
        """
        self._on_end_change.trigger(self, event_type=event_type, x=x, y=y)
    # end _end_changed

    # endregion EVENTS

    # region PRIVATE

    # Create bounding box
    def _create_bbox(self):
        """
        Create the bounding box.
        """
        return BoundingBox.from_objects(
            upper_left=Point2D(
                min(self.start.x, self.end.x),
                min(self.start.y, self.end.y)
            ),
            width=Scalar(abs(self.end.x - self.start.x)),
            height=Scalar(abs(self.end.y - self.start.y))
        )
    # end _create_bbox

    # Move
    def _translate_object(
            self,
            dp: Point2D
    ):
        """
        Move the path by a given displacement.

        Args:
            dp (Point2D): Displacement to move the object by
        """
        # Translate the start and end points
        self._start.x += dp.x
        self._start.y += dp.y
        self._end.x += dp.x
        self._end.y += dp.y
    # end translate

    # Scale object
    def _scale_object(
            self,
            scale: Scalar,
            center: Point2D = None,
    ):
        """
        Scale the object.

        Args:
            scale (Scalar): Scale to apply
            center (Point2D): Center of the scaling
        """
        # Scale
        m = scale.value if isinstance(scale, Scalar) else scale

        # Center
        if center is None:
            center = self.middle_point.copy()
        # end if

        # Scale
        self._start.scale_(m, center)
        self._end.scale_(m, center)
    # end _scale_object

    # Rotate object
    def _rotate_object(
            self,
            angle: Union[float, Scalar],
            center: Point2D = None
    ):
        """
        Rotate the object.

        Args:
            angle (float): Angle to rotate the object by
        """
        if center is None:
            center = self.middle_point.copy()
        # end if

        # Temporary points
        start = self.start.copy()
        end = self.end.copy()

        # Angle value
        angle = angle.value if isinstance(angle, Scalar) else angle

        # Center around "center"
        start.x -= center.x
        start.y -= center.y
        end.x -= center.x
        end.y -= center.y

        # Rotate the start and end points
        self._start.x = start.x * np.cos(angle) - start.y * np.sin(angle) + center.x
        self._start.y = start.x * np.sin(angle) + start.y * np.cos(angle) + center.y
        self._end.x = end.x * np.cos(angle) - end.y * np.sin(angle) + center.x
        self._end.y = end.x * np.sin(angle) + end.y * np.cos(angle) + center.y
    # end _rotate_object

    # endregion PRIVATE

    # region OVERRIDE

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
        return Line.__str__(self)
    # end __repr__

     # endregion OVERRIDE

    # region CLASS_METHODS

    # From objects
    @classmethod
    def from_objects(
            cls,
            start: Point2D,
            end: Point2D,
            line_width: Scalar = Scalar(1.0),
            line_color: Color = utils.WHITE
    ):
        """
        Create a line from two points.

        Args:
            start (Point2D): Start point of the line
            end (Point2D): End point of the line
            line_width (float): Width of the line
            line_color (Color): Color of the line
        """
        return cls(
            start,
            end,
            line_width,
            line_color
        )
    # end from_objects

    # endregion CLASS_METHODS

# end Line

