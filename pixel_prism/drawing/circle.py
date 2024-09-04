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

from pixel_prism.animate import MovableMixin
from pixel_prism.animate import FadeableMixin
from pixel_prism.data import Point2D, Scalar, Color, EventMixin, ObjectChangedEvent
import pixel_prism.utils as utils
from .drawablemixin import DrawableMixin
from .boundingboxmixin import BoundingBoxMixin
from ..base import Context


# Circle class
class Circle(
    DrawableMixin,
    BoundingBoxMixin,
    EventMixin,
    MovableMixin,
    FadeableMixin
):
    """
    A simple circle class that can be drawn to a cairo context.
    """

    def __init__(
            self,
            position: Point2D,
            radius: Scalar,
            fill_color: Color = utils.WHITE,
            line_color: Color = utils.WHITE,
            line_width: Scalar = Scalar(1),
            fill: bool = True,
            on_change=None
    ):
        """
        Initialize the point.

        Args:
            position (Point2D): Position of the point
            radius (Scalar): Radius of the point
            fill_color (Color): Fill color of the point
            line_color (Color): Border color of the point
            line_width (Scalar): Border width of the point
            fill (bool): Fill the circle or not
            on_change (function): On change event
        """
        # Constructors
        DrawableMixin.__init__(self)
        MovableMixin.__init__(self)
        EventMixin.__init__(self)

        # Position and radius
        self._position = position
        self._radius = radius

        # Properties
        self._fill_color = fill_color
        self._fill = fill
        self._line_color = line_color
        self._line_width = line_width

        # Update points
        self.update_points()

        # Set events
        self._position.add_event_listener("on_change", self._on_position_changed)
        self._radius.add_event_listener("on_change", self._on_radius_changed)

        # List of event listeners (per events)
        self.add_event("on_change")
        if on_change: self.add_event_listener("on_change", on_change)
    # end __init__

    # region PROPERTIES

    # Get position
    @property
    def position(self) -> Point2D:
        """
        Get the position of the circle.

        Returns:
            Point2D: Position of the circle
        """
        return self._position
    # end position

    # Set position
    @position.setter
    def position(self, value: Point2D):
        """
        Set the position of the circle.

        Args:
            value (Point2D): Position of the circle
        """
        self._position.x = value.x
        self._position.y = value.y
        self.update_points()
        self._position.dispatch_event("on_change", ObjectChangedEvent(self, attribute="position", value=value))
    # end position

    # Get radius
    @property
    def radius(self) -> Scalar:
        """
        Get the radius of the circle.

        Returns:
            Scalar: Radius of the circle
        """
        return self._radius
    # end radius

    # Set radius
    @radius.setter
    def radius(self, value: Scalar):
        """
        Set the radius of the circle.

        Args:
            value (Scalar): Radius of the circle
        """
        self._radius = value
        self.update_points()
        self._radius.dispatch_event("on_change", ObjectChangedEvent(self, attribute="radius", value=value))
    # end radius

    # X position
    @property
    def x(self) -> float:
        """
        Get the x position of the circle.

        Returns:
            float: X position of the circle
        """
        return self._position.x
    # end x

    # Set x position
    @x.setter
    def x(self, value: float):
        """
        Set the x position of the circle.

        Args:
            value (float): X position of the circle
        """
        self._position.x = value
        self.update_points()
        self._position.dispatch_event("on_change", ObjectChangedEvent(self, attribute="position", value=self._position))
    # end x

    # Y position
    @property
    def y(self) -> float:
        """
        Get the y position of the circle.

        Returns:
            float: Y position of the circle
        """
        return self._position.y
    # end y

    # Set y position
    @y.setter
    def y(self, value: float):
        """
        Set the y position of the circle.

        Args:
            value (float): Y position of the circle
        """
        self._position.y = value
        self.update_points()
        self._position.dispatch_event("on_change", ObjectChangedEvent(self, attribute="position", value=self._position))
    # end y

    # Line width
    @property
    def line_width(self):
        """
        Get the line width of the arc.
        """
        return self._line_width
    # end line_width

    # Line color
    @property
    def line_color(self):
        """
        Get the line color of the arc.
        """
        return self._line_color
    # end line_color

    # Fill color
    @property
    def fill_color(self):
        """
        Get the fill color of the arc.
        """
        return self._fill_color
    # end fill_color

    # Length (perimeter)
    @property
    def length(self) -> float:
        """
        Get the length of the circle.

        Returns:
            float: Length of the circle
        """
        return 2.0 * math.pi * self._radius.value
    # end length

    # Area
    @property
    def area(self) -> float:
        """
        Get the area of the circle.

        Returns:
            float: Area of the circle
        """
        return math.pi * self._radius.value ** 2
    # end area

    # Movable position
    @property
    def movable_position(self) -> Point2D:
        """
        Get the movable position of the circle.

        Returns:
            Point2D: Movable position of the circle
        """
        return self._position
    # end movable_position

    # Movable position
    @movable_position.setter
    def movable_position(self, value: Point2D):
        """
        Set the movable position of the circle.

        Args:
            value (Point2D): Movable position of the circle
        """
        self.position = value
    # end movable_position

    # endregion PROPERTIES

    # region PUBLIC

    # Update points
    def update_points(self):
        """
        Update the points of the circle.
        """
        pass
    # end update_points

    # endregion PUBLIC

    # region DRAW

    def draw(
            self,
            context: Context,
            draw_bboxes: bool = False,
            draw_reference_point: bool = False,
            draw_points: bool = False,
            *args,
            **kwargs
    ):
        """
        Draw the point to the context.

        Args:
            context (Context): Cairo context
            draw_bboxes (bool): Draw bounding boxes
            draw_reference_point (bool): Draw reference point
            draw_points (bool): Draw
        """
        # Save context
        context.save()

        # Create arc path
        context.arc(
            self.position.x,
            self.position.y,
            self.radius.value,
            0.0,
            math.pi * 2.0
        )

        # Fill color is set
        if self.fill_color is not None:
            # Set fill color
            context.set_source_rgb_alpha(
                color=self.fill_color,
                alpha=self.fadablemixin_state.opacity
            )

            # Stroke or not
            if self.line_width.value == 0:
                context.fill()
            else:
                context.fill_preserve()
            # end if
        # end if

        # Stroke
        if self.line_width.value > 0:
            # Set line color
            context.set_source_rgb_alpha(
                color=self.line_color,
                alpha=self.fadablemixin_state.opacity
            )

            # Set line width
            context.set_line_width(self.line_width.value)
            context.stroke()
        # end if

        # Restore context
        context.restore()
    # end draw

    # endregion DRAW

    # region EVENTS

    # On position changed
    def _on_position_changed(self, event):
        """
        On position changed event.

        Args:
            event (ObjectChangedEvent): Event
        """
        # Update points
        self.update_points()

        # Dispatch event
        self.dispatch_event("on_change", ObjectChangedEvent(self, attribute="position", value=self._position))
    # end _on_position_changed

    # On radius changed
    def _on_radius_changed(self, event):
        """
        On radius changed event.

        Args:
            event (ObjectChangedEvent): Event
        """
        self.update_points()
        self.dispatch_event("on_change", ObjectChangedEvent(self, attribute="radius", value=self._radius))
    # end _on_radius_changed

    # endregion EVENTS

# end Circle
