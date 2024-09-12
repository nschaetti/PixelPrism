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
from pixel_prism.animate import MovableMixin
from pixel_prism.animate import FadeableMixin
from pixel_prism.data import Point2D, Scalar, Event, Transform, call_after, EventType, Style
from . import BoundingBox
from .drawablemixin import DrawableMixin
from .boundingboxmixin import BoundingBoxMixin
from ..base import Context


# Circle class
class Circle(
    DrawableMixin,
    BoundingBoxMixin,
    MovableMixin,
    FadeableMixin
):
    """
    A simple circle class that can be drawn to a cairo context.
    """

    def __init__(
            self,
            center: Point2D,
            radius: Scalar,
            style: Style,
            transform: Transform = None,
            on_change=None,
            on_center_change=None,
            on_radius_change=None
    ):
        """
        Initialize the point.

        Args:
            center (Point2D): Center of the circle
            radius (Scalar): Radius of the circle
            style (DrawableMixin.Style): Style of the circle
            transform (Transform): Transform of the circle
            on_change (function): On change event callback
            on_center_change (function): On center change event callback
            on_radius_change (function): On radius change event callback
        """
        # Constructors
        DrawableMixin.__init__(
            self,
            transform=transform,
            style=style
        )

        # Constructors for animation mixins
        MovableMixin.__init__(self)
        FadeableMixin.__init__(self)

        # Position and radius
        self._center = center
        self._radius = radius

        # Bounding box
        BoundingBoxMixin.__init__(self)

        # Update points
        self.update_points()

        # On change
        self._on_change = Event()
        self._on_change += on_change

        # On center change
        self._on_center_change = Event()
        self._on_center_change += on_center_change

        # On radius change
        self._on_radius_change = Event()
        self._on_radius_change += on_radius_change

        # Set events
        self._center.on_change.subscribe(self._on_center_changed)
        self._radius.on_change.subscribe(self._on_radius_changed)
    # end __init__

    # region PROPERTIES

    # Get center
    @property
    def center(self) -> Point2D:
        """
        Get the center of the circle.

        Returns:
            Point2D: Position of the circle
        """
        return self._center
    # end center

    # Set center
    @center.setter
    def center(self, value: Point2D):
        """
        Set the center of the circle.

        Args:
            value (Point2D): Center of the circle
        """
        self._center.x = value.x
        self._center.y = value.y
        self.update_points()
        self._trigger_on_change()
        self._trigger_on_center_change()
    # end center

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
        self._trigger_on_change()
        self._trigger_on_radius_change()
    # end radius

    # X position
    @property
    def x(self) -> float:
        """
        Get the x position of the circle.

        Returns:
            float: X position of the circle
        """
        return self._center.x
    # end x

    # Set x position
    @x.setter
    def x(self, value: float):
        """
        Set the x position of the circle.

        Args:
            value (float): X position of the circle
        """
        self._center.x = value
        self.update_points()
        self._trigger_on_change()
        self._trigger_on_center_change()
    # end x

    # Y position
    @property
    def y(self) -> float:
        """
        Get the y position of the circle.

        Returns:
            float: Y position of the circle
        """
        return self._center.y
    # end y

    # Set y position
    @y.setter
    def y(self, value: float):
        """
        Set the y position of the circle.

        Args:
            value (float): Y position of the circle
        """
        self._center.y = value
        self.update_points()
        self._trigger_on_change()
        self._trigger_on_center_change()
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

    # region PRIVATE

    def _create_bbox(self):
        """
        Create the bounding box of the circle.
        """
        return BoundingBox(
            upper_left=Point2D(self.position.x - self.radius.value, self.position.y - self.radius.value),
            width=self.radius.value * 2.0,
            height=self.radius.value * 2.0
        )
    # end _create_bbox

    def _trigger_on_change(self):
        """
        Trigger the on change event.
        """
        self._on_change.trigger(self, event_type=EventType.CIRCLE_CHANGED, center=self._center, radius=self._radius)
    # end _trigger_on_change

    def _trigger_on_center_change(self):
        """
        Trigger the on center change event.
        """
        self._on_center_change.trigger(self, event_type=EventType.POSITION_CHANGED, center=self._center)
    # end _trigger_on_center_change

    def _trigger_on_radius_change(self):
        """
        Trigger the on radius change event.
        """
        self._on_radius_change.trigger(self, event_type=EventType.VALUE_CHANGED, radius=self._radius)
    # end _trigger_on_radius_change

    # endregion PRIVATE

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

        # Apply transformation
        context.set_transform(self.transform)

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

    # On center changed
    @call_after('update_data')
    def _on_center_changed(self, sender, event_type, x, y):
        """
        On center changed event.

        Args:
            sender (Any): Sender
            event_type (EventType): Event type
            x (float): X position
            y (float): Y position
        """
        # Dispatch event
        self._trigger_on_change()
        self._trigger_on_center_change()
    # end _on_center_changed

    # On radius changed
    @call_after('update_data')
    def _on_radius_changed(self, sender, event_type, radius):
        """
        On radius changed event.

        Args:
            sender (Any): Sender
            event_type (EventType): Event type
            radius (Scalar): Radius
        """
        # Dispatch event
        self._trigger_on_change()
        self._trigger_on_radius_change()
    # end _on_radius_changed

    # endregion EVENTS

# end Circle
