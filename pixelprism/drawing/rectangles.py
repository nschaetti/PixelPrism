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
from typing import Tuple, Any, Callable
from pixelprism.data import Point2D, Scalar, Color, Style, Transform, call_after, EventType, Event, TPoint2D
from pixelprism.animate import MovableMixin, FadeableMixin, CallableMixin, animeattr
import pixelprism.utils as utils
from .bounding_box import BoundingBox
from .drawablemixin import DrawableMixin
from .boundingboxmixin import BoundingBoxMixin


# A 2D rectangle
@animeattr("upper_left")
@animeattr("width")
@animeattr("height")
@animeattr("style")
@animeattr("transform")
class Rectangle(
    DrawableMixin,
    BoundingBoxMixin,
    MovableMixin,
    FadeableMixin,
    CallableMixin
):
    """
    A class to represent a rectangle in 2D space.
    """

    def __init__(
            self,
            upper_left: Point2D,
            width: Scalar,
            height: Scalar,
            style: Style,
            transform: Transform = None,
            on_change: Callable = None
    ):
        """
        Initialize the rectangle with its upper left corner, width, and height.

        Args:
            upper_left (Point2D): Upper left corner of the rectangle
            width (Scalar): Width of the rectangle
            height (Scalar): Height of the rectangle
            style (Style): Style of the rectangle
            transform (Transform): Transform of the rectangle
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

        # Initialize the rectangle
        self._upper_left = upper_left
        self._width = width if isinstance(width, Scalar) else Scalar(width)
        self._height = height if isinstance(height, Scalar) else Scalar(height)

        # Bottom left
        self._bottom_left = TPoint2D(
            lambda up, w, h: Point2D(up.x + w, up.y + h),
            up=self._upper_left,
            w=self._width,
            h=self._height
        )

        # Bounding box
        BoundingBoxMixin.__init__(self)

        # On change
        self._on_change = Event()
        self._on_change += on_change

        # Upper left, width, and height
        self._upper_left.on_change.subscribe(self._on_upper_left_changed)
        self._width.on_change.subscribe(self._on_width_changed)
        self._height.on_change.subscribe(self._on_height_changed)

        # Events
        self._on_upper_left_change = Event()
        self._on_width_change = Event()
        self._on_height_change = Event()
    # end __init__

    # region PROPERTIES

    @property
    def upper_left(self):
        """
        Get the upper left corner.
        """
        return self._upper_left
    # end upper_left

    @upper_left.setter
    def upper_left(self, value):
        """
        Set the upper left corner.
        """
        self._upper_left.x = value.x
        self._upper_left.y = value.y
        self.update_data()
    # end upper_left

    @property
    def width(self):
        """
        Get the width.
        """
        return self._width
    # end width

    @width.setter
    def width(self, value):
        """
        Set the width.
        """
        self._width.value = value
        self.update_bbox()
    # end width

    @property
    def height(self):
        """
        Get the height.
        """
        return self._height
    # end height

    @height.setter
    def height(self, value):
        """
        Set the height.
        """
        self._height.value = value
        self.update_bbox()
    # end height

    @property
    def bottom_left(self):
        """
        Get the bottom left corner.
        """
        return self._bottom_left
    # end bottom_left

    @property
    def x1(self):
        """
        Get the X-coordinate of the upper left corner.
        """
        return self.upper_left.x
    # end x1

    @property
    def y1(self):
        """
        Get the Y-coordinate of the upper left corner.
        """
        return self.upper_left.y
    # end y1

    @property
    def x2(self):
        """
        Get the X-coordinate of the lower right corner.
        """
        return self.upper_left.x + self._width.value
    # end x2

    @property
    def y2(self):
        """
        Get the Y-coordinate of the lower right corner.
        """
        return self.upper_left.y + self._height.value
    # end y2

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
    def update_data(self):
        """
        Update the data of the rectangle.
        """
        self.update_points()
        self.update_bbox()
    # end update_data

    # Update points
    def update_points(self):
        """
        Update the points of the rectangle.
        """
        pass
    # end update_points

    # Update bounding box
    def update_bbox(
            self
    ):
        """
        Update the bounding box of the rectangle.
        """
        bbox = self._create_bbox()
        bbox.upper_left.x = self._upper_left.x
        bbox.upper_left.y = self._upper_left.y
        bbox.width.value = self._width.value
        bbox.height.value = self._height.value
    # end update_bbox

    # Set alpha
    def set_alpha(self, alpha: float):
        """
        Set the alpha value of the rectangle.

        Args:
            alpha (float): Alpha value
        """
        self.style.fill_color.alpha = alpha
        self.style.line_color.alpha = alpha
    # end set_alpha

    def get_upper_left(self):
        """
        Get the upper left corner of the rectangle.
        """
        return self.upper_left
    # end get_upper_left

    def set_upper_left(self, x: float, y: float):
        """
        Set the upper left corner of the rectangle.

        Args:
            x (float): X-coordinate of the upper left corner
            y (float): Y-coordinate of the upper left corner
        """
        self.upper_left.set(x, y)
        self.update_bbox()
    # end set_upper_left

    def set_width(self, width: float):
        """
        Set the width of the rectangle.

        Args:
            width (float): Width of the rectangle
        """
        self._width.set(width)
        self.update_bbox()
    # end set_width

    def set_height(self, height: float):
        """
        Set the height of the rectangle.

        Args:
            height (float): Height of the rectangle
        """
        self._height.set(height)
        self.update_bbox()
    # end set_height

    # Union
    def union(self, other):
        """
        Return the union of the rectangle with another object.

        Args:
            other (Rectangle): Object to union with
        """
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return Rectangle(
            upper_left=Point2D(x1, y1),
            width=x2 - x1,
            height=y2 - y1,
            style=self.style.copy(),
            transform=self.transform.copy(True)
        )
    # end union

    # Move
    def translate(self, dx: float, dy: float):
        """
        Move the rectangle by a delta.

        Args:
            dx (float): Delta X-coordinate
            dy (float): Delta Y-coordinate
        """
        self._upper_left.x = self._upper_left.x + dx
        self._upper_left.y = self._upper_left.y + dy
        self.update_bbox()
    # end translate

    # Copy
    def copy(self, deep: bool = False):
        """
        Return a copy of the rectangle.

        Args:
            deep (bool): Copy transform deeply
        """
        if deep:
            return Rectangle.from_objects(
                self.upper_left.copy(),
                self.width.copy(),
                self.height.copy(),
                self.style.copy(),
                self.transform.copy()
            )
        else:
            return Rectangle.from_objects(
                self.upper_left.copy(),
                self.width.copy(),
                self.height.copy(),
                self.style.copy(),
                self.transform
            )
        # end if
    # end copy

    # endregion PUBLIC

    # region DRAW

    # Draw bounding box anchors
    def draw_bbox_anchors(self, context):
        """
        Draw the bounding box anchors of the rectangle.
        """
        if self.bounding_box is not None:
            self.bounding_box.draw_anchors(context)
        # end if
    # end draw_bbox_anchors

    # Draw bounding box anchors
    def draw_anchors(self, context):
        """
        Draw the bounding box anchors of the rectangle.

        Args:
            context: Context to draw the rectangle to
        """
        # Save context
        context.save()

        # Point size
        point_size = 0.08
        font_size = 0.1

        # Draw upper left position
        upper_left = self._upper_left
        context.rectangle(
            upper_left.x - point_size / 2.0,
            upper_left.y - point_size / 2.0,
            point_size,
            point_size
        )
        context.set_source_rgba(Color(255, 255, 255, 1))
        context.fill()

        # Draw upper left position
        context.rectangle(
            self.x2 - point_size / 2.0,
            self.y2 - point_size / 2.0,
            point_size,
            point_size
        )
        context.set_source_rgba(Color(255, 255, 255, 1))
        context.fill()

        # Draw text upper left
        context.set_font_size(font_size)
        point_position = f"({self.x1:0.01f}, {self.y1:0.01f})"
        extents = context.text_extents(point_position)
        context.move_to(self.x1 - extents.width / 2, self.y1 - extents.height * 2)
        context.show_text(point_position)
        context.fill()

        # Draw text bottom right
        context.set_font_size(font_size)
        point_position = f"({self.x2:0.01f}, {self.y2:0.01f})"
        extents = context.text_extents(point_position)
        context.move_to(self.x2 - extents.width / 2, self.y2 + extents.height * 2)
        context.show_text(point_position)
        context.fill()

        # Restore context
        context.restore()
    # end draw_bounding_box_anchors

    def draw(
            self,
            context,
            *args,
            **kwargs
    ):
        """
        Draw the rectangle to the context.

        Args:
            context: Context to draw the rectangle to
            *args: Arguments
            **kwargs: Keyword arguments
        """
        # print(f"Rectangle.draw()")
        # Save the context
        context.save()

        # Transform
        context.set_transform(self.transform)

        # Set style
        context.set_style(self.style)

        # Set the color and draw the rectangle
        context.rectangle(
            self.upper_left,
            self.width.value,
            self.height.value
        )

        # Fill the path
        context.set_source_rgba(self.style.fill_color)
        context.fill_preserve()

        # Stroke the path
        context.set_source_rgba(self.style.line_color)
        context.stroke()

        # Restore context
        context.restore()
    # end draw

    # endregion DRAW

    # region PRIVATE

    def _create_bbox(
            self,
            border_width: float = 1.0,
            border_color: Color = utils.WHITE
    ):
        """
        Get the bounding box of the rectangle.

        Args:
            border_width (float): Width of the border
            border_color (Color): Color of the border
        """
        # Get the bounding box of the path
        bbox = BoundingBox.from_objects(
            self.upper_left.copy(),
            Scalar(self._width),
            Scalar(self._height)
        )

        # Return the bounding box
        return bbox
    # end _create_bbox

    # endregion PRIVATE

    # region EVENTS

    @call_after("update_data")
    def _on_upper_left_changed(self, sender, event_type, x, y):
        """
        Handle the upper left corner changed event.

        Args:
            sender: Sender of the event
            event_type: Type of the event
            x: X-coordinate of the upper left corner
            y: Y-coordinate of the upper left corner
        """
        self._on_upper_left_changed.trigger(self, event_type=EventType.POSITION_CHANGED, x=x, y=y)
    # end _on_upper_left_changed

    @call_after("update_data")
    def _on_width_changed(self, sender, event_type, value):
        """
        Handle the width changed event.

        Args:
            sender: Sender of the event
            event_type: Type of the event
            value: New width
        """
        self._on_width_changed.trigger(self, event_type=EventType.VALUE_CHANGED, value=value)
    # end _on_width_changed

    @call_after("update_data")
    def _on_height_changed(self, sender, event_type, value):
        """
        Handle the height changed event.

        Args:
            sender: Sender of the event
            event_type: Type of the event
            value: New height
        """
        self._on_height_changed.trigger(self, event_type=EventType.VALUE_CHANGED, value=value)
    # end _on_height_changed

    # endregion EVENTS

    # region FADE_IN

    def start_fadein(self, start_value: Any):
        """
        Start fading in the path segment.

        Args:
            start_value (any): The start value of the path segment
        """
        self.set_alpha(0)
    # end start_fadein

    def end_fadein(self, end_value: Any):
        """
        End fading in the path segment.
        """
        self.set_alpha(1)
    # end end_fadein

    def animate_fadein(self, t, duration, interpolated_t, env_value):
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

    # region OVERRIDE

    def __str__(self):
        """
        Return a string representation of the rectangle.
        """
        return (
            f"Rectangle(\n"
            f"\tupper_left={self.upper_left},\n"
            f"\twidth={self.width},\n"
            f"\theight={self.height}\n"
            f"\tfill_color={self._fill_color},\n"
            f"\tborder_color={self._border_color},\n"
            f"\tborder_width={self.border_width},\n"
            f"\tfill={self.fill}\n"
            f")"
        )
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the rectangle.
        """
        return Rectangle.__str__(self)
    # end __repr__

    # endregion OVERRIDE

    # region STATIC

    @classmethod
    def from_objects(
            cls,
            upper_left: Point2D,
            width: Scalar,
            height: Scalar,
            style: Style,
            transform: Transform = None
    ):
        """
        Create a rectangle from its objects.

        Args:
            upper_left (Point2D): Upper left corner of the rectangle
            width (Scalar): Width of the rectangle
            height (Scalar): Height of the rectangle
            style (Style): Style of the rectangle
            transform (Transform): Transform of the rectangle
        """
        return cls(
            upper_left,
            width,
            height,
            style,
            transform
        )
    # end from_objects

    @classmethod
    def from_bbox(
            cls,
            bbox: Tuple[float, float, float, float],
            translate: Tuple[float, float] = (0, 0)
    ):
        """
        Create a rectangle from a bounding box.

        Args:
            bbox (Tuple[float, float, float, float]): Bounding box of the rectangle
            translate (Tuple[float, float]): Translation of the rectangle
        """
        # Translate the bounding box
        bbox = (
            bbox[0] + translate[0],
            bbox[1] + translate[0],
            bbox[2] + translate[1],
            bbox[3] + translate[1]
        )

        # Create the rectangle
        upper_left = Point2D(x=bbox[0], y=bbox[2])
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        return cls.from_objects(
            upper_left=upper_left,
            width=Scalar(width),
            height=Scalar(height)
        )
    # end from_bbox

    # endregion STATIC

# end Rectangle

