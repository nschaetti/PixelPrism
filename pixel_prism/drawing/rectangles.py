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
from typing import Tuple, Any
from pixel_prism.data import Point2D, Scalar, Color
from pixel_prism.animate.able import BuildableMixin, DestroyableMixin
from pixel_prism.animate import MovableMixin
import pixel_prism.utils as utils
from .bounding_box import BoundingBox

from .drawablemixin import DrawableMixin
from .boundingboxmixin import BoundingBoxMixin


# A 2D rectangle
class Rectangle(
    DrawableMixin,
    BoundingBoxMixin,
    MovableMixin,
    BuildableMixin,
    DestroyableMixin
):
    """
    A class to represent a rectangle in 2D space.
    """

    def __init__(
            self,
            upper_left: Point2D,
            width: Scalar,
            height: Scalar,
            fill_color: Color = utils.WHITE,
            border_color: Color = utils.WHITE,
            border_width: Scalar = Scalar(1),
            fill: bool = True,
            is_built: bool = True,
            build_ratio: float = 1.0
    ):
        """
        Initialize the rectangle with its upper left corner, width, and height.

        Args:
            upper_left (Point2D): Upper left corner of the rectangle
            width (Scalar): Width of the rectangle
            height (Scalar): Height of the rectangle
        """
        DrawableMixin.__init__(self)
        MovableMixin.__init__(self)
        BuildableMixin.__init__(self, is_built, build_ratio)

        # Initialize the rectangle
        self._upper_left = upper_left
        self._width = width if isinstance(width, Scalar) else Scalar(width)
        self._height = height if isinstance(height, Scalar) else Scalar(height)
        self._fill_color = fill_color
        self._border_color = border_color
        self._border_width = border_width if isinstance(border_width, Scalar) else Scalar(border_width)
        self.fill = fill

        # Bounding box
        BoundingBoxMixin.__init__(self)
    # end __init__

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

    @property
    def border_width(self):
        """
        Get the border width.
        """
        return self._border_width
    # end border_width

    @border_width.setter
    def border_width(self, value):
        """
        Set the border width.
        """
        self._border_width.value = value
    # end border_width

    # Set alpha
    def set_alpha(self, alpha: float):
        """
        Set the alpha value of the rectangle.

        Args:
            alpha (float): Alpha value
        """
        self._fill_color.alpha = alpha
        self._border_color.alpha = alpha
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

    # Union
    def union(self, other: 'Rectangle'):
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
            Point2D(x1, y1),
            x2 - x1,
            y2 - y1
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

        # Fill color
        context.set_source_rgba(self._fill_color)

        # Its build
        if self.is_built:
            # Set the color and draw the rectangle
            context.rectangle(
                self.upper_left.x,
                self.upper_left.y,
                self.width.value,
                self.height.value
            )

            # Fill the circle or draw the border
            if self.fill and self.border_width.value == 0:
                context.set_line_width(self.border_width.value)
                context.fill()
            elif self.fill:
                context.fill_preserve()
                context.set_source_rgba(self._border_color)
                context.set_line_width(self.border_width.value)
                context.stroke()
            else:
                context.set_source_rgba(self._border_color)
                context.set_line_width(self.border_width.value)
                context.stroke()
            # end if
        else:
            # Set the color and draw the rectangle
            context.rectangle(
                self.upper_left.x + self.width.value * (1 - self.build_ratio) / 2,
                self.upper_left.y,
                self.width.value * self.build_ratio,
                self.height.value
            )
        # end if

        # Fill the circle or draw the border
        if self.fill and self.border_width.value == 0:
            context.set_line_width(self.border_width.value)
            context.fill()
        elif self.fill:
            context.fill_preserve()
            context.set_source_rgba(self._border_color)
            context.set_line_width(self.border_width.value)
            context.stroke()
        else:
            context.set_source_rgba(self._border_color)
            context.set_line_width(self.border_width.value)
            context.stroke()
        # end if

        # Restore context
        context.restore()
    # end draw

    # Copy
    def copy(self, deep: bool = False):
        """
        Return a copy of the rectangle.

        Args:
            deep (bool): Whether to perform a deep copy
        """
        if deep:
            return Rectangle(
                self.upper_left.copy(),
                self.width.copy(),
                self.height.copy(),
                self._fill_color.copy(),
                self._border_color.copy(),
                self.border_width.copy(),
                self.fill,
                self.is_built,
                self.build_ratio
            )
        else:
            return Rectangle.from_objects(
                self.upper_left,
                Scalar(self._width),
                Scalar(self._height)
            )
        # end if
    # end copy

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
            fill_color: Color = utils.WHITE,
            border_color: Color = utils.WHITE,
            border_width: Scalar = Scalar(1),
            fill: bool = True,
            is_built: bool = True,
            build_ratio: float = 1.0
    ):
        """
        Create a rectangle from its objects.

        Args:
            upper_left (Point2D): Upper left corner of the rectangle
            width (Scalar): Width of the rectangle
            height (Scalar): Height of the rectangle
            fill_color (Color): Fill color of the rectangle
            border_color (Color): Border color of the rectangle
            border_width (Scalar): Border width of the rectangle
            fill (bool): Whether to fill the rectangle
            is_built (bool): Whether the rectangle is built
            build_ratio (float): Build ratio of the rectangle
        """
        return cls(
            upper_left,
            width,
            height,
            fill_color,
            border_color,
            border_width,
            fill,
            is_built,
            build_ratio
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

