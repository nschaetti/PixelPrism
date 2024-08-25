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
import pixel_prism.utils as utils
from .drawablemixin import DrawableMixin


# A bounding box
class BoundingBox(DrawableMixin):
    """
    A bounding box.
    """

    def __init__(
            self,
            upper_left: Point2D,
            width: Scalar,
            height: Scalar,
    ):
        """
        Initialize the bounding box.

        Args:
            upper_left (Point2D): Upper left corner of the bounding box
            width (Scalar): Width of the bounding box
            height (Scalar): Height of the bounding box
        """
        super().__init__()

        # Upper left corner
        self._upper_left = upper_left
        self._width = width
        self._height = height
    # end __init__

    # region PROPERTIES

    # Properties
    @property
    def upper_left(self) -> Point2D:
        return self._upper_left
    # end upper_left

    @upper_left.setter
    def upper_left(self, value):
        self._upper_left.x = value.x
        self._upper_left.y = value.y
    # end upper_left

    @property
    def width(self) -> Scalar:
        return self._width
    # end width

    @width.setter
    def width(self, value):
        self._width.value = value
    # end width

    @property
    def height(self) -> Scalar:
        return self._height
    # end height

    @height.setter
    def height(self, value):
        self._height.value = value
    # end height

    @property
    def x1(self):
        """
        Get the X-coordinate of the upper left corner.
        """
        return self.upper_left.x
    # end x1

    @x1.setter
    def x1(self, value):
        """
        Set the X-coordinate of the upper left corner.
        """
        self.upper_left.x = value
    # end x1

    @property
    def y1(self):
        """
        Get the Y-coordinate of the upper left corner.
        """
        return self.upper_left.y
    # end y1

    @y1.setter
    def y1(self, value):
        """
        Set the Y-coordinate of the upper left corner.
        """
        self.upper_left.y = value
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

    # endregion PROPERTIES

    # region PUBLIC

    # Draw anchors
    def draw_anchors(
            self,
            context: Any,
            point_size: float,
            font_size: float
    ):
        """
        Draw the anchors of the bounding box.

        Args:
            context (Any): Context to draw to
            point_size (float): Size of the anchor points
            font_size (float): Size of the font
        """
        # Save context
        context.save()

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
    # end draw_anchors

    # Draw the bounding box
    def draw(
            self,
            context: Any,
            border_color: Color = utils.BLUE.copy(),
            border_width: float = 0.01
    ):
        """
        Draw the bounding box to the context.

        Args:
            context (Any): Context to draw to
            border_color (Color): Border color
            border_width (Scalar): Border width
        """
        # Save the context
        context.save()

        # Set the color and draw the rectangle
        context.rectangle(
            self.upper_left.x,
            self.upper_left.y,
            self.width.value,
            self.height.value
        )

        # Draw
        context.set_source_rgba(border_color)
        context.set_line_width(border_width)
        context.stroke()

        # Restore context
        context.restore()
    # end draw

    # endregion PUBLIC

    # region CLASS_METHODS

    # From objects
    @classmethod
    def from_objects(
            cls,
            upper_left: Point2D,
            width: Scalar,
            height: Scalar
    ):
        """
        Create a bounding box from two points.

        Args:
            upper_left (Point2D): Upper left corner
            width (Scalar): Width
            height (Scalar): Height
        """
        return cls(
            upper_left,
            width,
            height
        )
    # end from_objects

    # From scalar
    @classmethod
    def from_scalar(
            cls,
            upper_left: Point2D,
            width: Scalar,
            height: Scalar
    ):
        """
        Create a bounding box from a scalar.

        Args:
            upper_left (Point2D): Upper left corner
            width (Scalar): Width
            height (Scalar): Height
        """
        return cls(
            upper_left,
            width,
            height
        )
    # end from

    # From tuple
    @classmethod
    def from_tuple(
            cls,
            positions: Tuple[float, float, float, float],
            translate: Tuple[float, float] = (0, 0)
    ):
        """
        Create a bounding box from a tuple.

        Args:
            positions (Tuple[float, float]): x1, x2, y1, y2
            translate (Tuple[float, float]): Translation to apply to the bounding
        """
        positions = (
            positions[0] + translate[0],
            positions[1] + translate[0],
            positions[2] + translate[1],
            positions[3] + translate[1]
        )

        return cls(
            Point2D(positions[0], positions[2]),
            Scalar(positions[1] - positions[0]),
            Scalar(positions[3] - positions[2])
        )
    # end from

    # Union
    @classmethod
    def union(cls, other1: 'BoundingBox', other2: 'BoundingBox'):
        """
        Return the union of the rectangle with another object.

        Args:
            other1 (Rectangle): Object to union with
            other2 (Rectangle): Object to union with
        """
        x1 = min(other1.x1, other2.x1)
        y1 = min(other1.y1, other2.y1)
        x2 = max(other1.x2, other2.x2)
        y2 = max(other1.y2, other2.y2)
        return BoundingBox.from_objects(
            upper_left=Point2D(x1, y1),
            width=Scalar(x2 - x1),
            height=Scalar(y2 - y1)
        )
    # end union

    # endregion CLASS_METHODS

    # region OVERRIDE

    # Translate object (to override)
    def _translate_object(
            self,
            dx,
            dy
    ):
        """
        Translate the object.

        Args:
            dx (float): Translation along X-axis
            dy (float): Translation along Y-axis
        """
        self.upper_left.x += dx
        self.upper_left.y += dy
    # end _translate_object

    # To string
    def __str__(self):
        return f"BoundingBox({self.upper_left}, {self.width}, {self.height})"
    # end __str__

    # To representation
    def __repr__(self):
        return f"BoundingBox({self.upper_left}, {self.width}, {self.height})"
    # end __repr__

    # endregion OVERRIDE

# end BoundingBox



