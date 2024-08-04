

# Imports
from typing import Tuple, Any, Union
from pixel_prism.data import Point2D, Scalar, Color
from pixel_prism.animate.able import MovAble
import pixel_prism.utils as utils

from .drawablemixin import DrawableMixin


# A 2D rectangle
class Rectangle(DrawableMixin, MovAble):
    """
    A class to represent a rectangle in 2D space.
    """

    def __init__(
            self,
            upper_left: Point2D,
            width: Union[Scalar, float],
            height: Union[Scalar, float],
            fill_color: Color = utils.WHITE,
            border_color: Color = utils.WHITE,
            border_width: Union[Scalar, float] = Scalar(1),
            fill: bool = True
    ):
        """
        Initialize the rectangle with its upper left corner, width, and height.

        Args:
            upper_left (Point2D): Upper left corner of the rectangle
            width (Scalar): Width of the rectangle
            height (Scalar): Height of the rectangle
        """
        super().__init__()
        self.upper_left = upper_left
        self.width = width if isinstance(width, Scalar) else Scalar(width)
        self.height = height if isinstance(height, Scalar) else Scalar(height)
        self.fill_color = fill_color
        self.border_color = border_color
        self.border_width = border_width if isinstance(border_width, Scalar) else Scalar(border_width)
        self.fill = fill
    # end __init__

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
        return self.upper_left.x + self.width.value
    # end x2

    @property
    def y2(self):
        """
        Get the Y-coordinate of the lower right corner.
        """
        return self.upper_left.y + self.height.value
    # end y2

    # Bounding box
    @property
    def bbox(self):
        """
        Get the bounding box of the rectangle.
        """
        return self.copy()
    # end bbox

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
    # end set_upper_left

    def set_width(self, width: float):
        """
        Set the width of the rectangle.

        Args:
            width (float): Width of the rectangle
        """
        self.width.set(width)
    # end set_width

    def set_height(self, height: float):
        """
        Set the height of the rectangle.

        Args:
            height (float): Height of the rectangle
        """
        self.height.set(height)
    # end set_height

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

    # Draw bounding box
    def draw_bounding_box(self, context):
        """
        Draw the bounding box of the rectangle.
        """
        # Set the color and draw the rectangle
        self.set_source_rgba(context, utils.GREEN.copy())
        context.rectangle(
            self.upper_left.x,
            self.upper_left.y,
            self.width.value,
            self.height.value
        )
        context.set_line_width(0.12)
        context.stroke()
    # end draw_bbox

    def draw(
            self,
            context
    ):
        """
        Draw the rectangle to the context.

        Args:
            context: Context to draw the rectangle to
        """
        # Save the context
        context.save()

        # Set the color and draw the rectangle
        self.set_source_rgba(context, self.fill_color)

        context.rectangle(
            self.upper_left.x,
            self.upper_left.y,
            self.width.value,
            self.height.value
        )

        # Fill the circle or draw the border
        if self.fill and self.border_width.value == 0:
            context.fill()
        elif self.fill:
            context.fill_preserve()
            self.set_source_rgba(context, self.border_color)
            context.set_line_width(self.border_width.value)
            context.stroke()
        else:
            self.set_source_rgba(context, self.border_color)
            context.set_line_width(self.border_width.value)
            context.stroke()
        # end if

        # Restore context
        context.restore()
    # end draw

    # Copy
    def copy(self):
        """
        Return a copy of the rectangle.
        """
        return Rectangle(
            self.upper_left,
            self.width,
            self.height
        )
    # end copy

    def __str__(self):
        """
        Return a string representation of the rectangle.
        """
        return (
            f"Rectangle(\n"
            f"\tupper_left={self.upper_left},\n"
            f"\twidth={self.width},\n"
            f"\theight={self.height}\n"
            f"\tfill_color={self.fill_color},\n"
            f"\tborder_color={self.border_color},\n"
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

    @classmethod
    def from_bbox(
            cls,
            bbox: Tuple[float, float, float, float]
    ):
        """
        Create a rectangle from a bounding box.

        Args:
            bbox (Tuple[float, float, float, float]): Bounding box of the rectangle
        """
        upper_left = Point2D(x=bbox[0], y=bbox[2])
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        return cls(
            upper_left,
            Scalar(width),
            Scalar(height)
        )
    # end from_bbox

# end Rectangle

