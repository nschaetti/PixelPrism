

# Imports
from typing import Tuple, Any, Union
from pixel_prism.data import Point2D, Scalar, Color
from pixel_prism.animate.able import MovableMixin
import pixel_prism.utils as utils

from .drawablemixin import DrawableMixin


# A 2D rectangle
class Rectangle(DrawableMixin, MovableMixin):
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

    def get_bbox(
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
        bbox = self.copy()

        # Set fill, border color and witdth
        bbox.fill = False
        bbox.border_color = border_color
        bbox.border_width.value = border_width

        # Return the bounding box
        return bbox
    # end get_bbox

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

    # Move
    def translate(self, dx: float, dy: float):
        """
        Move the rectangle by a delta.

        Args:
            dx (float): Delta X-coordinate
            dy (float): Delta Y-coordinate
        """
        self.upper_left.x = self.upper_left.x + dx
        self.upper_left.y = self.upper_left.y + dy
    # end translate

    # Draw bounding box anchors
    def draw_bounding_box_anchors(self, context):
        """
        Draw the bounding box anchors of the rectangle.

        Args:
            context: Context to draw the rectangle to
        """
        # Draw upper left position
        upper_left = self.upper_left
        context.rectangle(
            upper_left.x - 0.25,
            upper_left.y - 0.25,
            0.5,
            0.5
        )
        context.set_source_rgba(255, 255, 255, 1)
        context.fill()

        # Draw upper left position
        context.rectangle(
            self.x2 - 0.25,
            self.y2 - 0.25,
            0.5,
            0.5
        )
        context.set_source_rgba(255, 255, 255, 1)
        context.fill()

        # Draw text upper left
        context.set_font_size(0.6)
        point_position = f"({self.x1:0.02f}, {self.y1:0.02f})"
        extents = context.text_extents(point_position)
        context.move_to(self.x1 - extents.width / 2, self.y1 - extents.height)
        context.show_text(point_position)
        context.fill()

        # Draw text bottom right
        context.set_font_size(0.6)
        point_position = f"({self.x2:0.02f}, {self.y2:0.02f})"
        extents = context.text_extents(point_position)
        context.move_to(self.x2 - extents.width / 2, self.y2 + extents.height * 2)
        context.show_text(point_position)
        context.fill()
    # end draw_bounding_box_anchors

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

        # Draw upper left
        context.rectangle(
            self.upper_left.x - 0.25,
            self.upper_left.y - 0.25,
            0.5,
            0.5
        )
        context.set_source_rgba(255, 255, 255, 1)
        context.fill()
        context.close_path()

        # Draw bottom right
        context.rectangle(
            self.x2 - 0.25,
            self.y2 - 0.25,
            0.5,
            0.5
        )
        context.set_source_rgba(255, 255, 255, 1)
        context.fill()
        context.close_path()
    # end draw_bbox

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
        # Save the context
        context.save()

        # Fill color
        self.set_source_rgba(context, self.fill_color)

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
        return cls(
            upper_left,
            Scalar(width),
            Scalar(height)
        )
    # end from_bbox

# end Rectangle

