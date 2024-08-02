#
#
#

# Imports
from typing import Any
import numpy as np
from pixel_prism.animate.able import MovAble, FadeInAble, FadeOutAble
from pixel_prism.data import Point2D, Scalar, Color
import pixel_prism.utils as utils
from .drawablemixin import DrawableMixin


class Circle(DrawableMixin, Point2D, FadeInAble, FadeOutAble):
    """
    A simple circle class that can be drawn to a cairo context.
    """

    def __init__(
            self,
            x,
            y,
            fill_color: Color = utils.WHITE,
            radius: Scalar = Scalar(5),
            border_color: Color = utils.WHITE,
            border_width: Scalar = Scalar(1),
            fill: bool = True
    ):
        """
        Initialize the point.

        Args:
            x (float): X-coordinate of the point
            y (float): Y-coordinate of the point
            fill_color (Color): Color of the point
            radius (Scalar): Radius of the point
            border_color (tuple): Color of the border
            border_width (Scalar): Width of the border
            fill (bool): Whether to fill the circle
        """
        # Constructors
        DrawableMixin.__init__(self)
        Point2D.__init__(self, x, y, dtype=np.float32)

        # Properties
        self.fill_color = fill_color
        self.radius = radius
        self.fill = fill
        self.border_color = border_color
        self.border_width = border_width
    # end __init__

    def draw(
            self,
            context
    ):
        """
        Draw the point to the context.

        Args:
            context (cairo.Context): Context to draw the point to
        """
        # Set the color and draw the circle
        context.set_source_rgba(*self.fill_color.value, self.fill_color.opacity)
        context.arc(float(self.pos[0]), float(self.pos[1]), self.radius.value, 0, 2 * 3.14159)

        # Fill the circle or draw the border
        if self.fill and self.border_width.value == 0:
            context.fill()
        elif self.fill:
            context.fill_preserve()
            context.set_source_rgba(*self.border_color.value, self.border_color.opacity)
            context.set_line_width(self.border_width.value)
            context.stroke()
        else:
            context.set_source_rgba(*self.border_color.value, self.border_color.opacity)
            context.set_line_width(self.border_width.value)
            context.stroke()
        # end if
    # end draw

# end Circle
