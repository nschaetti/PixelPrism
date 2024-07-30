#
# Description: This file contains the Line class, which is a subclass of the Drawable class.
#


# Imports
from typing import Tuple, Any
import cairo

from pixel_prism.animate.able import FadeInAble, FadeOutAble
from pixel_prism.data import Point2D, Scalar
from .element import Element


class Line(Element, FadeInAble, FadeOutAble):

    def __init__(
            self,
            start: Point2D,
            end: Point2D,
            color=(1, 1, 1),
            thickness: Scalar = Scalar(2),
            opacity: Scalar = Scalar(1.0)
    ):
        """
        Initialize the line.

        Args:
            start (Point2D): Start point of the line
            end (Point2D): End point of the line
            color (tuple): Color of the line
            thickness (Scalar): Thickness of the line
            opacity (Scalar): Opacity of the line
        """
        super().__init__()
        self.start = start
        self.end = end
        self.color = color
        self.thickness = thickness
        self.opacity = opacity
    # end __init__

    def draw(
            self,
            context
    ):
        """
        Draw the line to the context.

        Args:
            context (cairo.Context): Context to draw the line to
        """
        context.set_source_rgba(*self.color, self.opacity.value)
        context.set_line_width(self.thickness.value)
        context.move_to(float(self.start.x), float(self.start.y))
        context.line_to(float(self.end.x), float(self.end.y))
        context.stroke()
    # end draw

# end Line

