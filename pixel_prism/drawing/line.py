#
# Description: This file contains the Line class, which is a subclass of the Drawable class.
#


# Imports
from typing import Tuple, Any
import cairo

from pixel_prism.animate.able import FadeInAble, FadeOutAble
from pixel_prism.data import Point2D, Scalar, Line as LineData, Color
import pixel_prism.utils as utils
from .drawable import Drawable


class Line(Drawable, LineData, FadeInAble, FadeOutAble):

    def __init__(
            self,
            start: Point2D,
            end: Point2D,
            color: Color = utils.WHITE,
            thickness: Scalar = Scalar(2)
    ):
        """
        Initialize the line.

        Args:
            start (Point2D): Start point of the line
            end (Point2D): End point of the line
            color (Color): Color of the line
            thickness (Scalar): Thickness of the line
        """
        # Constructors
        Drawable.__init__(self)
        LineData.__init__(self, start, end)

        # Properties
        self.color = color
        self.thickness = thickness
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
        context.set_source_rgba(*self.color.value, self.color.opacity.value)
        context.set_line_width(self.thickness.value)
        context.move_to(float(self.start.x), float(self.start.y))
        context.line_to(float(self.end.x), float(self.end.y))
        context.stroke()
    # end draw

# end Line

