#
# Description: This file contains the Line class, which is a subclass of the Drawable class.
#


# Imports
from typing import Tuple
import cairo

from .element import Element


class Line(Element):

    def __init__(
            self,
            start: Tuple[int, int],
            end: Tuple[int, int],
            color=(1, 1, 1),
            thickness=2
    ):
        """
        Initialize the line.

        Args:
            start (tuple): Start point of the line
            end (tuple): End point of the line
            color (tuple): Color of the line
            thickness (int): Thickness of the line
        """
        super().__init__()
        self.start = start
        self.end = end
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
        context.set_source_rgb(*self.color)
        context.set_line_width(self.thickness)
        context.move_to(*self.start)
        context.line_to(*self.end)
        context.stroke()
    # end draw

# end Line

