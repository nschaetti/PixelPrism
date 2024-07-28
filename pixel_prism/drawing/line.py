#
# Description: This file contains the Line class, which is a subclass of the Drawable class.
#


# Imports
from typing import Tuple
import cairo

from pixel_prism.animate.able import FadeInAble
from .element import Element


class Line(Element, FadeInAble):

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
        self.opacity = 0
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
        context.set_source_rgba(*self.color, self.opacity)
        context.set_line_width(self.thickness)
        context.move_to(float(self.start[0]), float(self.start[1]))
        context.line_to(float(self.end[0]), float(self.end[1]))
        context.stroke()
    # end draw

    def animate_fadein(self, t, duration, interpolated_t):
        """
        Animate the fade-in effect.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
        """
        self.opacity = interpolated_t
    # end animate_fadein

# end Line

