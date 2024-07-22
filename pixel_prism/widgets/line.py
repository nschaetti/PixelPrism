#
# Description: This file contains the Line class, which is a subclass of the Drawable class.
#


# Imports
import cairo
from pixel_prism.widgets import Widget


class Line(Widget):

    def __init__(
            self,
            start_point,
            end_point,
            color=(1, 1, 1),
            thickness=2
    ):
        """
        Initialize the line.

        Args:
            start_point (tuple): Start point of the line
            end_point (tuple): End point of the line
            color (tuple): Color of the line
            thickness (int): Thickness of the line
        """
        super().__init__()
        self.start_point = start_point
        self.end_point = end_point
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
        context.move_to(*self.start_point)
        context.line_to(*self.end_point)
        context.stroke()
    # end draw

# end Line

