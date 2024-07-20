#
# Description: This file contains the Line class, which is a subclass of the Drawable class.
#


# Imports
import cairo
from .base import Drawable


class Line(Drawable):

    def __init__(self, start_point, end_point, color=(1, 1, 1), thickness=2):
        self.start_point = start_point
        self.end_point = end_point
        self.color = color
        self.thickness = thickness

    def draw(self, context):
        context.set_source_rgb(*self.color)
        context.set_line_width(self.thickness)
        context.move_to(*self.start_point)
        context.line_to(*self.end_point)
        context.stroke()
    # end draw

# end Line

