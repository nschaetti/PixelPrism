#
# Description: A simple point class that can be drawn to a cairo context.
#

import cairo
from .base import Drawable


class Point(Drawable):

    def __init__(self, x, y, color=(1, 1, 1), radius=5):
        self.x = x
        self.y = y
        self.color = color
        self.radius = radius
    # end __init__

    def draw(self, context):
        context.set_source_rgb(*self.color)
        context.arc(self.x, self.y, self.radius, 0, 2 * 3.14159)
        context.fill()
    # end draw

# end Point
