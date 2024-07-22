#
# Description: A simple point class that can be drawn to a cairo context.
#

from .widget import Widget


class Point(Widget):

    def __init__(self, color=(1, 1, 1), radius=5):
        super().__init__()
        self.color = color
        self.radius = radius
    # end __init__

    def draw(self, context):
        context.set_source_rgb(*self.color)
        context.arc(self.x, self.y, self.radius, 0, 2 * 3.14159)
        context.fill()
    # end draw

# end Point
