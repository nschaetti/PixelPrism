#
# Description: A simple point class that can be drawn to a cairo context.
#

import numpy as np
from .element import Element


class Point(Element):

    def __init__(self, x, y, color=(1, 1, 1), radius=5):
        """
        Initialize the point.

        Args:
            x (float): X-coordinate of the point
            y (float): Y-coordinate of the point
            color (tuple): Color of the point
            radius (int): Radius of the point
        """
        super().__init__()
        self.pos = np.array([x, y])
        self.color = color
        self.radius = radius
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
        context.set_source_rgb(*self.color)
        context.arc(float(self.pos[0]), float(self.pos[1]), self.radius, 0, 2 * 3.14159)
        context.fill()
    # end draw

# end Point
