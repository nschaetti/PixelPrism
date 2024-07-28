#
# Description: A simple point class that can be drawn to a cairo context.
#

import numpy as np
from pixel_prism.animate.able import MovAble
from .element import Element


class Point(Element, MovAble):

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

    def animate_move(self, t, duration, interpolated_t, end_value):
        """
        Animate the movement of the point.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            end_value (tuple): The end position of the point (x, y)
        """
        print(f"Time {t}, {duration}, {interpolated_t} {self.pos}, end_value: {end_value}")
        self.pos = self.pos * (1 - interpolated_t) + np.array(end_value) * interpolated_t
    # end animate_move

# end Point
