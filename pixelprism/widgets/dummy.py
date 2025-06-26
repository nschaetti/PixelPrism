#
# The dummy widget is a simple widget that fills the surface with a specified color.
# The color is specified as an RGB tuple (R, G, B).
#

# Imports
import cairo
from .widget import Widget


class Dummy(Widget):
    """
    A dummy widget that fills the surface with a specified color.
    """

    def __init__(self, color):
        """
        Initialize the dummy widget with a specified color.

        Args:
            color (tuple): The color to fill the widget with (R, G, B).
        """
        super().__init__()
        self.color = color
    # end __init__

    def draw(self, context: cairo.Context):
        """
        Draw the dummy widget, filling the surface with the specified color.

        Args:
            context (cairo.Context): The context to draw on.
        """
        context.set_source_rgb(*self.color)
        context.paint()
    # end draw

# end DummyWidget
