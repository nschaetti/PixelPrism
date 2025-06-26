

# Imports
import cairo
import numpy as np
from pixelprism.widgets import Widget


# Plot class
class Plot(Widget):

    def __init__(
            self,
            function,
            x_range,
            color=(1, 1, 1),
            thickness=2
    ):
        """
        Initialize the plot.

        Args:
            function (function): Function to plot
            x_range (tuple): Range of x-values to plot
            color (tuple): Color of the plot
            thickness (int): Thickness of the plot
        """
        super().__init__()
        self.function = function
        self.x_range = x_range
        self.color = color
        self.thickness = thickness
    # end __init__

    def draw(
            self,
            context
    ):
        """
        Draw the plot to the context.

        Args:
            context (cairo.Context): Context to draw the plot to
        """
        width = context.get_target().get_width()
        height = context.get_target().get_height()
        x_values = np.linspace(self.x_range[0], self.x_range[1], width)
        y_values = self.function(x_values)
        context.set_source_rgb(*self.color)
        context.set_line_width(self.thickness)
        context.move_to(0, height - y_values[0])
        for i in range(1, len(x_values)):
            context.line_to(i, height - y_values[i])
        # end
        context.stroke()
    # end draw

# end Plot
