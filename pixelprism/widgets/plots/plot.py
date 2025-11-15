# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2024 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
