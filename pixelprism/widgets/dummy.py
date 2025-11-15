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
