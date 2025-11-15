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
from pixelprism.widgets.widget import Widget
from .container import Container


class PositionedContainer(Container):
    """
    Positioned container class.
    """

    # Create surface
    def create_surface(
            self,
            widget: Widget,
            **kwargs
    ):
        """
        Create a sub-surface for a widget.

        Args:
            widget (Widget): The widget to create a surface for.
            kwargs: Additional arguments (not used in this implementation)
        """
        # Get the widget's position and size
        x = kwargs.get("x", 0)
        y = kwargs.get("y", 0)
        width = kwargs.get("width", 0)
        height = kwargs.get("height", 0)

        # Create a sub-surface for the widget
        sub_surface = self.surface.create_for_rectangle(
            x,
            y,
            width,
            height
        )
        return sub_surface
    # end create_surface

    def add_widget(
            self,
            widget,
            x,
            y,
            width,
            height
    ):
        """
        Add a widget to the container.

        Args:
            widget (Widget): The widget to add to the container.
            x (int): X-coordinate of the widget
            y (int): Y-coordinate of the widget
            width (int): Width of the widget
            height (int): Height of the widget
        """
        super().add_widget(widget, x=x, y=y, width=width, height=height)
    # end add_widget

# end PositionedContainer

