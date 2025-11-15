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
from typing import List
import cairo
from pixelprism.widgets.widget import Widget


class Container(Widget):

    def __init__(self):
        """
        Initialize the container.
        """
        super().__init__()
        self.widgets = []
    # end __init__

    def add_widget(
            self,
            widget,
            **kwargs
    ):
        """
        Add a widget to the container.

        Args:
            widget (Widget): The widget to add to the container.
            kwargs: Additional arguments (not used in this implementation
        """
        # Add the widget to the container
        self.widgets.append((widget, kwargs))
    # end add_widget

    # Add widgets
    def add_widgets(
            self,
            widgets: List[Widget]
    ):
        """
        Add multiple widgets to the container.

        Args:
            widgets (list): List of widgets to add to the container.
        """
        for widget in widgets:
            self.add_widget(widget)
        # end for
    # end add_widgets

    # Create sub-surface for a widget
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
        raise NotImplementedError("create_surface() must be implemented in a subclass.")
    # end create_surface

    def render(
            self,
            surface: cairo.ImageSurface
    ):
        """
        Render the container to the surface.

        Args:
            surface (cairo.ImageSurface): Surface to render the container to
        """
        # Super call
        super().render(surface)

        # For each widget and parameters
        for widget, kwargs in self.widgets:
            # Create a new sub-surface for the widget
            widget_surface = self.create_surface(widget, **kwargs)
            widget.render(widget_surface)
        # end for
    # end render

# end Container


