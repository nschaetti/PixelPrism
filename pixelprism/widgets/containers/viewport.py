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
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

# Imports
from typing import Tuple, List
import cairo
from pixelprism.widgets.widget import Widget
from pixelprism.base import Context
from pixelprism.data import Point2D, Transform


# Viewport class, container for a scalable and scrollable area
class Viewport(Widget):
    """
    Viewport class, container for a scalable and scrollable area
    """

    # Init
    def __init__(
            self,
            transform: Transform = None
    ):
        """
        Initialize the viewport.

        Args:
            transform (Transform): Transform to apply to the viewport
        """
        super().__init__()
        self._transform = transform
        self._widgets = []
    # end __init__

    # region PROPERTIES

    # Transform
    @property
    def transform(self) -> Transform:
        """
        Get the transform of the viewport.
        """
        return self._transform
    # end transform

    # Widgets
    @property
    def widgets(self) -> List[Widget]:
        """
        Get the widgets of the viewport.
        """
        return self._widgets
    # end widgets

    # endregion PROPERTIES

    # Add widget
    def add_widget(self, widget):
        """
        Add a widget to the viewport.

        Args:
            widget (Widget): The widget to add to the viewport.
        """
        self._widgets.append(widget)
    # end add_widget

    # Draw
    def draw(
            self,
            context: Context,
            *args,
            **kwargs
    ):
        """
        Draw the viewport to the context.

        Args:
            context (cairo.Context): Context to draw the viewport to
        """
        # Transform
        if self.transform: self.context.set_transform(self.transform)

        # Draw the widgets
        for widget in self.widgets:
            widget.draw(context, *args, **kwargs)
        # end for
    # end draw

    # Render
    def render(
            self,
            context: Context,
            *args,
            **kwargs
    ):
        """
        Render the viewport to the surface.

        Args:
            context (Context): Context to render the viewport to
        """
        # Get draw params
        draw_params = kwargs.get("draw_params", {})

        # Draw the widgets
        self.draw(
            context,
            **draw_params
        )
    # end render

# end Viewport
