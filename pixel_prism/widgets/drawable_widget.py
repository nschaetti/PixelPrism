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
import cairo
from .widget import Widget
from ..drawing import DrawableMixin


class DrawableWidget(Widget):
    """
    Drawable widget
    """

    def __init__(self):
        """
        Initialize the widget.
        """
        super().__init__()
        self.primitives = []
    # end __init__

    # Draw the widget
    def draw(
            self,
            context,
            *args,
            **kwargs
    ):
        """
        Draw the widget to the context.

        Args:
            context (cairo.Context): Context to draw the widget to
        """
        # Save context
        context.save()

        # Antialiasing
        # context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)

        # For each primitive, draw it
        for primitive in self.primitives:
            assert isinstance(primitive, DrawableMixin)
            primitive.draw(context, *args, **kwargs)
        # end for

        # Restore context
        context.restore()
    # end draw

    def add(
            self,
            drawable
    ):
        """
        Add a drawable.

        Args:
            drawable (Drawable): Drawable
        """
        self.primitives.append(drawable)
    # end add

# end DrawableWidget

