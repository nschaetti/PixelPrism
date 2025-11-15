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
import cairo
from pixelprism.drawing import DrawableMixin
from .widget import Widget


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
        if isinstance(drawable, DrawableMixin):
            self.primitives.append(drawable)
        elif isinstance(drawable, (list, tuple)):
            for d in drawable:
                self.add(d)
            # end for
        elif isinstance(drawable, dict):
            for d in drawable.values():
                self.add(d)
            # end for
        else:
            raise ValueError(f"Drawable object of type {type(drawable)} is not supported.")
        # end if
    # end add

# end DrawableWidget

