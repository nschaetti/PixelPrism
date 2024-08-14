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

#
# Bounding box mixin
#

# Imports
from pixel_prism import utils
from pixel_prism.base import Context
from pixel_prism.data import Color, Scalar, Point2D


class BoundingBoxMixin(object):
    """
    A mixin class to add bounding box functionality to a class.
    """

    def __init__(
            self
    ):
        """
        Initialize the bounding box mixin.
        """
        super().__init__()

        # Bounding box
        self._bounding_box = self._create_bbox()
    # end __init__

    # region PROPERTIES

    # Bounding box
    @property
    def bounding_box(self):
        return self._bounding_box
    # end bounding_box

    # endregion PROPERTIES

    # region PUBLIC

    # Update bounding box
    def update_bbox(
            self
    ):
        """
        Update the bounding box of the drawable.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.update_bbox method must be implemented in subclass.")
    # end update_bbox

    # Draw bounding box anchors
    def draw_bbox_anchors(
            self,
            context: Context,
            point_size: float = 0.08,
            font_size: float = 0.1
    ):
        """
        Draw the bounding box anchors to the context.

        Args:
            context (cairo.Context): Context to draw the bounding box anchors to
            point_size (float): Size of the points
            font_size (float): Size of the font
        """
        if self.bounding_box is not None:
            self.bounding_box.draw_anchors(
                context=context,
                point_size=point_size,
                font_size=font_size
            )
        # end if
    # end draw_bbox_anchors

    # Draw bounding box
    def draw_bbox(
            self,
            context: Context,
            border_width: Scalar
    ):
        """
        Draw the bounding box to the context.

        Args:
            context (cairo.Context): Context to draw the bounding box to (Cairo context)
            border_width (Scalar): Width of the border
        """
        if self.bounding_box is not None:
            self.bounding_box.draw(
                context=context,
                border_width=border_width
            )
        # end if
    # end draw_bbox

    # endregion PUBLIC

    # region PRIVATE

    # Create bounding box
    def _create_bbox(
            self
    ):
        """
        Create the bounding box.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _create_bbox method."
        )
    # end _create_bbox

    # endregion PRIVATE

# end BoundingBoxMixin
