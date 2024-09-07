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
from pixel_prism.data import Point2D as p2


# Coordinate system
class CoordSystem:
    """
    Coordinate system.
    """

    # Initialize
    def __init__(
            self,
            image_width: int,
            image_height: int,
            size: int = 10
    ):
        """
        Initialize the coordinate system.

        Args:
            size (int): Size of the coordinate system
        """
        self._image_width = image_width
        self._image_height = image_height
        self._size = size
        self._width, self._height = self._compute_relative_dimensions()
        self._x_range, self._y_range = self._compute_ranges()

        # Points
        self.center = p2(0, 0, readonly=True)
        self.upper_right = self.ur = p2(self.width / 2, self.height / 2, readonly=True)
        self.upper_right_square = self.urs = p2(self.width / 4, self.height / 4, readonly=True)
        self.lower_right = self.lr = p2(self.width / 2, -self.height / 2, readonly=True)
        self.lower_right_square = self.lrs = p2(self.width / 4, -self.height / 4, readonly=True)
        self.lower_left = self.ll = p2(-self.width / 2, -self.height / 2, readonly=True)
        self.lower_left_square = self.lls = p2(-self.width / 4, -self.height / 4, readonly=True)
        self.upper_left = self.ul = p2(-self.width / 2, self.height / 2, readonly=True)
        self.upper_left_square = self.uls = p2(-self.width / 4, self.height / 4, readonly=True)
        self.middle_bottom = self.mb = p2(0, -self.height / 2, readonly=True)
        self.middle_top = self.mt = p2(0, self.height / 2, readonly=True)
        self.middle_left = self.ml = p2(-self.width / 2, 0, readonly=True)
        self.middle_right = self.mr = p2(self.width / 2, 0, readonly=True)
    # end __init__

    # region PROPERTIES

    @property
    def image_width(self):
        """
        Get the width.
        """
        return self._image_width
    # end image_width

    @property
    def image_height(self):
        """
        Get the height.
        """
        return self._image_height
    # end image_height

    @property
    def size(self):
        """
        Get the size.
        """
        return self._size
    # end size

    @property
    def width(self):
        """
        Get the width.
        """
        return self._width
    # end width

    @property
    def height(self):
        """
        Get the height.
        """
        return self._height
    # end height

    @property
    def x_range(self):
        """
        Get the x range.
        """
        return self._x_range
    # end x_range

    @property
    def y_range(self):
        """
        Get the y range.
        """
        return self._y_range
    # end y_range

    # endregion PROPERTIES

    # region PUBLIC

    def setup(
            self,
            context
    ):
        """
        Configure the Cairo context to use relative coordinates.
        """
        x_scale = self._image_width / self._width
        y_scale = self._image_height / self._height
        context.translate(self._image_width / 2, self._image_height / 2)
        context.scale(x_scale, -y_scale)

        # Anti-aliasing
        context.set_antialias(cairo.Antialias.BEST)
    # end setup

    # endregion PUBLIC

    # region PRIVATE

    # Compute range
    def _compute_ranges(self):
        """
        Compute the ranges.
        """
        x_range = (
            -self._width // 2,
            self.width // 2
        )

        y_range = (
            -self._height // 2,
            self._height // 2
        )

        return x_range, y_range
    # end _compute_ranges

    # Compute relative size
    def _compute_relative_dimensions(self):
        """
        Compute the relative dimensions.
        """
        if self._image_width > self._image_height:
            width = self._size
            height = self._size * self._image_height / self._image_width
        else:
            width = self._size * self._image_width / self._image_height
            height = self._size
        # end

        return width, height
    # end _compute_relative_dimensions

    # endregion PRIVATE
