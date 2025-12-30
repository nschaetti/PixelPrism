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
from typing import Union

# Imports
from pixelprism import p2, s, c
from pixelprism.base import Context, CoordSystem
from pixelprism.animate import animeattr
from pixelprism.math_old import Scalar, Style
from .drawablemixin import DrawableMixin


# A drawable which show debug grid
@animeattr("grid_style")
@animeattr("axis_style")
@animeattr("major_width")
@animeattr("minor_width")
class DebugGrid(DrawableMixin):

    # Constructor
    def __init__(
            self,
            coord_system: CoordSystem,
            grid_style: Style,
            axis_style: Style,
            major_width: Scalar,
            minor_width: Scalar
    ):
        """
        Initialize the debug grid.

        Args:
            coord_system (CoordSystem): Coordinate system
            grid_style (Style): Grid style
            axis_style (Style): Axis style
            major_width (Scalar): Major line width
            minor_width (Scalar): Minor line width
        """
        # Drawable
        super().__init__()

        # Properties
        self._coord_system = coord_system
        self._grid_style = grid_style
        self._axis_style = axis_style
        self._major_width = major_width
        self._minor_width = minor_width
    # end __init__

    # region PROPERTIES

    @property
    def coord_system(self) -> CoordSystem:
        """
        Get the coordinate system.

        Returns:
            CoordSystem: Coordinate system
        """
        return self._coord_system
    # end coord_system

    @property
    def grid_style(self) -> Style:
        """
        Get the grid style.

        Returns:
            Style: Grid style
        """
        return self._grid_style
    # end grid_style

    @property
    def axis_style(self) -> Style:
        """
        Get the axis style.

        Returns:
            Style: Axis style
        """
        return self._axis_style
    # end axis_style

    @property
    def major_width(self) -> Scalar:
        """
        Get the major line width.

        Returns:
            Scalar: Major line width
        """
        return self._major_width
    # end major_width

    @property
    def minor_width(self) -> Scalar:
        """
        Get the minor line width.

        Returns:
            Scalar: Minor line width
        """
        return self._minor_width
    # end minor_width

    # endregion PROPERTIES

    # region DRAW

    def draw(
            self,
            context: Context,
            *args,
            **kwargs
    ):
        """
        Draw the debug grid.

        Args:
            context (Context): Context

        """
        # Save context
        context.save()

        # Set line width
        context.set_line_width(self.grid_style.line_width)
        context.set_source_rgba(self.grid_style.line_color)

        # Draw major grid
        self._draw_grid(
            context,
            x_range=self.coord_system.x_range[1],
            y_range=self.coord_system.y_range[1],
            grid_size=self.major_width
        )

        # Draw minor grid
        self._draw_grid(
            context,
            x_range=self.coord_system.x_range[1],
            y_range=self.coord_system.y_range[1],
            grid_size=self.minor_width,
            dashes=[1, 0.5, 0.5]
        )

        # Draw the axis
        context.set_line_width(self.axis_style.line_width * 2)
        context.set_source_rgba(self.axis_style.line_color)
        context.set_dash([])
        context.cairo_move_to(p2(self.coord_system.x_range[0], 0))
        context.cairo_line_to(p2(self.coord_system.x_range[1], 0))
        context.cairo_move_to(p2(0, self.coord_system.y_range[0]))
        context.cairo_line_to(p2(0, self.coord_system.y_range[1]))
        context.stroke()

        # Restore context
        context.restore()
    # end draw

    # endregion DRAW

    # region PRIVATE

    # Draw major
    def _draw_grid(
            self,
            context: Context,
            x_range: Union[Scalar, int],
            y_range: Union[Scalar, int],
            grid_size: Union[Scalar, int],
            dashes: list = None
    ):
        """
        Draw major grid.

        Args:
            context (Context): Context
            grid_size (int): Grid size
        """
        # X and Y range, grid size
        x_range = int(x_range)
        y_range = int(y_range)
        grid_size = int(grid_size)

        # Set dashes
        if dashes is not None:
            context.set_dash(dashes)
        # end if

        # Major and minor grid
        # Draw grid starting from 0 to ranges
        for x in range(0, x_range + 1, grid_size):
            # Right to axis
            context.cairo_move_to(p2(x, self.coord_system.y_range[0]))
            context.cairo_line_to(p2(x, self.coord_system.y_range[1]))
            context.stroke()

            # Left to axis
            context.cairo_move_to(p2(-x, self.coord_system.y_range[0]))
            context.cairo_line_to(p2(-x, self.coord_system.y_range[1]))
            context.stroke()
        # end for

        # Set dashes
        if dashes is not None:
            context.set_dash([dashes[-1], *dashes[:-1]])
        # end if

        for y in range(0, y_range + 1, grid_size):
            # Up to axis
            context.cairo_move_to(p2(self.coord_system.x_range[0], y))
            context.cairo_line_to(p2(self.coord_system.x_range[1], y))
            context.stroke()

            # Down to axis
            context.cairo_move_to(p2(self.coord_system.x_range[0], -y))
            context.cairo_line_to(p2(self.coord_system.x_range[1], -y))
            context.stroke()
        # end for
    # end _draw_grid

    # endregion PRIVATE

# end DebugGrid
