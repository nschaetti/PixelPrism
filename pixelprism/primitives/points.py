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

class Point:
    """
    A class to represent a point in 2D space.
    """

    def __init__(self, x, y, size):
        """
        Initialize the point with its coordinates and size.

        Args:
            x (int): X-coordinate of the point
            y (int): Y-coordinate of the point
            size (int): Size of the point
        """
        self.x = x
        self.y = y
        self.size = size
    # end __init__

    def __repr__(self):
        """
        Return a string representation of the point.
        """
        return f"Point(x={self.x}, y={self.y}, size={self.size})"
    # end __repr__

# end Point
