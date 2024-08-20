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
import importlib
from pixel_prism.data import Point2D, Scalar, Color
import pixel_prism.utils as utils


# Create a point
def p2(
        x: float,
        y: float
) -> Point2D:
    """
    Create a point.

    Args:
        x (float): X-coordinate
        y (float): Y-coordinate

    Returns:
        Point2D: Point
    """
    return Point2D(x, y)
# end p2


# Create a scalar
def s(
        value: float
) -> Scalar:
    """
    Create a scalar.

    Args:
        value (float): Value

    Returns:
        Scalar: Scalar
    """
    return Scalar(value)
# end s


# Get a color from name with importlib
def c(
        name: str
) -> Color:
    """
    Get a color from its name.

    Args:
        name (str): Color name

    Returns:
        Color: Color
    """
    return getattr(utils, name)
# end c
