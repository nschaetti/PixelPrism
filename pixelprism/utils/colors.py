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
import random
from pixelprism.math_old import Color, Scalar as s


# Colors
RED = Color.from_objects(s(1.0), s(0), s(0), readonly=True)
GREEN = Color.from_objects(s(0), s(1.0), s(0), readonly=True)
BLUE = Color.from_objects(s(0), s(0), s(1.0), readonly=True)
WHITE = Color.from_objects(s(1.0), s(1.0), s(1.0), readonly=True)
BLACK = Color.from_objects(s(0), s(0), s(0), readonly=True)
YELLOW = Color.from_objects(s(1.0), s(1.0), s(0), readonly=True)
MAGENTA = Color.from_objects(s(1.0), s(0), s(1.0), readonly=True)
DARK_CYAN = Color.from_objects(s(0), s(0.5450), s(0.5450), readonly=True)
TEAL = Color.from_objects(s(0), s(0.5), s(0.5), readonly=True)
DARK_SLATE_GRAY = Color.from_objects(s(0.184313725), s(0.309803922), s(0.309803922), readonly=True)
ORANGE = Color.from_objects(s(1.0), s(0.647058824), s(0), readonly=True)
PURPLE = Color.from_objects(s(0.5), s(0), s(0.5), readonly=True)
PINK = Color.from_objects(s(1.0), s(0.752941176), s(0.796078431), readonly=True)
BROWN = Color.from_objects(s(0.647058824), s(0.164705882), s(0.164705882), readonly=True)
LIGHT_GRAY = Color.from_objects(s(0.82745098), s(0.82745098), s(0.82745098), readonly=True)
DARK_GRAY = Color.from_objects(s(0.662745098), s(0.662745098), s(0.662745098), readonly=True)
DARKER_GRAY = Color.from_objects(s(0.462745098), s(0.462745098), s(0.462745098), readonly=True)
CYAN = Color.from_objects(s(0), s(1.0), s(1.0), readonly=True)
LIME = Color.from_objects(s(0.196078431), s(0.803921569), s(0.196078431), readonly=True)
GOLD = Color.from_objects(s(1.0), s(0.843137255), s(0), readonly=True)
INDIGO = Color.from_objects(s(0.294117647), s(0), s(0.509803922), readonly=True)
SILVER = Color.from_objects(s(0.752941176), s(0.752941176), s(0.752941176), readonly=True)
MAROON = Color.from_objects(s(0.5), s(0), s(0), readonly=True)
NAVY = Color.from_objects(s(0), s(0), s(0.5), readonly=True)
OLIVE = Color.from_objects(s(128), s(0.5), s(0), readonly=True)
TURQUOISE = Color.from_objects(s(0.25), s(0.878431373), s(0.815686275), readonly=True)
VIOLET = Color.from_objects(s(0.921568627), s(0.509803922), s(0.933333333), readonly=True)
SALMON = Color.from_objects(s(0.980392157), s(0.5), s(0.447058824), readonly=True)

# Colors
colors = [
    RED,
    GREEN,
    BLUE,
    WHITE,
    BLACK,
    YELLOW,
    MAGENTA,
    DARK_CYAN,
    TEAL,
    DARK_SLATE_GRAY,
    ORANGE,
    PURPLE,
    PINK,
    BROWN,
    LIGHT_GRAY,
    DARK_GRAY,
    CYAN,
    LIME,
    GOLD,
    INDIGO,
    SILVER,
    MAROON,
    NAVY,
    OLIVE,
    TURQUOISE,
    VIOLET,
    SALMON
]


# Get a color from hexadecimal string
def from_hex(hex_string: str, alpha: float = 1.0):
    """
    Get a color from a hexadecimal string.

    Args:
        hex_string (str): Hexadecimal string
        alpha (float): Alpha value

    Returns:
        Color: Color
    """
    return Color.from_hex(hex_string, alpha)
# end from_hex



# Get a random color
def random_color():
    """
    Get a random color from the list of colors.
    """
    return random.choice([
        RED,
        GREEN,
        BLUE,
        YELLOW,
        MAGENTA,
        DARK_CYAN,
        TEAL,
        DARK_SLATE_GRAY
    ])
# end random_color
