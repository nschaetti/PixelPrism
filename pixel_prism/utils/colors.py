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
from pixel_prism.data import Color, Scalar as s


# Colors
RED = Color.from_objects(s(255), s(0), s(0), readonly=True)
GREEN = Color.from_objects(s(0), s(255), s(0), readonly=True)
BLUE = Color.from_objects(s(0), s(0), s(255), readonly=True)
WHITE = Color.from_objects(s(255), s(255), s(255), readonly=True)
BLACK = Color.from_objects(s(0), s(0), s(0), readonly=True)
YELLOW = Color.from_objects(s(255), s(255), s(0), readonly=True)
MAGENTA = Color.from_objects(s(255), s(0), s(255), readonly=True)
DARK_CYAN = Color.from_objects(s(0), s(139), s(139), readonly=True)
TEAL = Color.from_objects(s(0), s(128), s(128), readonly=True)
DARK_SLATE_GRAY = Color.from_objects(s(47), s(79), s(79), readonly=True)
ORANGE = Color.from_objects(s(255), s(165), s(0), readonly=True)
PURPLE = Color.from_objects(s(128), s(0), s(128), readonly=True)
PINK = Color.from_objects(s(255), s(192), s(203), readonly=True)
BROWN = Color.from_objects(s(165), s(42), s(42), readonly=True)
LIGHT_GRAY = Color.from_objects(s(211), s(211), s(211), readonly=True)
DARK_GRAY = Color.from_objects(s(169), s(169), s(169), readonly=True)
CYAN = Color.from_objects(s(0), s(255), s(255), readonly=True)
LIME = Color.from_objects(s(50), s(205), s(50), readonly=True)
GOLD = Color.from_objects(s(255), s(215), s(0), readonly=True)
INDIGO = Color.from_objects(s(75), s(0), s(130), readonly=True)
SILVER = Color.from_objects(s(192), s(192), s(192), readonly=True)
MAROON = Color.from_objects(s(128), s(0), s(0), readonly=True)
NAVY = Color.from_objects(s(0), s(0), s(128), readonly=True)
OLIVE = Color.from_objects(s(128), s(128), s(0), readonly=True)
TURQUOISE = Color.from_objects(s(64), s(224), s(208), readonly=True)
VIOLET = Color.from_objects(s(238), s(130), s(238), readonly=True)
SALMON = Color.from_objects(s(250), s(128), s(114), readonly=True)

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
