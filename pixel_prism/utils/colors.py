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

import random
from pixel_prism.data import Color


# Colors
RED = Color(255, 0, 0)
GREEN = Color(0, 255, 0)
BLUE = Color(0, 0, 255)
WHITE = Color(255, 255, 255)
BLACK = Color(0, 0, 0)
YELLOW = Color(255, 255, 0)
MAGENTA = Color(255, 0, 255)
DARK_CYAN = Color(0, 139, 139)
TEAL = Color(0, 128, 128)
DARK_SLATE_GRAY = Color(47, 79, 79)

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
    DARK_SLATE_GRAY
]


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
