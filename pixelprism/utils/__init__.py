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

"""
Pixel Prism Utils - Utility Functions and Constants
================================================

This subpackage provides utility functions and constants used throughout the Pixel Prism library.
These include color definitions, LaTeX rendering, logging utilities, and SVG parsing functions.

Main Components
--------------
- Color Utilities:
  - Predefined colors: `BLACK`, `WHITE`, `RED`, `GREEN`, `BLUE`, etc.
  - :func:`~pixelprism.utils.colors.random_color`: Generate a random color
  - :func:`~pixelprism.utils.colors.from_hex`: Convert a hex color string to RGB

- LaTeX Utilities:
  - :func:`~pixelprism.utils.latex.render_latex_to_svg`: Render LaTeX to SVG

- Logging Utilities:
  - :func:`~pixelprism.utils.logging.setup_logger`: Set up a logger with custom formatting
  - :class:`~pixelprism.utils.logging.CustomFormatter`: Custom formatter for logging

- SVG Utilities:
  - :func:`~pixelprism.utils.svg.draw_svg`: Draw an SVG on an image
  - :func:`~pixelprism.utils.svg.parse_svg`: Parse an SVG file
  - :func:`~pixelprism.utils.svg.parse_path`: Parse an SVG path

- Enumerations:
  - :class:`~pixelprism.utils.Anchor`: Enumeration of anchor points for positioning

These utilities provide common functionality used throughout the Pixel Prism library.
"""


from enum import Enum

from .colors import (
    BLUE,
    BLACK,
    GREEN,
    RED,
    WHITE,
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
    DARKER_GRAY,
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
    SALMON,
    colors,
    random_color,
    from_hex
)

# Imports
from .latex import render_latex_to_svg
from .logging import setup_logger, CustomFormatter
from .svg import draw_svg, parse_svg, parse_path


# Anchor points (enum)
class Anchor(Enum):
    UPPER_LEFT = 1
    UPPER_CENTER = 2
    UPPER_RIGHT = 3
    MIDDLE_LEFT = 4
    MIDDLE_CENTER = 5
    MIDDLE_RIGHT = 6
    LOWER_LEFT = 7
    LOWER_CENTER = 8
    LOWER_RIGHT = 9
# end Anchor


# ALL
__all__ = [
    # Anchor points
    "Anchor",
    # Colors
    "BLUE",
    "BLACK",
    "GREEN",
    "RED",
    "WHITE",
    "YELLOW",
    "ORANGE",
    "PURPLE",
    "PINK",
    "BROWN",
    "LIGHT_GRAY",
    "DARK_GRAY",
    "CYAN",
    "LIME",
    "GOLD",
    "INDIGO",
    "SILVER",
    "MAROON",
    "NAVY",
    "OLIVE",
    "TURQUOISE",
    "VIOLET",
    "SALMON",
    "colors",
    "random_color",
    "from_hex",
    # Latex
    "render_latex_to_svg",
    "setup_logger",
    "CustomFormatter",
    # SVG
    "draw_svg",
    "parse_svg",
    "parse_path"
]
