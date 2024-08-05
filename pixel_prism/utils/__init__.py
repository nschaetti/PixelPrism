
from enum import Enum

from .colors import (
    BLUE,
    BLACK,
    GREEN,
    RED,
    WHITE,
    YELLOW,
    colors,
    random_color
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
    "colors",
    "random_color",
    # Latex
    "render_latex_to_svg",
    "setup_logger",
    "CustomFormatter",
    # SVG
    "draw_svg",
    "parse_svg",
    "parse_path"
]

