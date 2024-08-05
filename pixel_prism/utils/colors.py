
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
