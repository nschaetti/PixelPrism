#
#
#

# Imports
from pixel_prism.animate.able import RangeAble

from .data import Data


class Color(Data, RangeAble):
    """
    A class to represent a scalar value
    """

    def __init__(self, red=0, green=0, blue=0, alpha=1):
        """
        Initialize the scalar value.

        Args:
            red (int): Red value
            green (int): Green value
            blue (int): Blue value
            alpha (int): Alpha value
        """
        super().__init__()
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
    # end __init__

    @property
    def value(self):
        return (
            self.red,
            self.green,
            self.blue
        )
    # end value

    @property
    def opacity(self):
        return self.alpha
    # end opacity

    def set(self, red, green, blue, alpha):
        """
        Set the scalar value.

        Args:
            red (int): Red value
            green (int): Green value
            blue (int): Blue value
            alpha (int): Alpha value
        """
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
    # end set

    def get(self):
        """
        Get the scalar value.
        """
        return self.red, self.green, self.blue, self.alpha
    # end get

    def __str__(self):
        """
        Return a string representation of the scalar value.
        """
        return f"Color(red={self.red}, green={self.green}, blue={self.blue}, alpha={self.alpha})"
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the scalar value.
        """
        return self.__str__()
    # end __repr__

# end Color
