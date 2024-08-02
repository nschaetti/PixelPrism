#
#
#

# Imports
import numpy as np
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
        self.value = np.array([red, green, blue, alpha])
    # end __init__

    @property
    def red(self):
        return self.value[0]
    # end red

    @red.setter
    def red(self, value):
        self.value[0] = value
    # end red

    @property
    def green(self):
        return self.value[1]
    # end green

    @green.setter
    def green(self, value):
        self.value[1] = value
    # end green

    @property
    def blue(self):
        return self.value[2]
    # end blue

    @blue.setter
    def blue(self, value):
        self.value[2] = value
    # end blue

    @property
    def alpha(self):
        return self.value[3]
    # end alpha

    @alpha.setter
    def alpha(self, value):
        self.value[3] = value
    # end alpha

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
        self.value = np.array([red, green, blue, alpha])
    # end set

    def get(self):
        """
        Get the scalar value.
        """
        return self.value[0], self.value[1], self.value[2], self.value[3]
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
