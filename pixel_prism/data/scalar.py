#
# This file contains the Scalar class, which is used to represent a scalar value.
#

# Imports
from pixel_prism.animate.able import RangeAble

from .data import Data


class Scalar(Data, RangeAble):
    """
    A class to represent a scalar value
    """

    def __init__(self, value=0):
        """
        Initialize the scalar value.
        """
        super().__init__()
        self.value = value
    # end __init__

    def set(self, value):
        """
        Set the scalar value.

        Args:
            value (any): Value to set
        """
        self.value = value
    # end set

    def get(self):
        """
        Get the scalar value.
        """
        return self.value
    # end get

    def __str__(self):
        """
        Return a string representation of the scalar value.
        """
        return str(self.value)
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the scalar value.
        """
        return f"Scalar(value={self.value})"
    # end __repr__

# end Scalar
