#
# Data classes for Pixel Prism
#


class Data(object):
    """
    A data class that holds data.
    """

    def __init__(self):
        """
        Initialize the scalar value.
        """
        super().__init__()
    # end __init__

    def set(self, *args, **kwargs):
        """
        Set the scalar value.

        Args:
            value (any): Value to set
        """
        raise NotImplementedError(f"{self.__class__.__name__}.set method must be implemented in subclass.")
    # end set

    def get(self):
        """
        Get the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.get method must be implemented in subclass.")
    # end get

    def copy(self):
        """
        Return a copy of the data.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.copy method must be implemented in subclass.")
    # end copy

    def __str__(self):
        """
        Return a string representation of the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.__str__ method must be implemented in subclass.")
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the scalar value.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.__repr__ method must be implemented in subclass.")
    # end __repr__

# end Scalar
