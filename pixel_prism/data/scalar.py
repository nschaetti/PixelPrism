#
# This file contains the Scalar class, which is used to represent a scalar value.
#

# Imports
from pixel_prism.animate.able import RangeableMixin

from .data import Data


class Scalar(Data, RangeableMixin):
    """
    A class to represent a scalar value
    """

    def __init__(
            self,
            value=0,
            on_change=None
    ):
        """
        Initialize the scalar value.
        """
        Data.__init__(self)
        RangeableMixin.__init__(self, "value")

        # Value
        self._value = value

        # List of event listeners (per events)
        self.event_listeners = {
            "on_change": [] if on_change is None else [on_change]
        }
    # end __init__

    @property
    def value(self):
        """
        Get the scalar value.
        """
        return self.get()
    # end value

    @value.setter
    def value(self, value):
        """
        Set the scalar value.
        """
        self.set(value)
    # end value

    def set(self, value):
        """
        Set the scalar value.

        Args:
            value (any): Value to set
        """
        self._value = value
        self.dispatch_event("on_change", value)
    # end set

    def get(self):
        """
        Get the scalar value.
        """
        return self._value
    # end get

    def __str__(self):
        """
        Return a string representation of the scalar value.
        """
        return str(self._value)
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the scalar value.
        """
        return f"Scalar(value={self._value})"
    # end __repr__

# end Scalar
