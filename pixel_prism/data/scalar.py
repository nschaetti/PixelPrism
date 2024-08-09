#
# This file contains the Scalar class, which is used to represent a scalar value.
#

# Imports
from pixel_prism.animate.able import RangeableMixin

from .data import Data
from .eventmixin import EventMixin


class Scalar(Data, EventMixin, RangeableMixin):
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
        if isinstance(value, Scalar):
            value = value.get()
        # end if
        self._value = value
        self.dispatch_event("on_change", value)
    # end set

    def get(self):
        """
        Get the scalar value.
        """
        return self._value
    # end get

    def copy(self):
        """
        Return a copy of the data.
        """
        return Scalar(self._value)
    # end copy

    # region OVERRIDE

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

    # Operator overloading
    def __add__(self, other):
        """
        Add the scalar value to another scalar or value.

        Args:
            other (any): Scalar or value to add
        """
        if isinstance(other, Scalar):
            return Scalar(self._value + other._value)
        # end if
        return Scalar(self._value + other)
    # end __add__

    def __radd__(self, other):
        """
        Add the scalar value to another scalar or value.

        Args:
            other (any): Scalar or value to add
        """
        return self.__add__(other)
    # end __radd__

    def __sub__(self, other):
        if isinstance(other, Scalar):
            return Scalar(self._value - other._value)
        return Scalar(self._value - other)

    # end __sub__

    def __rsub__(self, other):
        """
        Subtract the scalar value from another scalar or value.

        Args:
            other (any): Scalar or value to subtract
        """
        if isinstance(other, Scalar):
            return Scalar(other._value - self._value)
        # end if
        return Scalar(other - self._value)
    # end __rsub__

    def __mul__(self, other):
        """
        Multiply the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to multiply
        """
        if isinstance(other, Scalar):
            return Scalar(self._value * other._value)
        # end if
        return Scalar(self._value * other)
    # end __mul__

    def __rmul__(self, other):
        """
        Multiply the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to multiply
        """
        return self.__mul__(other)
    # end __rmul__

    def __truediv__(self, other):
        """
        Divide the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to divide by
        """
        if isinstance(other, Scalar):
            return Scalar(self._value / other._value)
        # end if
        return Scalar(self._value / other)

    # end __truediv__

    def __rtruediv__(self, other):
        """
        Divide the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to divide by
        """
        if isinstance(other, Scalar):
            return Scalar(other / self._value)
        # end if
        return Scalar(other / self._value)
    # end __rtruediv__

    def __eq__(self, other):
        """
        Check if the scalar value is equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        if isinstance(other, Scalar):
            return self._value == other._value
        # end if
        return self._value == other
    # end __eq__

    def __ne__(self, other):
        """
        Check if the scalar value is not equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        return not self.__eq__(other)
    # end __ne__

    # endregion OVERRIDE

# end Scalar
