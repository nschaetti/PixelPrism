#
# This file contains the Scalar class, which is used to represent a scalar value.
#
from typing import List, Dict

# Imports
import numpy as np
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

    # Override less
    def __lt__(self, other):
        """
        Check if the scalar value is less than another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        if isinstance(other, Scalar):
            return self._value < other._value
        # end if
        return self._value < other
    # end __lt__

    # Override less or equal
    def __le__(self, other):
        """
        Check if the scalar value is less than or equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        if isinstance(other, Scalar):
            return self._value <= other._value
        # end if
        return self._value <= other
    # end __le__

    # Override greater
    def __gt__(self, other):
        """
        Check if the scalar value is greater than another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        if isinstance(other, Scalar):
            return self._value > other._value
        # end if
        return self._value > other
    # end __gt__

    # Override greater or equal
    def __ge__(self, other):
        """
        Check if the scalar value is greater than or equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        if isinstance(other, Scalar):
            return self._value >= other._value
        # end if
        return self._value >= other
    # end __ge__

    # endregion OVERRIDE

# end Scalar


class TScalar(Scalar):
    """
    A class to represent a scalar value that is dynamically computed based on other Scalars.
    """

    def __init__(self, func, on_change=None, **scalars):
        """
        Initialize the TScalar.

        Args:
            func (function): A function that computes the value dynamically.
            on_change (function): A function to call when the value changes.
            scalars: Scalar objects that this TScalar depends on.
        """
        self._func = func
        self._scalars = scalars

        # Initialize the base class with the computed value
        initial_value = self._func(**self._scalars)
        super().__init__(initial_value)

        # Listen to changes in the original point
        if on_change is not None:
            for scalar in self._scalars.values():
                scalar.add_event_listener("on_change", on_change)
            # end for
        # end if
    # end __init__

    # region PROPERTIES

    @property
    def func(self):
        """
        Get the function that computes the value.
        """
        return self._func
    # end func

    @property
    def scalars(self):
        """
        Source scalar
        """
        return self._scalars
    # end sources

    @property
    def value(self):
        """
        Get the scalar value.
        """
        self._value = self.get()
        return self._value
    # end value

    @value.setter
    def value(self, value):
        """
        Set the scalar value.
        """
        raise AttributeError("Cannot set value directly on TScalar. It's computed based on other Scalars.")
    # end value

    # endregion PROPERTIES

    # region PUBLIC

    # Override set to prevent manual setting
    def set(self, value):
        """
        Prevent manual setting of the value. It should be computed only.
        """
        raise AttributeError("Cannot set value directly on TScalar. It's computed based on other Scalars.")
    # end set

    def get(self):
        """
        Get the current computed value.
        """
        return self._func(**self._scalars)
    # end get

    def scalar(self):
        """
        Get the current computed value.
        """
        return Scalar(self._func(**self._scalars))
    # end get

    # endregion PUBLIC

    # region OVERRIDE

    def __eq__(self, other):
        """
        Check if the scalar value is equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        if isinstance(other, Scalar):
            return self.value == other.value
        # end if
        return self.value == other
    # end __eq__

    def __ne__(self, other):
        """
        Check if the scalar value is not equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        return not self.__eq__(other)
    # end __ne__

    # Override less
    def __lt__(self, other):
        """
        Check if the scalar value is less than another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        if isinstance(other, Scalar):
            return self.value < other.value
        # end if
        return self.value < other
    # end __lt__

    # Override less or equal
    def __le__(self, other):
        """
        Check if the scalar value is less than or equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        if isinstance(other, Scalar):
            return self.value <= other.value
        # end if
        return self.value <= other
    # end __le__

    # Override greater
    def __gt__(self, other):
        """
        Check if the scalar value is greater than another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        if isinstance(other, Scalar):
            return self.value > other.value
        # end if
        return self.value > other
    # end __gt__

    # Override greater or equal
    def __ge__(self, other):
        """
        Check if the scalar value is greater than or equal to another scalar or value.

        Args:
            other (any): Scalar or value to compare
        """
        if isinstance(other, Scalar):
            return self.value >= other.value
        # end if
        return self.value >= other
    # end __ge__

    # endregion OVERRIDE

# end TScalar


def floor_t(scalar: Scalar):
    """
    Create a TScalar that applies the floor function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the floor function to.
    """
    return TScalar(lambda s: np.floor(s.value), s=scalar)
# end floor_t


def ceil_t(scalar: Scalar):
    """
    Create a TScalar that applies the ceil function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the ceil function to.
    """
    return TScalar(lambda s: np.ceil(s.value), s=scalar)
# end ceil_t


def trunc_t(scalar: Scalar):
    """
    Create a TScalar that applies the trunc function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the trunc function to.
    """
    return TScalar(lambda s: np.trunc(s.value), s=scalar)
# end trunc_t


def frac_t(scalar: Scalar):
    """
    Create a TScalar that returns the fractional part of the scalar.

    Args:
        scalar (Scalar): The scalar to get the fractional part of.
    """
    return TScalar(lambda s: s.value - np.floor(s.value), s=scalar)
# end frac_t


def sqrt_t(scalar: Scalar):
    """
    Create a TScalar that applies the sqrt function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the sqrt function to.
    """
    return TScalar(lambda s: np.sqrt(s.value), s=scalar)
# end sqrt_t


def exp_t(scalar: Scalar):
    """
    Create a TScalar that applies the exp function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the exp function to.
    """
    return TScalar(lambda s: np.exp(s.value), s=scalar)
# end exp_t


def expm1_t(scalar: Scalar):
    """
    Create a TScalar that applies the expm1 function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the expm1 function to.
    """
    return TScalar(lambda s: np.expm1(s.value), s=scalar)
# end expm1_t


def log_t(scalar: Scalar):
    """
    Create a TScalar that applies the log function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the log function to.
    """
    return TScalar(lambda s: np.log(s.value), s=scalar)
# end log_t


def log1p_t(scalar: Scalar):
    """
    Create a TScalar that applies the log1p function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the log1p function to.
    """
    return TScalar(lambda s: np.log1p(s.value), s=scalar)
# end log1p_t


def log2_t(scalar: Scalar):
    """
    Create a TScalar that applies the log2 function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the log2 function to.
    """
    return TScalar(lambda s: np.log2(s.value), s=scalar)
# end log2_t


def log10_t(scalar: Scalar):
    """
    Create a TScalar that applies the log10 function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the log10 function to.
    """
    return TScalar(lambda s: np.log10(s.value), s=scalar)
# end log10_t

def sin_t(scalar: Scalar):
    """
    Create a TScalar that applies the sin function to the scalar.

    Args:
        scalar (Scalar): The scalar to apply the sin function to.
    """
    return TScalar(lambda s: np.sin(s.value), s=scalar)
# end sin_t

def cos_t(scalar: Scalar):
    return TScalar(lambda s: np.cos(s.value), s=scalar)
# end cos_t

def tan_t(scalar: Scalar):
    return TScalar(lambda s: np.tan(s.value), s=scalar)
# end tan_t

def asin_t(scalar: Scalar):
    return TScalar(lambda s: np.arcsin(s.value), s=scalar)
# end asin_t

def acos_t(scalar: Scalar):
    return TScalar(lambda s: np.arccos(s.value), s=scalar)
# end acos_t

def atan_t(scalar: Scalar):
    return TScalar(lambda s: np.arctan(s.value), s=scalar)
# end atan_t

def atan2_t(y: Scalar, x: Scalar):
    return TScalar(lambda p1, p2: np.arctan2(p1.value, p2.value), p1=y, p2=x)
# end atan2_t

def sinh_t(scalar: Scalar):
    return TScalar(lambda s: np.sinh(s.value), s=scalar)
# end sinh_t

def cosh_t(scalar: Scalar):
    return TScalar(lambda s: np.cosh(s.value), s=scalar)
# end cosh_t

def tanh_t(scalar: Scalar):
    return TScalar(lambda s: np.tanh(s.value), s=scalar)
# end tanh_t

def asinh_t(scalar: Scalar):
    return TScalar(lambda s: np.arcsinh(s.value), s=scalar)
# end asinh_t

def acosh_t(scalar: Scalar):
    return TScalar(lambda s: np.arccosh(s.value), s=scalar)
# end acosh_t

def atanh_t(scalar: Scalar):
    return TScalar(lambda s: np.arctanh(s.value), s=scalar)
# end atanh_t

def degrees_t(scalar: Scalar):
    return TScalar(lambda s: np.degrees(s.value), s=scalar)
# end degrees_t



