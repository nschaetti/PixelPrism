#
# This file is part of the Pixel Prism distribution (https://github.com/nschaetti/PixelPrism).
# Copyright (c) 2024 Nils Schaetti.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
from abc import ABC
#
# This file contains the Scalar class, which is used to represent a scalar value.
#


# Imports
from typing import List, Dict, Union
import numpy as np
from pixelprism.animate import RangeableMixin
from .data import Data

from .math_expr import MathExpr


# Type hint for addition
# Scalar + float = MathExpr
# Scalar + Scalar = Scalar
# Scalar + Point2D = Point2D
# Scalar + Matrix2D = Matrix2D

# Scalar class
class Scalar(MathExpr, RangeableMixin, ABC):
    """
    A class to represent a scalar value
    """

    def __init__(
            self,
            value=0,
            on_change=None,
            readonly: bool = False
    ):
        """
        Initialize the scalar value.

        Args:
            value (any): Initial value of the scalar.
            on_change (function): Function to call when the value changes.
            readonly (bool): Read only flag.
        """
        # Initialize RangeableMixin (animation)
        RangeableMixin.__init__(self, "value")

        # Not a scalar in a scalar
        if isinstance(value, Scalar):
            value = value.get()
        # end if

        # Value
        self._value = value
    # end __init__

    # region PROPERTIES

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

    @property
    def on_change(self):
        """
        Get the on change event.
        """
        return self._on_change
    # end on_change

    # endregion PROPERTIES

    # region PUBLIC

    def set(self, value):
        """
        Set the scalar value.

        Args:
            value (any): Value to set
        """
        self.check_closed()
        if isinstance(value, Scalar):
            value = value.get()
        # end if
        self._value = value
        self._trigger_on_change()
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

    def register_event(self, event_name, listener):
        """
        Add an event listener to the data object.

        Args:
            event_name (str): Event to listen for
            listener (function): Listener function
        """
        if hasattr(self, event_name):
            event_attr = getattr(self, event_name)
            event_attr += listener
        # end if
    # end register_event

    def unregister_event(self, event_name, listener):
        """
        Remove an event listener from the data object.

        Args:
            event_name (str): Event to remove listener from
            listener (function): Listener function to remove
        """
        # Unregister from all sources
        if hasattr(self, event_name):
            event_attr = getattr(self, event_name)
            event_attr -= listener
        # end if
    # end unregister_event

    # To list
    def to_list(self):
        """
        Convert the scalar to a list.
        """
        return [self._value]
    # end to_list

    # endregion PUBLIC

    # region PRIVATE

    # Trigger on change event
    def _trigger_on_change(self):
        """
        Trigger the on change event.
        """
        self._on_change.trigger(self, event_type=EventType.VALUE_CHANGED, value=self.value)
    # end _trigger_on_change

    # endregion PRIVATE

    # region OVERRIDE

    # Override the integer conversion
    def __int__(self):
        """
        Return the integer representation of the scalar value.
        """
        return int(self._value)
    # end __int__

    # Override the float conversion
    def __float__(self):
        """
        Return the float representation of the scalar value.
        """
        return float(self._value)
    # end __float__

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
        # Imports
        from .points import Point2D, TPoint2D
        from .matrices import Matrix2D, TMatrix2D

        # Scalar, TScalar
        if isinstance(other, float) or isinstance(other, int):
            # Scalar + float = Scalar
            return Scalar(self.value + other)
        elif isinstance(other, TScalar):
            # Scalar + TScalar = TScalar
            return TScalar(lambda s, o: s.value + o.value, s=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar + Scalar = Scalar
            return Scalar(self.value + other.value)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Scalar + TPoint2D = TPoint2D
            return TPoint2D(lambda s, p: (s.value + p.x, s.value + p.y), s=self, p=other)
        elif isinstance(other, Point2D):
            # Scalar + Point2D = Point2D
            return Point2D(self.value + other.x, self.value + other.y)
        elif isinstance(other, TMatrix2D):
            # Scalar + TMatrix2D = TMatrix2D
            return TMatrix2D(lambda s, m: m.data + s.value, s=self, m=other)
        elif isinstance(other, Matrix2D):
            # Scalar + Matrix2D = Matrix2D
            return Matrix2D(self.value + other.data)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Scalar' and '{}'".format(type(other)))
        # end if
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
        """
        Subtract the scalar value from another scalar or value.

        Args:
            other (any): Scalar or value to subtract
        """
        # Imports
        from .points import Point2D, TPoint2D
        from .matrices import Matrix2D, TMatrix2D

        # float, int
        if isinstance(other, float) or isinstance(other, int):
            # Scalar - float = Scalar
            return Scalar(self.value - other)
        # Scalar, TScalar
        elif isinstance(other, TScalar):
            # Scalar - TScalar = TScalar
            return TScalar(lambda s, o: s.value - o.value, s=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar - Scalar = Scalar
            return Scalar(self.value - other.value)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Scalar - TPoint2D = TPoint2D
            return TPoint2D(lambda s, p: (s.value - p.x, s.value - p.y), s=self, p=other)
        elif isinstance(other, Point2D):
            # Scalar - Point2D = Point2D
            return Point2D(self.value - other.x, self.value - other.y)
        # Matrix2D, TMatrix2D
        elif isinstance(other, TMatrix2D):
            # Scalar - TMatrix2D = TMatrix2D
            return TMatrix2D(lambda s, m: s.value - m.data, s=self, m=other)
        elif isinstance(other, Matrix2D):
            # Scalar - Matrix2D = Matrix2D
            return Matrix2D(self.value - other.data)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Scalar' and '{}'".format(type(other)))
        # end if
    # end __sub__

    def __rsub__(self, other):
        """
        Subtract the scalar value from another scalar or value.

        Args:
            other (any): Scalar or value to subtract
        """
        # Imports
        from .points import Point2D, TPoint2D
        from .matrices import Matrix2D, TMatrix2D

        # float, int
        if isinstance(other, float) or isinstance(other, int):
            # float - Scalar = Scalar
            return Scalar(other - self.value)
        # Scalar, TScalar
        elif isinstance(other, TScalar):
            # TScalar - Scalar = TScalar
            return TScalar(lambda s, o: o.value - s.value, s=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar - Scalar = Scalar
            return Scalar(other.value - self.value)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # TPoint2D - Scalar = TPoint2D
            return TPoint2D(lambda s, p: (p.x - s.value, p.y - s.value), s=self, p=other)
        elif isinstance(other, Point2D):
            # Point2D - Scalar = Point2D
            return Point2D(other.x - self.value, other.y - self.value)
        # Matrix2D, TMatrix2D
        elif isinstance(other, TMatrix2D):
            # TMatrix2D - Scalar = TMatrix2D
            return TMatrix2D(lambda s, m: m.data - s.value, s=self, m=other)
        elif isinstance(other, Matrix2D):
            # Matrix2D - Scalar = Matrix2D
            return Matrix2D(other.data - self.value)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Scalar' and '{}'".format(type(other)))
        # end if
    # end __rsub__

    def __mul__(self, other):
        """
        Multiply the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to multiply
        """
        # Imports
        from .points import Point2D, TPoint2D
        from .matrices import Matrix2D, TMatrix2D

        # float, int
        if isinstance(other, float) or isinstance(other, int):
            # Scalar * float = Scalar
            return Scalar(self.value * other)
        # Scalar, TScalar
        elif isinstance(other, TScalar):
            # Scalar * TScalar = TScalar
            return TScalar(lambda s, o: o.value - s.value, s=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar * Scalar = Scalar
            return Scalar(self.value * other.value)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Scalar * TPoint2D = TPoint2D
            return TPoint2D(lambda s, p: (s.value * p.x, s.value * p.y), s=self, p=other)
        elif isinstance(other, Point2D):
            # Scalar * Point2D = Point2D
            return Point2D(self.value * other.x, self.value * other.y)
        # Matrix2D, TMatrix2D
        elif isinstance(other, TMatrix2D):
            # Scalar * TMatrix2D = TMatrix2D
            return TMatrix2D(lambda s, m: s.value * m.data, s=self, m=other)
        elif isinstance(other, Matrix2D):
            # Scalar * Matrix2D = Matrix2D
            return Matrix2D(self.value * other.data)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Scalar' and '{}'".format(type(other)))
        # end if
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
        # Imports
        from .points import Point2D, TPoint2D
        from .matrices import Matrix2D, TMatrix2D

        # float, int
        if isinstance(other, float) or isinstance(other, int):
            # Scalar / float = Scalar
            return Scalar(self.value / other)
        # Scalar, TScalar
        elif isinstance(other, TScalar):
            # Scalar / TScalar = TScalar
            return TScalar(lambda s, o: s.value / o.value, s=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar / Scalar = Scalar
            return Scalar(self.value / other.value)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Scalar / TPoint2D = TPoint2D
            return TPoint2D(lambda s, p: (s.value / p.x, s.value / p.y), s=self, p=other)
        elif isinstance(other, Point2D):
            # Scalar / Point2D = Point2D
            return Point2D(self.value / other.x, self.value / other.y)
        # Matrix2D, TMatrix2D
        elif isinstance(other, TMatrix2D):
            # Scalar / TMatrix2D = TMatrix2D
            return TMatrix2D(lambda s, m: s.value / m.data, s=self, m=other)
        elif isinstance(other, Matrix2D):
            # Scalar / Matrix2D = Matrix2D
            return Matrix2D(self.value / other.data)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Scalar' and '{}'".format(type(other)))
        # end if
    # end __truediv__

    def __rtruediv__(self, other):
        """
        Divide the scalar value by another scalar or value.

        Args:
            other (any): Scalar or value to divide by
        """
        # Imports
        from .points import Point2D, TPoint2D
        from .matrices import Matrix2D, TMatrix2D

        # float, int
        if isinstance(other, float) or isinstance(other, int):
            # float / Scalar = Scalar
            return Scalar(other / self.value)
        # Scalar, TScalar
        elif isinstance(other, TScalar):
            # TScalar / Scalar = TScalar
            return TScalar(lambda s, o: o.value / s.value, s=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar / Scalar = Scalar
            return Scalar(other.value / self.value)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # TPoint2D / Scalar = TPoint2D
            return TPoint2D(lambda s, p: (p.x / s.value, p.y / s.value), s=self, p=other)
        elif isinstance(other, Point2D):
            # Point2D / Scalar = Point2D
            return Point2D(other.x / self.value, other.y / self.value)
        # Matrix2D, TMatrix2D
        elif isinstance(other, TMatrix2D):
            # TMatrix2D / Scalar = TMatrix2D
            return TMatrix2D(lambda s, m: m.data / s.value, s=self, m=other)
        elif isinstance(other, Matrix2D):
            # Matrix2D / Scalar = Matrix2D
            return Matrix2D(other.data / self.value)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Scalar' and '{}'".format(type(other)))
        # end if
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