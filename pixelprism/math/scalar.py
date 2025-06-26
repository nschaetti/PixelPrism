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


#
# This file contains the Scalar class, which is used to represent a scalar value.
#


# Imports
from abc import ABC
from typing import List, Dict, Union, Any
import numpy as np

from pixelprism.animate import RangeableMixin
from .math_expr import MathExpr, MathLeaf
import functional as F


# Type hint for addition
# Scalar + float = MathExpr
# Scalar + Scalar = Scalar
# Scalar + Point2D = Point2D
# Scalar + Matrix2D = Matrix2D

# Scalar class
class Scalar(MathLeaf, RangeableMixin, ABC):
    """
    A class to represent a scalar numerical value.

    The Scalar class is a fundamental building block in the PixelPrism library,
    representing a single numerical value that can be used in mathematical expressions.
    It inherits from MathLeaf, making it a terminal node in expression trees, and
    from RangeableMixin, allowing it to be animated over time.

    Scalar values can be combined with other mathematical expressions using standard
    arithmetic operators (+, -, *, /), compared using comparison operators
    (<, <=, >, >=, ==, !=), and converted to various numeric types (int, float).

    Example:
        ```python
        # Create scalar values
        a = Scalar(5)
        b = Scalar(10)

        # Arithmetic operations
        c = a + b       # c.value is 15
        d = a * 3       # d.value is 15
        e = b / a       # e.value is 2.0

        # Comparison operations
        is_greater = b > a  # True
        is_equal = a == 5   # True

        # Type conversion
        int_value = int(a)  # 5
        float_value = float(a)  # 5.0

        # Event handling
        def on_change(event_data):
            print(f"Value changed from {event_data.past_value} to {event_data.value}")

        f = Scalar(0, on_change=on_change)
        f.value = 42  # Prints: Value changed from 0 to 42

        # Animation (using RangeableMixin)
        a.animate(10, duration=1.0)  # Animates value from 5 to 10 over 1 second
        ```

    Attributes:
        expr_type (str): The type of expression ("Scalar").
        return_type (str): The type of value returned ("Scalar").
    """

    # Expression type - identifies this as a Scalar expression
    expr_type = "Scalar"

    # Return type - specifies that this expression returns a Scalar value
    return_type = 'Scalar'

    def __init__(
            self,
            value=0,
            on_change=None,
            readonly: bool = False
    ):
        """
        Initialize a new Scalar instance with the specified value.

        This constructor sets up the scalar with its initial value, configures
        event handling for value changes, and initializes the animation capabilities.

        Args:
            value (Union[int, float, Scalar]): Initial value of the scalar. Default is 0.
                If a Scalar instance is provided, its value will be extracted.
            on_change (Optional[Callable[[MathEventData], None]]): Function to call when the value changes.
                The function should accept a MathEventData parameter.
            readonly (bool): If True, the value cannot be changed after initialization.

        Example:
            ```python
            # Create a scalar with default value (0)
            s1 = Scalar()

            # Create a scalar with a specific value
            s2 = Scalar(42)

            # Create a scalar with a change listener
            def on_change(data):
                print(f"Value changed from {data.past_value} to {data.value}")

            s3 = Scalar(10, on_change=on_change)

            # Create a read-only scalar
            s4 = Scalar(100, readonly=True)
            ```
        """
        # Super
        super().__init__(
            value=value,
            on_change=on_change,
            readonly=readonly
        )

        # Initialize RangeableMixin (animation)
        RangeableMixin.__init__(self, "value")

        # Not a scalar in a scalar
        if isinstance(value, Scalar):
            value = value.get()
        # end if

        # Value
        self._value = value
    # end __init__

    # region PUBLIC

    def get(self) -> Union[int, float]:
        """
        Get the current value of this Scalar.

        This method is provided for backward compatibility with older code.
        In new code, it's recommended to use the `value` property instead.

        Returns:
            Union[int, float]: The current value of the scalar.

        Example:
            ```python
            # Get the value using the get() method
            s = Scalar(42)
            value = s.get()  # value is 42

            # Equivalent to:
            value = s.value
            ```
        """
        return self._value
    # end get

    def set(self, value: Union[int, float, 'Scalar']) -> None:
        """
        Set the value of this Scalar.

        This method is provided for backward compatibility with older code.
        In new code, it's recommended to use the `value` property instead.

        Args:
            value (Union[int, float, Scalar]): The new value to set.
                If a Scalar instance is provided, its value will be extracted.

        Example:
            ```python
            # Set the value using the set() method
            s = Scalar(42)
            s.set(100)  # s.value is now 100

            # Equivalent to:
            s.value = 100
            ```
        """
        self.value = value
    # end set

    def copy(self) -> 'Scalar':
        """
        Create a deep copy of this Scalar instance.

        This method creates a new Scalar instance with the same value as the current
        instance, but without sharing any mutable state. Changes to the copy will not
        affect the original, and vice versa.

        Returns:
            Scalar: A new Scalar instance with the same value as the current object.

        Example:
            ```python
            # Create a copy of a scalar
            original = Scalar(42)
            copy = original.copy()  # copy has value 42 but is a different object

            copy.value = 100  # Changes copy but not original
            print(original.value)  # Still 42
            ```
        """
        return Scalar(self._value)
    # end copy

    # endregion PUBLIC

    # region OVERRIDE

    # Override the integer conversion
    def __int__(self):
        """
        Convert this Scalar to an integer.

        This method allows Scalar instances to be used in contexts where an integer
        is expected, such as indexing or integer arithmetic.

        Returns:
            int: The integer representation of the scalar value.

        Example:
            ```python
            # Convert a scalar to an integer
            s = Scalar(3.14)
            i = int(s)  # i is 3

            # Use a scalar as an index
            lst = [10, 20, 30, 40, 50]
            index = Scalar(2)
            value = lst[int(index)]  # value is 30
            ```
        """
        return int(self._value)
    # end __int__

    # Override the float conversion
    def __float__(self):
        """
        Convert this Scalar to a float.

        This method allows Scalar instances to be used in contexts where a float
        is expected, such as floating-point arithmetic or mathematical functions.

        Returns:
            float: The floating-point representation of the scalar value.

        Example:
            ```python
            # Convert a scalar to a float
            s = Scalar(42)
            f = float(s)  # f is 42.0

            # Use a scalar with math functions
            import math
            angle = Scalar(math.pi/2)
            sine = math.sin(float(angle))  # sine is 1.0
            ```
        """
        return float(self._value)
    # end __float__

    def __str__(self):
        """
        Convert this Scalar to a string.

        This method is called when a string representation of the Scalar is needed,
        such as when using the str() function or print().

        Returns:
            str: A string representation of the scalar value.

        Example:
            ```python
            # Convert a scalar to a string
            s = Scalar(42)
            text = str(s)  # text is "42"

            # Print a scalar
            print(s)  # Prints: 42
            ```
        """
        return str(self._value)
    # end __str__

    def __repr__(self):
        """
        Return a string representation of this Scalar for debugging.

        This method is called when a developer-friendly representation of the Scalar
        is needed, such as in the Python interactive shell or debugger.

        Returns:
            str: A string representation that includes the class name and value.

        Example:
            ```python
            # Get the representation of a scalar
            s = Scalar(42)
            repr_str = repr(s)  # repr_str is "Scalar(value=42)"

            # In the Python interactive shell
            >>> s = Scalar(42)
            >>> s
            Scalar(value=42)
            ```
        """
        return f"Scalar(value={self._value})"
    # end __repr__

    # Operator overloading
    def __add__(self, other):
        """
        ...
        """
        return F.add(self, other)
    # end __add__

    def __radd__(self, other):
        """
        Add another value to this Scalar (reverse addition).
        """
        return F.add(other, self)
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
