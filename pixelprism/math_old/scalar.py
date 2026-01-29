# ####   #####  #   #  #####  #
# #   #    #     # #   #      #
# ####     #      #    #####  #
# #        #     # #   #      #
# #      #####  #   #  #####  #####
#
# ####   ####   #####   ####  #   #
# #   #  #   #    #    #      ## ##
# ####   ####     #     ###   # # #
# #      #  #     #        #  #   #
# #      #   #  #####  ####   #   #
#
# Copyright (C) 2024 Pixel Prism
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
            value,
            on_change=None,
            readonly: bool = False
    ):
        """
        Initialize a new Scalar instance with the specified value.

        This constructor sets up the scalar with its initial value, configures
        event handling for value changes, and initializes the animation capabilities.

        Args:
            value (Union[int, float, Scalar]): Initial value of the scalar.
                If a Scalar instance is provided, its value will be extracted.
            on_change (Optional[Callable[[MathEventData], None]]): Function to call when the value changes.
                The function should accept a MathEventData parameter.
            readonly (bool): If True, the value cannot be changed after initialization.

        Example:
            ```python
            # Create a scalar with a specific value
            s1 = Scalar(42)

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

            # Use a scalar with math_old functions
            import math_old
            angle = Scalar(math_old.pi/2)
            sine = math_old.sin(float(angle))  # sine is 1.0
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


# Transformable Scalar
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

        # Initialize the base class with t§he computed value
        initial_value = self._func(**self._scalars)
        super().__init__(initial_value)

        # Listen to sources
        for scalar in self._scalars.values():
            if hasattr(scalar, "on_change"):
                scalar.on_change.subscribe(self._on_source_changed)
            # end if
        # end for

        # Listen to changes in the original point
        self._on_change += on_change
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

    # region EVENTS

    def _on_source_changed(self, sender, event_type, **kwargs):
        """
        Called when the source scalar changes.
        """
        new_value = self.get()
        self._value = new_value
        self._trigger_on_change()
    # end _on_source_changed

    # endregion EVENTS

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
        return f"TScalar(func={self._func}, scalars={self._scalars})"
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

        # int, float,
        if isinstance(other, (Scalar, TScalar, float, int)):
            # TScalar + Scalar = TScalar
            # TScalar + TScalar = TScalar
            # TScalar + float = TScalar
            # TScalar + int = TScalar
            return TScalar.add(self, other)
        elif isinstance(other, (TPoint2D, Point2D)):
            # TScalar + TPoint2D = TPoint2D
            return TPoint2D(lambda s, p: (s.value + p.x, s.value + p.y), s=self, p=other)
        elif isinstance(other, (TMatrix2D, Matrix2D)):
            # TScalar + TMatrix2D = TMatrix2D
            return TMatrix2D(lambda s, m: m.data + s.value, s=self, m=other)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'TScalar' and '{}'".format(type(other)))
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
        """
        # Imports
        from .points import Point2D, TPoint2D
        from .matrices import Matrix2D, TMatrix2D

        # int, float,
        if isinstance(other, (Scalar, TScalar, float, int)):
            # TScalar - Scalar = TScalar
            # TScalar - TScalar = TScalar
            # TScalar - float = TScalar
            # TScalar - int = TScalar
            return TScalar.sub(self, other)
        elif isinstance(other, (TPoint2D, Point2D)):
            # TScalar - TPoint2D = TPoint2D
            return TPoint2D(lambda s, p: (s.value - p.x, s.value - p.y), s=self, p=other)
        elif isinstance(other, (TMatrix2D, Matrix2D)):
            # TScalar - TMatrix2D = TMatrix2D
            return TMatrix2D(lambda s, m: s.value - m.data, s=self, m=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'TScalar' and '{}'".format(type(other)))
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

        # int, float,
        if isinstance(other, (Scalar, TScalar, float, int)):
            # float - TScalar = TScalar
            # int - TScalar = TScalar
            # Scalar - TScalar = TScalar
            # TSclar - TScalar = TScalar
            return TScalar.sub(other, self)
        elif isinstance(other, (TPoint2D, Point2D)):
            # TPoint2D - TScalar = TPoint2D
            # Point2D - TScalar = TPoint2D
            return TPoint2D(lambda s, p: (p.x - s.value, p.y - s.value), s=self, p=other)
        elif isinstance(other, (TMatrix2D, Matrix2D)):
            # TScalar - TMatrix2D = TMatrix2D
            return TMatrix2D(lambda s, m: m.data - s.value, s=self, m=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'TScalar' and '{}'".format(type(other)))
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

        # int, float,
        if isinstance(other, (Scalar, TScalar, float, int)):
            # TScalar * Scalar = TScalar
            # TScalar * TScalar = TScalar
            # TScalar * float = TScalar
            # TScalar * int = TScalar
            return TScalar.mul(self, other)
        elif isinstance(other, (TPoint2D, Point2D)):
            # TScalar * TPoint2D = TPoint2D
            return TPoint2D(lambda s, p: (s.value * p.x, s.value * p.y), s=self, p=other)
        elif isinstance(other, (TMatrix2D, Matrix2D)):
            # TScalar * TMatrix2D = TMatrix2D
            return TMatrix2D(lambda s, m: s.value * m.data, s=self, m=other)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'TScalar' and '{}'".format(type(other)))
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

        # int, float,
        if isinstance(other, (Scalar, TScalar, float, int)):
            # TScalar / Scalar = TScalar
            # TScalar / TScalar = TScalar
            # TScalar / float = TScalar
            # TScalar / int = TScalar
            return TScalar.div(self, other)
        elif isinstance(other, (TPoint2D, Point2D)):
            # TScalar / TPoint2D = TPoint2D
            return TPoint2D(lambda s, p: (s.value / p.x, s.value / p.y), s=self, p=other)
        elif isinstance(other, (TMatrix2D, Matrix2D)):
            # TScalar / TMatrix2D = TMatrix2D
            return TMatrix2D(lambda s, m: s.value / m.data, s=self, m=other)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'TScalar' and '{}'".format(type(other)))
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

        # int, float,
        if isinstance(other, (Scalar, TScalar, float, int)):
            # float / TScalar = TScalar
            # int / TScalar = TScalar
            # Scalar / TScalar = TScalar
            # TSclar / TScalar = TScalar
            return TScalar.div(other, self)
        elif isinstance(other, (TPoint2D, Point2D)):
            # TPoint2D / TScalar = TPoint2D
            # Point2D / TScalar = TPoint2D
            return TPoint2D(lambda s, p: (p.x / s.value, p.y / s.value), s=self, p=other)
        elif isinstance(other, (TMatrix2D, Matrix2D)):
            # TScalar / TMatrix2D = TMatrix2D
            return TMatrix2D(lambda s, m: m.data / s.value, s=self, m=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'TScalar' and '{}'".format(type(other)))
        # end if
    # end __rtruediv__

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

    # region CONSTRUCTORS

    # Basic TScalar (just return value of scalar)
    @classmethod
    def tscalar(cls, scalar: Union[Scalar, float, int]):
        """
        Create a TScalar that returns the value of a scalar.

        Args:
            scalar (Scalar): The scalar to return the value of.
        """
        if isinstance(scalar, float) or isinstance(scalar, int):
            scalar = Scalar(scalar)
        # end if
        return cls(lambda s: s.value, s=scalar)
    # end tscalar

    # endregion CONSTRUCTORS

    # region OPERATORS

    # Add to scalar
    @classmethod
    def add(cls, scalar1: Union[Scalar, float, int], scalar2: Union[Scalar, float, int]):
        """
        Create a TScalar that adds two scalar values.

        Args:
            scalar1 (Union[Scalar, float, int]): First scalar to add.
            scalar2 (Union[Scalar, float, int]): Second scalar to add.
        """
        if isinstance(scalar1, float) or isinstance(scalar1, int):
            scalar1 = Scalar(scalar1)
        # end if

        if isinstance(scalar2, float) or isinstance(scalar2, int):
            scalar2 = Scalar(scalar2)
        # end if

        return cls(lambda s1, s2: s1.value + s2.value, s1=scalar1, s2=scalar2)
    # end add

    @classmethod
    def sub(cls, scalar1: Union[Scalar, float], scalar2: Union[Scalar, float]):
        """
        Create a TScalar that subtracts two scalar values.

        Args:
            scalar1 (Union[Scalar, float]): First scalar to subtract.
            scalar2 (Union[Scalar, float]): Second scalar to subtract.
        """
        if isinstance(scalar1, float):
            scalar1 = Scalar(scalar1)
        # end if

        if isinstance(scalar2, float):
            scalar2 = Scalar(scalar2)
        # end if

        return cls(lambda s1, s2: s1.value - s2.value, s1=scalar1, s2=scalar2)
    # end sub

    @classmethod
    def mul(cls, scalar1: Union[Scalar, float], scalar2: Union[Scalar, float]):
        """
        Create a TScalar that multiplies two scalar values.

        Args:
            scalar1 (Union[Scalar, float]): First scalar to multiply.
            scalar2 (Union[Scalar, float]): Second scalar to multiply.
        """
        if isinstance(scalar1, float):
            scalar1 = Scalar(scalar1)
        # end if

        if isinstance(scalar2, float):
            scalar2 = Scalar(scalar2)
        # end if

        return cls(lambda s1, s2: s1.value * s2.value, s1=scalar1, s2=scalar2)
    # end mul

    @classmethod
    def div(cls, scalar1: Union[Scalar, float, int], scalar2: Union[Scalar, float, int]):
        """
        Create a TScalar that divides two scalar values.

        Args:
            scalar1 (Union[Scalar, float, int]): First scalar to divide.
            scalar2 (Union[Scalar, float, int]): Second scalar to divide.
        """
        if isinstance(scalar1, float):
            scalar1 = Scalar(scalar1)
        # end if

        if isinstance(scalar2, float):
            scalar2 = Scalar(scalar2)
        # end if

        return cls(lambda s1, s2: s1.value / s2.value, s1=scalar1, s2=scalar2)
    # end div

    # endregion OPERATORS

    # region MATH

    @classmethod
    def floor(cls, scalar: Scalar):
        """
        Create a TScalar that applies the floor function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the floor function to.
        """
        return cls(lambda s: np.floor(s.value), s=scalar)
    # end floor

    @classmethod
    def ceil(cls, scalar: Scalar):
        """
        Create a TScalar that applies the ceil function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the ceil function to.
        """
        return cls(lambda s: np.ceil(s.value), s=scalar)
    # end ceil

    @classmethod
    def trunc(cls, scalar: Scalar):
        """
        Create a TScalar that applies the trunc function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the trunc function to.
        """
        return cls(lambda s: np.trunc(s.value), s=scalar)
    # end trunc

    @classmethod
    def frac(cls, scalar: Scalar):
        """
        Create a TScalar that returns the fractional part of the scalar.

        Args:
            scalar (Scalar): The scalar to get the fractional part of.
        """
        return cls(lambda s: s.value - np.floor(s.value), s=scalar)
    # end frac

    @classmethod
    def sqrt(cls, scalar: Scalar):
        """
        Create a TScalar that applies the sqrt function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the sqrt function to.
        """
        return cls(lambda s: np.sqrt(s.value), s=scalar)
    # end sqrt

    @classmethod
    def exp(cls, scalar: Scalar):
        """
        Create a TScalar that applies the exp function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the exp function to.
        """
        return cls(lambda s: np.exp(s.value), s=scalar)
    # end exp

    @classmethod
    def expm1(cls, scalar: Scalar):
        """
        Create a TScalar that applies the expm1 function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the expm1 function to.
        """
        return cls(lambda s: np.expm1(s.value), s=scalar)
    # end expm1

    @classmethod
    def log(cls, scalar: Scalar):
        """
        Create a TScalar that applies the log function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the log function to.
        """
        return cls(lambda s: np.log(s.value), s=scalar)
    # end log

    @classmethod
    def log1p(cls, scalar: Scalar):
        """
        Create a TScalar that applies the log1p function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the log1p function to.
        """
        return cls(lambda s: np.log1p(s.value), s=scalar)
    # end log1p

    @classmethod
    def log2(cls, scalar: Scalar):
        """
        Create a TScalar that applies the log2 function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the log2 function to.
        """
        return cls(lambda s: np.log2(s.value), s=scalar)
    # end log2

    @classmethod
    def log10(cls, scalar: Scalar):
        """
        Create a TScalar that applies the log10 function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the log10 function to.
        """
        return cls(lambda s: np.log10(s.value), s=scalar)
    # end log10

    @classmethod
    def sin(cls, scalar: Scalar):
        """
        Create a TScalar that applies the sin function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the sin function to.
        """
        return cls(lambda s: np.sin(s.value), s=scalar)
    # end sin

    @classmethod
    def cos(cls, scalar: Scalar):
        """
        Create a TScalar that applies the cos function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the cos function to.
        """
        return cls(lambda s: np.cos(s.value), s=scalar)
    # end cos

    @classmethod
    def tan(cls, scalar: Scalar):
        """
        Create a TScalar that applies the tan function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the tan function to.
        """
        return cls(lambda s: np.tan(s.value), s=scalar)
    # end tan

    @classmethod
    def asin(cls, scalar: Scalar):
        """
        Create a TScalar that applies the asin function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the asin function
        """
        return cls(lambda s: np.arcsin(s.value), s=scalar)
    # end asin

    @classmethod
    def acos(cls, scalar: Scalar):
        """
        Create a TScalar that applies the acos function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the acos function
        """
        return cls(lambda s: np.arccos(s.value), s=scalar)
    # end acos

    @classmethod
    def atan(cls, scalar: Scalar):
        """
        Create a TScalar that applies the atan function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the atan function
        """
        return cls(lambda s: np.arctan(s.value), s=scalar)
    # end atan

    @classmethod
    def atan2(cls, y: Scalar, x: Scalar):
        """
        Create a TScalar that applies the atan2 function to the scalar.

        Args:
            y (Scalar): The y-coordinate of the point.
            x (Scalar): The x-coordinate of the point.
        """
        return cls(lambda p1, p2: np.arctan2(p1.value, p2.value), p1=y, p2=x)
    # end atan2

    @classmethod
    def sinh(cls, scalar: Scalar):
        """
        Create a TScalar that applies the sinh function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the sinh function
        """
        return cls(lambda s: np.sinh(s.value), s=scalar)
    # end sinh

    @classmethod
    def cosh(cls, scalar: Scalar):
        """
        Create a TScalar that applies the cosh function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the cosh function
        """
        return cls(lambda s: np.cosh(s.value), s=scalar)
    # end cosh

    @classmethod
    def tanh(cls, scalar: Scalar):
        """
        Create a TScalar that applies the tanh function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the tanh function
        """
        return cls(lambda s: np.tanh(s.value), s=scalar)
    # end tanh

    @classmethod
    def asinh(cls, scalar: Scalar):
        """
        Create a TScalar that applies the asinh function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the asinh function
        """
        return cls(lambda s: np.arcsinh(s.value), s=scalar)
    # end asinh

    @classmethod
    def acosh(cls, scalar: Scalar):
        """
        Create a TScalar that applies the acosh function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the acosh function
        """
        return cls(lambda s: np.arccosh(s.value), s=scalar)
    # end acosh

    @classmethod
    def atanh(cls, scalar: Scalar):
        """
        Create a TScalar that applies the atanh function to the scalar.

        Args:
            scalar (Scalar): The scalar to apply the atanh function
        """
        return cls(lambda s: np.arctanh(s.value), s=scalar)
    # end atanh

    @classmethod
    def degrees(cls, scalar: Scalar):
        """
        Create a TScalar that converts the scalar value from radians to degrees.

        Args:
            scalar (Scalar): The scalar
        """
        return cls(lambda s: np.degrees(s.value), s=scalar)
    # end degrees

    # endregion MATH

    # region GENERATION

    @classmethod
    def scalar_range(cls, start, stop=None, step=1, return_tscalar: bool = False):
        """
        Create a list of Scalars using the built-in range function.

        Args:
            start (int): Start value.
            stop (int, optional): Stop value.
            step (int, optional): Step value.
            return_tscalar (bool, optional): If True, return a list of TScalars.

        Returns:
            List[Scalar]: List of Scalars or TScalars.
        """
        if return_tscalar:
            scalars = [Scalar(i) for i in range(start, stop, step)]
            return [cls(lambda s: s.value, s=scalar) for scalar in scalars]
        else:
            return [Scalar(i) for i in range(start, stop, step)]
        # end if
    # end scalar_range

    @classmethod
    def linspace(cls, start, stop, num=50, return_tscalar: bool = False):
        """
        Create a list of TScalars using numpy's linspace.

        Args:
            start (float): Start value.
            stop (float): Stop value.
            num (int, optional): Number of samples to generate.
            return_tscalar (bool, optional): If True, return a list of TScalars.

        Returns:
            List[TScalar]: List of TScalars.
        """
        if return_tscalar:
            scalars = [Scalar(i) for i in np.linspace(start, stop, num)]
            return [cls(lambda s: s.value, s=scalar) for scalar in scalars]
        else:
            return [Scalar(s) for s in np.linspace(start, stop, num)]
        # end if
    # end linspace

    @classmethod
    def logspace(cls, start, stop, num=50, base=10.0, return_tscalar: bool = False):
        """
        Create a list of TScalars using numpy's logspace.

        Args:
            start (float): Start value.
            stop (float): Stop value.
            num (int, optional): Number of samples to generate.
            base (float, optional): Base of the logarithm.
            return_tscalar (bool, optional): If True, return a list of TScalars.

        Returns:
            List[TScalar]: List of TScalars.
        """
        if return_tscalar:
            scalars = [Scalar(i) for i in np.logspace(start, stop, num=num, base=base)]
            return [cls(lambda s: s.value, s=scalar) for scalar in scalars]
        else:
            return [Scalar(s) for s in np.logspace(start, stop, num=num, base=base)]
        # end if
    # end logspace

    @classmethod
    def uniform(cls, low=0.0, high=1.0, size=None, return_tscalar: bool = False):
        """
        Create a TScalar or a list of TScalars with uniform distribution.

        Args:
            low (float): Lower bound of the uniform distribution.
            high (float): Upper bound of the uniform distribution.
            size (int, optional): Number of samples to generate.
            return_tscalar (bool, optional): If True, return a list of TScalars.

        Returns:
            TScalar or List[TScalar]: TScalar or list of TScalars.
        """
        if size is None and return_tscalar:
            scalar = Scalar(np.random.uniform(low, high))
            return cls(lambda s: s.value, s=scalar)
        elif size is None:
            return Scalar(np.random.uniform(low, high))
        elif return_tscalar:
            scalars = [Scalar(np.random.uniform(low, high)) for _ in range(size)]
            return [cls(lambda s: s.value, s=scal) for scal in scalars]
        else:
            return [Scalar(np.random.uniform(low, high)) for _ in range(size)]
        # end if
    # end uniform

    @classmethod
    def normal(cls, loc=0.0, scale=1.0, size=None, return_tscalar: bool = False):
        """
        Create a TScalar or a list of TScalars with normal distribution.

        Args:
            loc (float): Mean of the distribution.
            scale (float): Standard deviation of the distribution.
            size (int, optional): Number of samples to generate.
            return_tscalar (bool, optional): If True, return a list of TScalars.

        Returns:
            TScalar or List[TScalar]: TScalar or list of TScalars.
        """
        if size is None and return_tscalar:
            scalar = Scalar(np.random.normal(loc, scale))
            return cls(lambda s: s.value, s=scalar)
        elif size is None:
            return Scalar(np.random.normal(loc, scale))
        elif return_tscalar:
            scalars = [Scalar(np.random.normal(loc, scale)) for _ in range(size)]
            return [cls(lambda s: s.value, s=scal) for scal in scalars]
        else:
            return [Scalar(np.random.normal(loc, scale)) for _ in range(size)]
        # end if
    # end normal

    @classmethod
    def poisson(cls, lam=1.0, size=None, return_tscalar: bool = False):
        """
        Create a TScalar or a list of TScalars with Poisson distribution.

        Args:
            lam (float): Expected number of events (lambda).
            size (int, optional): Number of samples to generate.
            return_tscalar (bool, optional): If True, return a list of TScalars.

        Returns:
            TScalar or List[TScalar]: TScalar or list of TScalars.
        """
        if size is None and return_tscalar:
            scalar = Scalar(np.random.poisson(lam))
            return cls(lambda s: s.value, s=scalar)
        elif size is None:
            return Scalar(np.random.poisson(lam))
        elif return_tscalar:
            scalars = [Scalar(np.random.poisson(lam)) for _ in range(size)]
            return [cls(lambda s: s.value, s=scal) for scal in scalars]
        else:
            return [Scalar(np.random.poisson(lam)) for _ in range(size)]
        # end if
    # end poisson

    @classmethod
    def randint(cls, low, high=None, size=None, return_tscalar: bool = False):
        """
        Create a Scalar or a list of Scalars with random integers from low (inclusive) to high (exclusive).

        Args:
            low (int): Lower bound of the range (inclusive).
            high (int, optional): Upper bound of the range (exclusive). If None, range is from 0 to low.
            size (int, optional): Number of samples to generate.
            return_tscalar (bool, optional): If True, return a list of TScalars.

        Returns:
            Scalar or List[Scalar]: Scalar or list of Scalars or TScalars.
        """
        if high is None:
            high = low
            low = 0
        # end if

        if size is None and return_tscalar:
            scalar = Scalar(np.random.randint(low, high))
            return cls(lambda s: s.value, s=scalar)
        elif size is None:
            return Scalar(np.random.randint(low, high))
        elif return_tscalar:
            scalars = [Scalar(np.random.randint(low, high)) for _ in range(size)]
            return [cls(lambda s: s.value, s=scal) for scal in scalars]
        else:
            return [Scalar(np.random.randint(low, high)) for _ in range(size)]
        # end if
    # end randint

    @classmethod
    def choice(cls, a, size=None, replace=True, return_tscalar: bool = False):
        """
        Create a Scalar or a list of Scalars with random choices from a given array.

        Args:
            a (array-like): Array to choose from.
            size (int, optional): Number of samples to generate.
            replace (bool, optional): Whether to sample with replacement.
            return_tscalar (bool, optional): If True, return a list of TScalars.

        Returns:
            Scalar or List[Scalar]: Scalar or list of Scalars or TScalars.
        """
        if size is None and return_tscalar:
            scalar = Scalar(np.random.choice(a, replace=replace))
            return cls(lambda s: s.value, s=scalar)
        elif size is None:
            return Scalar(np.random.choice(a, replace=replace))
        elif return_tscalar:
            scalars = [Scalar(np.random.choice(a, replace=replace)) for _ in range(size)]
            return [cls(lambda s: s.value, s=scal) for scal in scalars]
        else:
            return [Scalar(np.random.choice(a, replace=replace)) for _ in range(size)]
        # end if
    # end choice

    @classmethod
    def shuffle(cls, x, return_tscalar: bool = False):
        """
        Shuffle a sequence in place and return it as a list of Scalars.

        Args:
            x (array-like): Array to shuffle.
            return_tscalar (bool, optional): If True, return a list of TScalars.

        Returns:
            List[Scalar]: List of Scalars or TScalars.
        """
        np.random.shuffle(x)
        if return_tscalar:
            scalars = [Scalar(s) for s in x]
            return [cls(lambda s: s.value, s=scalar) for scalar in scalars]
        else:
            return [Scalar(s) for s in x]
        # end if

    # end shuffle

    @classmethod
    def scalar_arange(cls, start, stop=None, step=1, return_tscalar: bool = False):
        """
        Create a list of Scalars using numpy's arange function.

        Args:
            start (int or float): Start value.
            stop (int or float, optional): Stop value.
            step (int or float, optional): Step value.
            return_tscalar (bool, optional): If True, return a list of TScalars.

        Returns:
            List[Scalar]: List of Scalars or TScalars.
        """
        if return_tscalar:
            scalars = [Scalar(s) for s in np.arange(start, stop, step)]
            return [cls(lambda s: s.value, s=scalar) for scalar in scalars]
        else:
            return [Scalar(s) for s in np.arange(start, stop, step)]
        # end if
    # end arange

    # endregion GENERATION

# end TScalar

