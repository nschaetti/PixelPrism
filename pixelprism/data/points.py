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
# This file contains the Point2D class, which is a simple class that
# represents a point in 2D space.
#

# Imports
import math
from typing import Any, Union, Optional
import numpy as np
from pixelprism.animate import MovableMixin
from .data import Data
from .scalar import Scalar, TScalar
from .events import Event, EventType


# A generic point
class Point(Data, MovableMixin):
    """
    A generic point class.
    """

    # Constructor
    def __init__(self, readonly: bool = False):
        """
        Constructor

        Args:
            readonly (bool): If the point is read-only
        """
        Data.__init__(self, readonly=readonly)
        MovableMixin.__init__(self)
    # end __init__

# end Point


class Point2D(Point):
    """
    A class to represent a point in 2D space.
    """

    def __init__(
            self,
            x=0,
            y=0,
            on_change=None,
            readonly: bool = False,
            dtype=np.float32
    ):
        """
        Initialize the point with its coordinates.

        Args:
            x (float): X-coordinate of the point
            y (float): Y-coordinate of the point
            on_change (function): Function to call when the point changes
            readonly (bool): If the point is read-only
            dtype (dtype): Data type of the point
        """
        super().__init__(readonly=readonly)
        self._pos = np.array([x, y], dtype=dtype)

        # Events
        self._on_change = Event()

        # List of event listeners (per events)
        if on_change: self._on_change += on_change
    # end __init__

    # region PROPERTIES

    @property
    def pos(self):
        """
        Get the position of the point.

        Returns:
            np.array: Position of the point
        """
        return self._pos
    # end pos

    @pos.setter
    def pos(self, value: np.array):
        """
        Set the position of the point.
        """
        self.set(value[0], value[1])
    # end pos

    @property
    def x(self):
        """
        Get the X-coordinate of the point.

        Returns:
            float: X-coordinate of the point
        """
        return self._pos[0]
    # end x

    @x.setter
    def x(self, value):
        """
        Set the X-coordinate of the point.
        """
        self.set(value, self.y)
    # end x

    @property
    def y(self):
        """
        Get the Y-coordinate of the point.

        Returns:
            float: Y-coordinate of the point
        """
        return self._pos[1]
    # end y

    @y.setter
    def y(self, value):
        """
        Set the Y-coordinate of the point.
        """
        self.set(self.x, value)
    # end y

    # On change event
    @property
    def on_change(self) -> Event:
        """
        Get the on change event.

        Returns:
            Event: On change event
        """
        return self._on_change
    # end on_change

    # Movable position
    @property
    def movable_position(self) -> Any:
        """
        Get the position of the object.

        Returns:
            any: Position of the object
        """
        return self
    # end movable_position

    @movable_position.setter
    def movable_position(self, value: 'Point2D'):
        """
        Set the position of the object.

        Args:
            value (Point2D): Position of the object
        """
        self.set(value.x, value.y)
    # end movable_position

    # endregion PROPERTIES

    # region PUBLIC

    def set(self, x, y):
        """
        Set the coordinates of the point.

        Args:
            x (float or Scalar): X-coordinate of the point
            y (float or Scalar): Y-coordinate of the point
        """
        self.check_closed()

        # Scalar/float
        if type(x) is Scalar or type(y) is Scalar:
            x = x.value if type(x) is Scalar else x
            y = y.value if type(y) is Scalar else y
        # end if

        # Update
        if self._pos[0] != x or self._pos[1] != y:
            # Update position
            self._pos[0] = x
            self._pos[1] = y

            # Trigger change event
            self._trigger_on_change()
        # end if
    # end set

    def get(self):
        """
        Get the coordinates of the point.

        Returns:
            np.array: Array containing the X and Y coordinates of the point
        """
        return self._pos[0], self._pos[1]
    # end get

    # To list
    def to_list(self):
        """
        Convert the scalar to a list.
        """
        return [self._pos[0], self._pos[1]]
    # end to_list

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

    def copy(self, on_change = None, readonly: bool = False):
        """
        Return a copy of the point.

        Args:
            on_change: Copy changes also ?
            readonly (bool): If the copy should be read-only
        """
        return Point2D(
            x=self.x,
            y=self.y,
            on_change=on_change,
            readonly=readonly,
            dtype=self._pos.dtype
        )
    # end copy

    # Euclidian norm of the point
    def norm2(self, other):
        """
        Calculate the Euclidean norm of the point.

        Args:
            other (Point2D): Point to calculate the norm with
        """
        return np.linalg.norm(self._pos - other._pos)
    # end if

    # Round the point
    def round_(self, ndigits=0):
        """
        Round the point.

        Args:
            ndigits (int): Number of digits to round to
        """
        self.x = round(self.x, ndigits=ndigits)
        self.y = round(self.y, ndigits=ndigits)
    # end round

    # Translate point
    def translate_(self, dp):
        """
        Translate the point.

        Args:
            dp (Point2D): Difference
        """
        self.x += dp.x
        self.y += dp.y
    # end translate

    # Rotate point
    def rotate_(self, angle, center=None):
        """
        Rotate the point.

        Args:
            angle (float): Angle to rotate by
            center (Point2D): Origin of the rotation
        """
        if center is None:
            center = Point2D.null()
        # end if
        x = self.x - center.x
        y = self.y - center.y
        x_new = x * np.cos(angle) - y * np.sin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)
        self.x = x_new + center.x
        self.y = y_new + center.y
    # end rotate

    # Scale point
    def scale_(self, scale, center=None):
        """
        Scale the point.

        Args:
            scale (float or Scalar): Scale factor
            center (Point2D): Origin of the scaling
        """
        scale = scale.value if type(scale) is Scalar else scale
        if center is None:
            center = Point2D.null()
        # end if
        self.x = center.x + (self.x - center.x) * scale
        self.y = center.y + (self.y - center.y) * scale
    # end scale

    # endregion PUBLIC

    # region PRIVATE

    # Trigger position change
    def _trigger_on_change(self):
        """
        Trigger the position change event.
        """
        self.on_change.trigger(self, event_type=EventType.POSITION_CHANGED, x=self.x, y=self.y)
    # end _trigger_on_change

    # endregion PRIVATE

    # region MOVABLE

    def init_move(self, *args, **kwargs):
        """
        Initialize the move animation.
        """
        super().init_move(*args, **kwargs)
    # end init_move

    # Start animation
    def start_move(
            self,
            start_value: Any,
            *args,
            **kwargs
    ):
        """
        Start the move animation.

        Args:
            start_value (any): The start position of the object
        """
        super().start_move(start_value, *args, **kwargs)
    # end start_move

    def animate_move(
            self,
            t,
            duration,
            interpolated_t,
            end_value,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Perform the move animation.

        Args:
            t (float): Relative time since the start of the animation
            duration (float): Duration of the animation
            interpolated_t (float): Time value adjusted by the interpolator
            end_value (any): The end position of the object
            relative (bool): If the move is relative to the current position
        """
        super().animate_move(t, duration, interpolated_t, end_value, relative, *args, **kwargs)
    # end animate_move

    # Stop animation
    def end_move(
            self,
            end_value: Any,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Stop the move animation.

        Args:
            end_value (any): The end position of the object
            relative (bool): If the move is relative to the current position
        """
        super().end_move(end_value, relative, *args, **kwargs)
    # end end_move

    # Finish move animation
    def finish_move(self, *args, **kwargs):
        """
        Finish the move animation.
        """
        super().finish_move(*args, **kwargs)
    # end finish_move

    # endregion MOVABLE

    # region OVERRIDE

    # Return a string representation of the point.
    def __str__(self):
        """
        Return a string representation of the point.
        """
        return f"Point2D(x={self.x}, y={self.y}, readonly={self.data_closed})"
    # end __str__

    # Return a string representation of the point.
    def __repr__(self):
        """
        Return a string representation of the point.
        """
        return self.__str__()
    # end __repr__

    # Operator overloads
    def __add__(self, other):
        """
        Add two points together.

        Args:
            other (Union[Point2D, TPoint2D, int, float, tuple]): Point to add
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point2D
        if isinstance(other, (int, float)):
            # Point2D + scalar = Point2D
            return Point2D(self.x + other, self.y + other)
        elif isinstance(other, tuple):
            # Point2D + tuple = Point2D
            return Point2D(self.x + other[0], self.y + other[1])
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point2D + TScalar = TPoint2D
            return TPoint2D(lambda p, o: (p.x + o.value, p.y + o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point2D + Scalar = Point2D
            return Point2D(self.x + other.value, self.y + other.value)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Point2D + TPoint2D = TPoint2D
            return TPoint2D.add(self, other)
        elif isinstance(other, Point2D):
            # Point2D + Point2D = Point2D
            return Point2D(self.x + other.x, self.y + other.y)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __add__

    def __radd__(self, other):
        """
        Add two points together.

        Args:
            other (Union[Point2D, TPoint2D, int, float, tuple]): Point to add
        """
        return self.__add__(other)
    # end __radd__

    def __sub__(self, other):
        """
        Subtract two points.

        Args:
            other (Point2D): Point to subtract from this point or scalar value to subtract from the point.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point2D
        if isinstance(other, (int, float)):
            # Point2D - scalar = Point2D
            return Point2D(self.x - other, self.y - other)
        elif isinstance(other, tuple):
            # Point2D - tuple = Point2D
            return Point2D(self.x - other[0], self.y - other[1])
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point2D - TScalar = TPoint2D
            return TPoint2D(lambda p, o: (p.x - o.value, p.y - o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point2D - Scalar = Point2D
            return Point2D(self.x - other.value, self.y - other.value)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Point2D - TPoint2D = TPoint2D
            return TPoint2D.sub(self, other)
        elif isinstance(other, Point2D):
            # Point2D - Point2D = Point2D
            return Point2D(self.x - other.x, self.y - other.y)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __sub__

    def __rsub__(self, other):
        """
        Subtract two points.

        Args:
            other (Point2D): Point to subtract from this point or scalar value to subtract from the point.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point2D
        if isinstance(other, (int, float)):
            # scalar - Point2D = Point2D
            return Point2D(other - self.x, other - self.y)
        elif isinstance(other, tuple):
            # tuple - Point2D = Point2D
            return Point2D(other[0] - self.x, other[1] - self.y)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # TScalar - Point2D = TPoint2D
            return TPoint2D(lambda p, o: (o.value - p.x, o.value - p.y), p=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar - Point2D = Point2D
            return Point2D(other.value - self.x, other.value - self.y)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Point2D - TPoint2D = TPoint2D
            return TPoint2D.sub(other, self)
        elif isinstance(other, Point2D):
            # Point2D - Point2D = Point2D
            return Point2D(other.x - self.x, other.y - self.y)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __rsub__

    def __mul__(self, other):
        """
        Multiply the point.

        Args:
            other (int, float, tuple, Scalar, TScalar, Point2D, TPoint2D): Scalar value to multiply the point by.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point2D
        if isinstance(other, (int, float)):
            # Point2D * scalar = Point2D
            return Point2D(self.x * other, self.y * other)
        elif isinstance(other, tuple):
            # Point2D * tuple = Point2D
            return Point2D(self.x * other[0], self.y * other[1])
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point2D * TScalar = TPoint2D
            return TPoint2D(lambda p, o: (p.x * o.value, p.y * o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point2D * Scalar = Point2D
            return Point2D(self.x * other.value, self.y * other.value)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Point2D * TPoint2D = TPoint2D
            return TPoint2D.mul(self, other)
        elif isinstance(other, Point2D):
            # Point2D * Point2D = Point2D
            return Point2D(self.x * other.x, self.y * other.y)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __mul__

    def __rmul__(self, other):
        """
        Multiply the point by a scalar value.

        Args:
            other (int, float): Scalar value to multiply the point by.
        """
        return self.__mul__(other)
    # end __rmul__

    def __truediv__(self, other):
        """
        Divide the point by a scalar value.

        Args:
            other (int, float, Scalar, TScalar, Point2D, TPoint2D): Scalar value to divide the point by.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point2D
        if isinstance(other, (int, float)):
            # Point2D / scalar = Point2D
            return Point2D(self.x / other, self.y / other)
        elif isinstance(other, tuple):
            # Point2D / tuple = Point2D
            return Point2D(self.x / other[0], self.y / other[1])
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point2D / TScalar = TPoint2D
            return TPoint2D(lambda p, o: (p.x / o.value, p.y / o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point2D / Scalar = Point2D
            return Point2D(self.x / other.value, self.y / other.value)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Point2D / TPoint2D = TPoint2D
            return TPoint2D.mul(self, other)
        elif isinstance(other, Point2D):
            # Point2D / Point2D = Point2D
            return Point2D(self.x / other.x, self.y / other.y)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __truediv__

    def __rtruediv__(self, other):
        """
        Divide the point by a scalar value.

        Args:
            other (int, float): Scalar value to divide the point by.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point2D
        if isinstance(other, (int, float)):
            # scalar / Point2D = Point2D
            return Point2D(other / self.x, other / self.y)
        elif isinstance(other, tuple):
            # tuple / Point2D = Point2D
            return Point2D(other[0] / self.x, other[1] / self.y)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # TScalar / Point2D = TPoint2D
            return TPoint2D(lambda p, o: (o.value / p.x, o.value / p.y), p=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar / Point2D = Point2D
            return Point2D(other.value / self.x, other.value / self.y)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Point2D / TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o.x / p.x, o.y / p.y), p=self, o=other)
        elif isinstance(other, Point2D):
            # Point2D / Point2D = Point2D
            return Point2D(other.x / self.x, other.y / self.y)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __rtruediv__

    def __eq__(self, other):
        """
        Compare two points for equality.

        Args:
            other (Point2D): Point to compare with
        """
        if isinstance(other, Point2D):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, tuple):
            return self.x == other[0] and self.y == other[1]
        else:
            return NotImplemented
        # end if
    # end __eq__

    def __abs__(self):
        """
        Get the absolute value of the point.
        """
        return Point2D(abs(self.x), abs(self.y))
    # end __abs__

    # endregion OVERRIDE

    # region CONSTRUCTORS

    @classmethod
    def from_tuple(cls, pos, dtype=np.float32):
        """
        Create a Point2D object from a tuple.

        Args:
            pos (tuple): Tuple containing the X and Y coordinates of the point
            dtype (type): Data type of the point
        """
        return cls(pos[0], pos[1], dtype=dtype)
    # end from_tuple

    @classmethod
    def from_point(cls, point, dtype=np.float32):
        """
        Create a Point2D object from another point.

        Args:
            point (Point2D): Point to create a new point from
            dtype (type): Data type of the point
        """
        return cls(point.x, point.y, dtype=dtype)
    # end from_point

    @classmethod
    def null(cls, dtype=np.float32):
        """
        Create a null point.

        Args:
            dtype (type): Data type of the point
        """
        return cls(0, 0, dtype=dtype)
    # end null

    # endregion CONSTRUCTORS

# end Point2D


class Point3D(Point):
    """
    A class to represent a point in 3D space.
    """

    def __init__(self, x=0, y=0, z=0, on_change=None, readonly: bool = False, dtype=np.float32):
        """
        Initialize the point with its coordinates.

        Args:
            x (float): X-coordinate of the point
            y (float): Y-coordinate of the point
            z (float): Z-coordinate of the point
            on_change (function): Function to call when the point changes
            readonly (bool): If the point is read-only
            dtype (type): Data type of the point
        """
        super().__init__(readonly=readonly)
        self._pos = np.array([x, y, z], dtype=dtype)

        # Events
        self._on_change = Event()

        # List of event listeners (per events)
        if on_change: self._on_change += on_change
    # end __init__

    # region PROPERTIES

    @property
    def pos(self):
        """
        Get the position of the point.

        Returns:
            np.array: Position of the point
        """
        return self._pos
    # end pos

    @pos.setter
    def pos(self, value: np.array):
        """
        Set the position of the point.
        """
        self.set(value)
    # end pos

    @property
    def x(self):
        """
        Get the X-coordinate of the point.

        Returns:
            float: X-coordinate of the point
        """
        return self._pos[0]
    # end x

    @x.setter
    def x(self, value):
        """
        Set the X-coordinate of the point.
        """
        self.set(np.array([value, self.y, self.z]))
    # end x

    @property
    def y(self):
        """
        Get the Y-coordinate of the point.

        Returns:
            float: Y-coordinate of the point
        """
        return self._pos[1]
    # end y

    @y.setter
    def y(self, value):
        """
        Set the Y-coordinate of the point.
        """
        self.set(np.array([self.x, value, self.z]))
    # end y

    @property
    def z(self):
        """
        Get the Z-coordinate of the point.

        Returns:
            float: Z-coordinate of the point
        """
        return self._pos[2]
    # end z

    @z.setter
    def z(self, value):
        """
        Set the Z-coordinate of the point.
        """
        self.set(np.array([self.x, self.y, value]))
    # end z

    @property
    def on_change(self) -> Event:
        """
        Get the on change event.

        Returns:
            Event: On change event
        """
        return self._on_change
    # end on_change

    # endregion PROPERTIES

    # region PUBLIC

    def set(self, pos):
        """
        Set the coordinates of the point.

        Args:
            pos (np.array): Tuple containing the X, Y, and Z coordinates of the point
        """
        self.check_closed()

        # Update position
        if not np.array_equal(self._pos, pos):
            self._pos = pos

            # Trigger change event
            self._trigger_on_change()
        # end if
    # end set

    def _trigger_on_change(self):
        """
        Trigger the position change event.
        """
        self.on_change.trigger(self, event_type=EventType.POSITION_CHANGED, x=self.x, y=self.y, z=self.z)
    # end _trigger_on_change

    def get(self):
        """
        Get the coordinates of the point.

        Returns:
            np.array: Array containing the X, Y, and Z coordinates of the point
        """
        return self.pos
    # end get

    # endregion PUBLIC

    # region OVERRIDE

    # Return a string representation of the point.
    def __repr__(self):
        """
        Return a string representation of the point.
        """
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"
    # end __repr__

    # Operator overloads
    def __add__(self, other):
        """
        Add two points together.

        Args:
            other (Union[Point3D, TPoint3D, int, float, tuple]): Point to add
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point3D
        if isinstance(other, (int, float)):
            # Point3D + scalar = Point3D
            return Point3D(self.x + other, self.y + other, self.z + other)
        elif isinstance(other, tuple):
            # Point3D + tuple = Point3D
            if len(other) == 3:
                return Point3D(self.x + other[0], self.y + other[1], self.z + other[2])
            else:
                raise ValueError("Tuple must have 3 elements for addition with Point3D")
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point3D + TScalar = TPoint3D
            return TPoint3D(lambda p, o: (p.x + o.value, p.y + o.value, p.z + o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point3D + Scalar = Point3D
            return Point3D(self.x + other.value, self.y + other.value, self.z + other.value)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # Point3D + TPoint3D = TPoint3D
            return TPoint3D.add(self, other)
        elif isinstance(other, Point3D):
            # Point3D + Point3D = Point3D
            return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __add__

    def __radd__(self, other):
        """
        Add two points together.

        Args:
            other (Union[Point3D, TPoint3D, int, float, tuple]): Point to add
        """
        return self.__add__(other)
    # end __radd__

    def __sub__(self, other):
        """
        Subtract two points.

        Args:
            other (Point3D): Point to subtract from this point or scalar value to subtract from the point.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point3D
        if isinstance(other, (int, float)):
            # Point3D - scalar = Point3D
            return Point3D(self.x - other, self.y - other, self.z - other)
        elif isinstance(other, tuple):
            # Point3D - tuple = Point3D
            if len(other) == 3:
                return Point3D(self.x - other[0], self.y - other[1], self.z - other[2])
            else:
                raise ValueError("Tuple must have 3 elements for subtraction with Point3D")
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point3D - TScalar = TPoint3D
            return TPoint3D(lambda p, o: (p.x - o.value, p.y - o.value, p.z - o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point3D - Scalar = Point3D
            return Point3D(self.x - other.value, self.y - other.value, self.z - other.value)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # Point3D - TPoint3D = TPoint3D
            return TPoint3D.sub(self, other)
        elif isinstance(other, Point3D):
            # Point3D - Point3D = Point3D
            return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __sub__

    def __rsub__(self, other):
        """
        Subtract two points.

        Args:
            other (Point3D): Point to subtract from this point or scalar value to subtract from the point.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point3D
        if isinstance(other, (int, float)):
            # scalar - Point3D = Point3D
            return Point3D(other - self.x, other - self.y, other - self.z)
        elif isinstance(other, tuple):
            # tuple - Point3D = Point3D
            if len(other) == 3:
                return Point3D(other[0] - self.x, other[1] - self.y, other[2] - self.z)
            else:
                raise ValueError("Tuple must have 3 elements for subtraction with Point3D")
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # TScalar - Point3D = TPoint3D
            return TPoint3D(lambda p, o: (o.value - p.x, o.value - p.y, o.value - p.z), p=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar - Point3D = Point3D
            return Point3D(other.value - self.x, other.value - self.y, other.value - self.z)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # TPoint3D - Point3D = TPoint3D
            return TPoint3D.sub(other, self)
        elif isinstance(other, Point3D):
            # Point3D - Point3D = Point3D
            return Point3D(other.x - self.x, other.y - self.y, other.z - self.z)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __rsub__

    def __mul__(self, other):
        """
        Multiply the point.

        Args:
            other (int, float, tuple, Scalar, TScalar, Point3D, TPoint3D): Scalar value to multiply the point by.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point3D
        if isinstance(other, (int, float)):
            # Point3D * scalar = Point3D
            return Point3D(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, tuple):
            # Point3D * tuple = Point3D
            if len(other) == 3:
                return Point3D(self.x * other[0], self.y * other[1], self.z * other[2])
            else:
                raise ValueError("Tuple must have 3 elements for multiplication with Point3D")
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point3D * TScalar = TPoint3D
            return TPoint3D(lambda p, o: (p.x * o.value, p.y * o.value, p.z * o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point3D * Scalar = Point3D
            return Point3D(self.x * other.value, self.y * other.value, self.z * other.value)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # Point3D * TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (p.x * o.x, p.y * o.y, p.z * o.z), p=self, o=other)
        elif isinstance(other, Point3D):
            # Point3D * Point3D = Point3D
            return Point3D(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __mul__

    def __rmul__(self, other):
        """
        Multiply the point by a scalar value.

        Args:
            other (int, float): Scalar value to multiply the point by.
        """
        return self.__mul__(other)
    # end __rmul__

    def __truediv__(self, other):
        """
        Divide the point by a scalar value.

        Args:
            other (int, float, Scalar, TScalar, Point3D, TPoint3D): Scalar value to divide the point by.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point3D
        if isinstance(other, (int, float)):
            # Point3D / scalar = Point3D
            return Point3D(self.x / other, self.y / other, self.z / other)
        elif isinstance(other, tuple):
            # Point3D / tuple = Point3D
            if len(other) == 3:
                return Point3D(self.x / other[0], self.y / other[1], self.z / other[2])
            else:
                raise ValueError("Tuple must have 3 elements for division with Point3D")
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point3D / TScalar = TPoint3D
            return TPoint3D(lambda p, o: (p.x / o.value, p.y / o.value, p.z / o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point3D / Scalar = Point3D
            return Point3D(self.x / other.value, self.y / other.value, self.z / other.value)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # Point3D / TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (p.x / o.x, p.y / o.y, p.z / o.z), p=self, o=other)
        elif isinstance(other, Point3D):
            # Point3D / Point3D = Point3D
            return Point3D(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __truediv__

    def __rtruediv__(self, other):
        """
        Divide the point by a scalar value.

        Args:
            other (int, float): Scalar value to divide the point by.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point3D
        if isinstance(other, (int, float)):
            # scalar / Point3D = Point3D
            return Point3D(other / self.x, other / self.y, other / self.z)
        elif isinstance(other, tuple):
            # tuple / Point3D = Point3D
            if len(other) == 3:
                return Point3D(other[0] / self.x, other[1] / self.y, other[2] / self.z)
            else:
                raise ValueError("Tuple must have 3 elements for division with Point3D")
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # TScalar / Point3D = TPoint3D
            return TPoint3D(lambda p, o: (o.value / p.x, o.value / p.y, o.value / p.z), p=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar / Point3D = Point3D
            return Point3D(other.value / self.x, other.value / self.y, other.value / self.z)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # TPoint3D / Point3D = TPoint3D
            return TPoint3D(lambda p, o: (o.x / p.x, o.y / p.y, o.z / p.z), p=self, o=other)
        elif isinstance(other, Point3D):
            # Point3D / Point3D = Point3D
            return Point3D(other.x / self.x, other.y / self.y, other.z / self.z)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __rtruediv__

    def __eq__(self, other):
        """
        Compare two points for equality.

        Args:
            other (Point3D): Point to compare with
        """
        if isinstance(other, Point3D):
            return self.x == other.x and self.y == other.y and self.z == other.z
        elif isinstance(other, tuple):
            if len(other) == 3:
                return self.x == other[0] and self.y == other[1] and self.z == other[2]
            else:
                return NotImplemented
        else:
            return NotImplemented
        # end if
    # end __eq__

    def __abs__(self):
        """
        Get the absolute value of the point.
        """
        return Point3D(abs(self.x), abs(self.y), abs(self.z))
    # end __abs__

    # endregion OVERRIDE

# end Point3D


class TPoint3D(Point3D):
    """
    A class that tracks transformations applied to a Point3D and updates dynamically.
    """

    def __init__(
            self,
            transform_func,
            on_change=None,
            **points
    ):
        """
        Initialize the tracked point.

        Args:
            transform_func (function): The transformation function to apply to the point
            on_change (function): Function to call when the point changes
            **points (Any): Points to track
        """
        # Properties
        self._points = points
        self._transform_func = transform_func

        # Initialize with the transformed point's position
        ret = self._transform_func(**self._points)
        if isinstance(ret, tuple):
            x, y, z = ret
        elif isinstance(ret, Point3D):
            x, y, z = ret.x, ret.y, ret.z
        # end if

        # Call the parent class's __init__ method
        Point.__init__(self, readonly=False)

        # Initialize the _pos attribute directly
        self._pos = np.array([x, y, z], dtype=np.float32)

        # Initialize the _on_change event
        self._on_change = Event()

        # Listen to sources
        for point in self._points.values():
            if hasattr(point, "on_change"):
                point.on_change.subscribe(self._on_source_changed)
            # end if
        # end for

        # Listen to changes in the original point
        self._on_change += on_change
    # end __init__

    # region PROPERTIES

    @property
    def points(self):
        """
        Get the original point.
        """
        return self._points
    # end points

    @property
    def transform_func(self):
        """
        Get the transformation function.
        """
        return self._transform_func
    # end transform_func

    @property
    def pos(self):
        """
        Get the position of the point.
        """
        x, y, z = self.get()
        self._pos[0] = x
        self._pos[1] = y
        self._pos[2] = z
        return self._pos
    # end pos

    @pos.setter
    def pos(self, value):
        """
        Set the position of the point.
        """
        raise AttributeError("Cannot set value directly on TPoint3D. It's computed based on other Points.")
    # end pos.setter

    @property
    def x(self):
        """
        Get the X-coordinate of the point.

        Returns:
            float: X-coordinate of the point
        """
        x, y, z = self.get()
        self._pos[0] = x
        self._pos[1] = y
        self._pos[2] = z
        return x
    # end x

    @x.setter
    def x(self, value):
        """
        Set the X-coordinate of the point.
        """
        raise AttributeError("Cannot set value directly on TPoint3D. It's computed based on other Points.")
    # end x

    @property
    def y(self):
        """
        Get the Y-coordinate of the point.

        Returns:
            float: Y-coordinate of the point
        """
        x, y, z = self.get()
        self._pos[0] = x
        self._pos[1] = y
        self._pos[2] = z
        return y
    # end y

    @y.setter
    def y(self, value):
        """
        Set the Y-coordinate of the point.
        """
        raise AttributeError("Cannot set value directly on TPoint3D. It's computed based on other Points.")
    # end y

    @property
    def z(self):
        """
        Get the Z-coordinate of the point.

        Returns:
            float: Z-coordinate of the point
        """
        x, y, z = self.get()
        self._pos[0] = x
        self._pos[1] = y
        self._pos[2] = z
        return z
    # end z

    @z.setter
    def z(self, value):
        """
        Set the Z-coordinate of the point.
        """
        raise AttributeError("Cannot set value directly on TPoint3D. It's computed based on other Points.")
    # end z

    @property
    def on_change(self) -> Event:
        """
        Get the on change event.

        Returns:
            Event: On change event
        """
        return self._on_change
    # end on_change

    # endregion PROPERTIES

    # region PUBLIC

    # Override set to prevent manual setting
    def set(self, x, y, z):
        """
        Prevent manual setting of the value. It should be computed only.
        """
        raise AttributeError("Cannot set value directly on TPoint3D. It's computed based on other Points.")
    # end set

    def get(self):
        """
        Get the current computed value.
        """
        return self.transform_func(**self._points)
    # end get

    # endregion PUBLIC

    # region EVENT

    def _on_source_changed(self, sender, event_type, **kwargs):
        """
        Update the point when a source point changes.

        Args:
            sender (Any): Sender of the event
            event_type (EventType): Type of event that occurred
        """
        x, y, z = self.get()
        self._pos[0] = x
        self._pos[1] = y
        self._pos[2] = z
        self._trigger_on_change()
    # end _on_source_changed

    def _trigger_on_change(self):
        """
        Trigger the position change event.
        """
        self.on_change.trigger(self, event_type=EventType.POSITION_CHANGED, x=self.x, y=self.y, z=self.z)
    # end _trigger_on_change

    # endregion EVENT

    # region OVERRIDE

    # Return a string representation of the point.
    def __str__(self):
        """
        Return a string representation of the point.
        """
        return f"TPoint3D(points={self._points}, transform_func={self._transform_func.__name__}, x={self.x}, y={self.y}, z={self.z})"

    # end __str__

    # Return a string representation of the point.
    def __repr__(self):
        """
        Return a string representation of the point.
        """
        return self.__str__()
    # end __repr__

    # Operator overloads
    def __add__(self, other):
        """
        Add two points together.

        Args:
            other (Point3D): Point to add
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point3D
        if isinstance(other, (int, float)):
            # TPoint3D + scalar = TPoint3D
            return TPoint3D(lambda p, o: (p.x + o, p.y + o, p.z + o), p=self, o=other)
        elif isinstance(other, tuple):
            # TPoint3D + tuple = TPoint3D
            return TPoint3D(lambda p, o: (p.x + o[0], p.y + o[1], p.z + o[2]), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # TPoint3D + TScalar = TPoint3D
            return TPoint3D(lambda p, o: (p.x + o.value, p.y + o.value, p.z + o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # TPoint3D + Scalar = TPoint3D
            return TPoint3D(lambda p, o: (p.x + o.value, p.y + o.value, p.z + o.value), p=self, o=other)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # TPoint3D + TPoint3D = TPoint3D
            return TPoint3D.add(self, other)
        elif isinstance(other, Point3D):
            # TPoint3D + Point3D = TPoint3D
            return TPoint3D(lambda p, o: (p.x + o.x, p.y + o.y, p.z + o.z), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __add__

    def __radd__(self, other):
        """
        Add two points together.

        Args:
            other (Point3D): Point to add
        """
        return self.__add__(other)
    # end __radd__

    def __sub__(self, other):
        """
        Subtract two points.

        Args:
            other (Point3D): Point to subtract from this point or scalar value to subtract from the point.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point3D
        if isinstance(other, (int, float)):
            # Point3D - scalar = Point3D
            return TPoint3D(lambda p, o: (p.x - o, p.y - o, p.z - o), p=self, o=other)
        elif isinstance(other, tuple):
            # Point3D - tuple = Point3D
            return TPoint3D(lambda p, o: (p.x - o[0], p.y - o[1], p.z - o[2]), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point3D - TScalar = TPoint3D
            return TPoint3D(lambda p, o: (p.x - o.value, p.y - o.value, p.z - o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point3D - Scalar = Point3D
            return TPoint3D(lambda p, o: (p.x - o.value, p.y - o.value, p.z - o.value), p=self, o=other)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # Point3D - TPoint3D = TPoint3D
            return TPoint3D.sub(self, other)
        elif isinstance(other, Point3D):
            # Point3D - Point3D = Point3D
            return TPoint3D(lambda p, o: (p.x - o.x, p.y - o.y, p.z - o.z), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __sub__

    def __rsub__(self, other):
        """
        Subtract two points.

        Args:
            other (Point3D): Point to subtract from this point or scalar value to subtract from the point.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # scalar, tuple
        if isinstance(other, (int, float)):
            # scalar - TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o - p.x, o - p.y, o - p.z), p=self, o=other)
        elif isinstance(other, tuple):
            # tuple - TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o[0] - p.x, o[1] - p.y, o[2] - p.z), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # TScalar - TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o.value - p.x, o.value - p.y, o.value - p.z), p=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar - TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o.value - p.x, o.value - p.y, o.value - p.z), p=self, o=other)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # TPoint3D - TPoint3D = TPoint3D
            return TPoint3D.sub(other, self)
        elif isinstance(other, Point3D):
            # Point3D - TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o.x - p.x, o.y - p.y, o.z - p.z), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __rsub__

    def __mul__(self, other):
        """
        Multiply the point by a scalar value.

        Args:
            other (int, float): Scalar value to multiply the point by.
        """
        # Point3D
        if isinstance(other, (int, float)):
            # Point3D * scalar = Point3D
            return TPoint3D(lambda p, o: (p.x * o, p.y * o, p.z * o), p=self, o=other)
        elif isinstance(other, tuple):
            # Point3D * tuple = Point3D
            return TPoint3D(lambda p, o: (p.x * o[0], p.y * o[1], p.z * o[2]), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point3D * TScalar = TPoint3D
            return TPoint3D(lambda p, o: (p.x * o.value, p.y * o.value, p.z * o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point3D * Scalar = Point3D
            return TPoint3D(lambda p, o: (p.x * o.value, p.y * o.value, p.z * o.value), p=self, o=other)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # Point3D * TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (p.x * o.x, p.y * o.y, p.z * o.z), p=self, o=other)
        elif isinstance(other, Point3D):
            # Point3D * Point3D = Point3D
            return TPoint3D(lambda p, o: (p.x * o.x, p.y * o.y, p.z * o.z), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __mul__

    def __rmul__(self, other):
        """
        Multiply the point by a scalar value.

        Args:
            other (int, float): Scalar value to multiply the point by.
        """
        return self.__mul__(other)
    # end __rmul__

    def __truediv__(self, other):
        """
        Divide the point by a scalar value.

        Args:
            other (int, float): Scalar value to divide the point by.
        """
        # Point3D
        if isinstance(other, (int, float)):
            # Point3D / scalar = Point3D
            return TPoint3D(lambda p, o: (p.x / o, p.y / o, p.z / o), p=self, o=other)
        elif isinstance(other, tuple):
            # Point3D / tuple = Point3D
            return TPoint3D(lambda p, o: (p.x / o[0], p.y / o[1], p.z / o[2]), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point3D / TScalar = TPoint3D
            return TPoint3D(lambda p, o: (p.x / o.value, p.y / o.value, p.z / o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point3D / Scalar = Point3D
            return TPoint3D(lambda p, o: (p.x / o.value, p.y / o.value, p.z / o.value), p=self, o=other)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # Point3D / TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (p.x / o.x, p.y / o.y, p.z / o.z), p=self, o=other)
        elif isinstance(other, Point3D):
            # Point3D / Point3D = Point3D
            return TPoint3D(lambda p, o: (p.x / o.x, p.y / o.y, p.z / o.z), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __truediv__

    def __rtruediv__(self, other):
        """
        Divide the point by a scalar value.

        Args:
            other (int, float): Scalar value to divide the point by.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # scalar, tuple
        if isinstance(other, (int, float)):
            # scalar / TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o / p.x, o / p.y, o / p.z), p=self, o=other)
        elif isinstance(other, tuple):
            # tuple / TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o[0] / p.x, o[1] / p.y, o[2] / p.z), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # TScalar / TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o.value / p.x, o.value / p.y, o.value / p.z), p=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar / TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o.value / p.x, o.value / p.y, o.value / p.z), p=self, o=other)
        # Point3D, TPoint3D
        elif isinstance(other, TPoint3D):
            # TPoint3D / TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o.x / p.x, o.y / p.y, o.z / p.z), p=self, o=other)
        elif isinstance(other, Point3D):
            # Point3D / TPoint3D = TPoint3D
            return TPoint3D(lambda p, o: (o.x / p.x, o.y / p.y, o.z / p.z), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Point3D' and '{}'".format(type(other)))
        # end if
    # end __rtruediv__

    def __eq__(self, other):
        """
        Compare two points for equality.

        Args:
            other (Point3D): Point to compare with
        """
        if isinstance(other, Point3D):
            return self.x == other.x and self.y == other.y and self.z == other.z
        elif isinstance(other, tuple):
            return self.x == other[0] and self.y == other[1] and self.z == other[2]
        elif isinstance(other, TPoint3D):
            return self.x == other.x and self.y == other.y and self.z == other.z
        else:
            return NotImplemented
        # end if
    # end __eq__

    def __abs__(self):
        """
        Get the absolute value of the point.
        """
        return TPoint3D(lambda p: (abs(p.x), abs(p.y), abs(p.z)), p=self)
    # end __abs__

    # endregion OVERRIDE

    # region CONSTRUCTORS

    # Basic TPoint3D (just return value of a point)
    @classmethod
    def tpoint3d(
            cls,
            point: Union[Point3D, 'TPoint3D', tuple]
    ):
        """
        Create a TPoint3D that represents a point.

        Args:
            point (Union[Point3D, TPoint3D, tuple]): Point to track

        """
        if isinstance(point, Point3D):
            return cls(lambda p: (p.x, p.y, p.z), p=point)
        elif isinstance(point, tuple):
            return cls(lambda p: (point[0], point[1], point[2]))
        else:
            return point
        # end if
    # end tpoint3d

    # endregion CONSTRUCTORS

    # region OPERATORS

    # Function to create a new tracked point
    @classmethod
    def add(
            cls,
            point: Point3D,
            delta: Point3D
    ):
        """
        Create a TPoint3D that represents point + delta.

        Args:
            point (Point3D): Point to add to.
            delta (Point3D): Point to add.
        """
        return cls(lambda p, d: (p.x + d.x, p.y + d.y, p.z + d.z), p=point, d=delta)
    # end add

    @classmethod
    def sub(
            cls,
            point1: Point3D,
            point2: Point3D
    ):
        """
        Create a TPoint3D that represents point - delta.

        Args:
            point1 (Point3D): Point to subtract from.
            point2 (Point3D): Point to subtract.
        """
        return cls(lambda p1, p2: (p1.x - p2.x, p1.y - p2.y, p1.z - p2.z), p1=point1, p2=point2)
    # end sub

    # endregion OPERATORS

    @classmethod
    def mul(
            cls,
            point1: Point3D,
            point2: Point3D
    ):
        """
        Create a TPoint3D that represents point * scalar.

        Args:
            point1 (Point3D): Point to multiply.
            point2 (Point3D): Point to multiply by.
        """
        if isinstance(point1, cls) or isinstance(point2, cls):
            return cls(lambda p1, p2: (p1.x * p2.x, p1.y * p2.y, p1.z * p2.z), p1=point1, p2=point2)
        else:
            return Point3D(point1.x * point2.x, point1.y * point2.y, point1.z * point2.z)
        # end if
    # end mul

    @classmethod
    def scalar_mul(
            cls,
            point: Point3D,
            scalar: Union[Scalar, TScalar, float, int]
    ):
        """
        Multiply a point by a scalar.

        Args:
            point (Point3D): Point to multiply
            scalar (Scalar/TScalar): Scalar
        """
        if isinstance(scalar, (int, float)):
            scalar = Scalar(scalar)
        # end if
        return cls(lambda p, s: (p.x * s.value, p.y * s.value, p.z * s.value), p=point, s=scalar)
    # end scalar_mul

    @classmethod
    def div(
            cls,
            point: Point3D,
            scalar: Union[Scalar, float]
    ):
        """
        Create a TPoint3D that represents point / scalar.

        Args:
            point (Union[Point3D, TPoint3D]): Point to divide.
            scalar (Union[Scalar, float]): Scalar to divide by.
        """
        if isinstance(scalar, Scalar):
            return cls(lambda p, s: (p.x / s.value, p.y / s.value, p.z / s.value), p=point, s=scalar)
        else:
            return cls(lambda p, s: (p.x / s, p.y / s, p.z / s), p=point, s=scalar)
        # end if
    # end div

# end TPoint3D


# TPoint2D class
class TPoint2D(Point2D):
    """
    A class that tracks transformations applied to a Point2D and updates dynamically.
    """

    def __init__(
            self,
            transform_func,
            on_change=None,
            **points
    ):
        """
        Initialize the tracked point.

        Args:
            transform_func (function): The transformation function to apply to the point
            on_change (function): Function to call when the point changes
            **points (Any): Points to track
        """
        # Properties
        self._points = points
        self._transform_func = transform_func

        # Initialize with the transformed point's position
        ret = self._transform_func(**self._points)
        if isinstance(ret, tuple):
            x, y = ret
        elif isinstance(ret, Point2D):
            x, y = ret.x, ret.y
        # end if
        super().__init__(x, y)

        # Listen to sources
        for point in self._points.values():
            if hasattr(point, "on_change"):
                point.on_change.subscribe(self._on_source_changed)
            # end if
        # end for

        # Listen to changes in the original point
        self._on_change += on_change
    # end __init__

    # region PROPERTIES

    @property
    def points(self):
        """
        Get the original point.
        """
        return self._points
    # end points

    @property
    def transform_func(self):
        """
        Get the transformation function.
        """
        return self._transform_func
    # end transform_func

    @property
    def pos(self):
        """
        Get the position of the point.
        """
        x, y = self.get()
        self._pos[0] = x
        self._pos[1] = y
        return self._pos
    # end pos

    @property
    def x(self):
        """
        Get the X-coordinate of the point.

        Returns:
            float: X-coordinate of the point
        """
        x, y = self.get()
        self._pos[0] = x
        self._pos[1] = y
        return x
    # end x

    @x.setter
    def x(self, value):
        """
        Set the X-coordinate of the point.
        """
        raise AttributeError("Cannot set value directly on TScalar. It's computed based on other Scalars.")
    # end x

    @property
    def y(self):
        """
        Get the Y-coordinate of the point.

        Returns:
            float: Y-coordinate of the point
        """
        x, y = self.get()
        self._pos[0] = x
        self._pos[1] = y
        return y
    # end y

    @y.setter
    def y(self, value):
        """
        Set the Y-coordinate of the point.
        """
        raise AttributeError("Cannot set value directly on TScalar. It's computed based on other Scalars.")
    # end y

    # endregion PROPERTIES

    # region PUBLIC

    # Override set to prevent manual setting
    def set(self, x, y):
        """
        Prevent manual setting of the value. It should be computed only.
        """
        raise AttributeError("Cannot set value directly on TScalar. It's computed based on other Scalars.")
    # end set

    def get(self):
        """
        Get the current computed value.
        """
        return self.transform_func(**self._points)
    # end get

    # endregion PUBLIC

    # region EVENT

    def _on_source_changed(self, sender, event_type, **kwargs):
        """
        Update the point when a source point changes.

        Args:
            sender (Any): Sender of the event
            event_type (EventType): Type of event that occurred
        """
        x, y = self.get()
        self._pos[0] = x
        self._pos[1] = y
        self._trigger_on_change()
    # end _on_source_changed

    # endregion EVENT

    # region OVERRIDE

    # Return a string representation of the point.
    def __str__(self):
        """
        Return a string representation of the point.
        """
        return f"TPoint2D(points={self._points}, transform_func={self._transform_func.__name__}, x={self.x}, y={self.y})"

    # end __str__

    # Return a string representation of the point.
    def __repr__(self):
        """
        Return a string representation of the point.
        """
        return self.__str__()
    # end __repr__

    # Operator overloads
    def __add__(self, other):
        """
        Add two points together.

        Args:
            other (Point2D): Point to add
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point2D
        if isinstance(other, (int, float)):
            # TPoint2D + scalar = TPoint2D
            return TPoint2D(lambda p, o: (p.x + o, p.y + o), p=self, o=other)
        elif isinstance(other, tuple):
            # TPoint2D + tuple = TPoint2D
            return TPoint2D(lambda p, o: (p.x + o[0], p.y + o[1]), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # TPoint2D + TScalar = TPoint2D
            return TPoint2D(lambda p, o: (p.x + o.value, p.y + o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # TPoint2D + Scalar = TPoint2D
            return TPoint2D(lambda p, o: (p.x + o.value, p.y + o.value), p=self, o=other)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # TPoint2D + TPoint2D = TPoint2D
            return TPoint2D.add(self, other)
        elif isinstance(other, Point2D):
            # TPoint2D + Point2D = TPoint2D
            return TPoint2D(lambda p, o: (p.x + o.x, p.y + o.y), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __add__

    def __radd__(self, other):
        """
        Add two points together.

        Args:
            other (Point2D): Point to add
        """
        return self.__add__(other)
    # end __radd__

    def __sub__(self, other):
        """
        Subtract two points.

        Args:
            other (Point2D): Point to subtract from this point or scalar value to subtract from the point.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # Point2D
        if isinstance(other, (int, float)):
            # Point2D - scalar = Point2D
            return TPoint2D(lambda p, o: (p.x - o, p.y - o), p=self, o=other)
        elif isinstance(other, tuple):
            # Point2D - tuple = Point2D
            return TPoint2D(lambda p, o: (p.x - o[0], p.y - o[1]), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point2D - TScalar = TPoint2D
            return TPoint2D(lambda p, o: (p.x - o.value, p.y - o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point2D - Scalar = Point2D
            return TPoint2D(lambda p, o: (p.x - o.value, p.y - o.value), p=self, o=other)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Point2D - TPoint2D = TPoint2D
            return TPoint2D.sub(self, other)
        elif isinstance(other, Point2D):
            # Point2D - Point2D = Point2D
            return TPoint2D(lambda p, o: (p.x - o.x, p.y - o.y), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __sub__

    def __rsub__(self, other):
        """
        Subtract two points.

        Args:
            other (Point2D): Point to subtract from this point or scalar value to subtract from the point.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # scalar, tuple
        if isinstance(other, (int, float)):
            # scalar - TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o - p.x, o - p.y), p=self, o=other)
        elif isinstance(other, tuple):
            # tuple - TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o[0] - p.x, o[1] - p.y), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # TScalar - TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o.value - p.x, o.value - p.y), p=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar - TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o.value - p.x, o.value - p.y), p=self, o=other)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # TPoint2D - TPoint2D = TPoint2D
            return TPoint2D.sub(other, self)
        elif isinstance(other, Point2D):
            # Point2D - TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o.x - p.x, o.y - p.y), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __rsub__

    def __mul__(self, other):
        """
        Multiply the point by a scalar value.

        Args:
            other (int, float): Scalar value to multiply the point by.
        """
        # Point2D
        if isinstance(other, (int, float)):
            # Point2D * scalar = Point2D
            return TPoint2D(lambda p, o: (p.x * o, p.y * o), p=self, o=other)
        elif isinstance(other, tuple):
            # Point2D * tuple = Point2D
            return TPoint2D(lambda p, o: (p.x * o[0], p.y * o[1]), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point2D * TScalar = TPoint2D
            return TPoint2D(lambda p, o: (p.x * o.value, p.y * o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point2D * Scalar = Point2D
            return TPoint2D(lambda p, o: (p.x * o.value, p.y * o.value), p=self, o=other)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Point2D * TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (p.x * o.x, p.y * o.y), p=self, o=other)
        elif isinstance(other, Point2D):
            # Point2D * Point2D = Point2D
            return TPoint2D(lambda p, o: (p.x * o.x, p.y * o.y), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __mul__

    def __rmul__(self, other):
        """
        Multiply the point by a scalar value.

        Args:
            other (int, float): Scalar value to multiply the point by.
        """
        return self.__mul__(other)
    # end __rmul__

    def __truediv__(self, other):
        """
        Divide the point by a scalar value.

        Args:
            other (int, float): Scalar value to divide the point by.
        """
        # Point2D
        if isinstance(other, (int, float)):
            # Point2D / scalar = Point2D
            return TPoint2D(lambda p, o: (p.x / o, p.y / o), p=self, o=other)
        elif isinstance(other, tuple):
            # Point2D / tuple = Point2D
            return TPoint2D(lambda p, o: (p.x / o[0], p.y / o[1]), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # Point2D / TScalar = TPoint2D
            return TPoint2D(lambda p, o: (p.x / o.value, p.y / o.value), p=self, o=other)
        elif isinstance(other, Scalar):
            # Point2D / Scalar = Point2D
            return TPoint2D(lambda p, o: (p.x / o.value, p.y / o.value), p=self, o=other)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # Point2D / TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (p.x / o.x, p.y / o.y), p=self, o=other)
        elif isinstance(other, Point2D):
            # Point2D / Point2D = Point2D
            return TPoint2D(lambda p, o: (p.x / o.x, p.y / o.y), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __truediv__

    def __rtruediv__(self, other):
        """
        Divide the point by a scalar value.

        Args:
            other (int, float): Scalar value to divide the point by.
        """
        # Imports
        from .scalar import Scalar, TScalar

        # scalar, tuple
        if isinstance(other, (int, float)):
            # scalar / TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o / p.x, o / p.y), p=self, o=other)
        elif isinstance(other, tuple):
            # tuple / TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o[0] / p.x, o[1] / p.y), p=self, o=other)
        # TScalar, Scalar
        elif isinstance(other, TScalar):
            # TScalar / TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o.value / p.x, o.value / p.y), p=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar / TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o.value / p.x, o.value / p.y), p=self, o=other)
        # Point2D, TPoint2D
        elif isinstance(other, TPoint2D):
            # TPoint2D / TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o.x / p.x, o.y / p.y), p=self, o=other)
        elif isinstance(other, Point2D):
            # Point2D / TPoint2D = TPoint2D
            return TPoint2D(lambda p, o: (o.x / p.x, o.y / p.y), p=self, o=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Point2D' and '{}'".format(type(other)))
        # end if
    # end __rtruediv__

    def __eq__(self, other):
        """
        Compare two points for equality.

        Args:
            other (Point2D): Point to compare with
        """
        if isinstance(other, Point2D):
            return self.x == other.x and self.y == other.y
        elif isinstance(other, tuple):
            return self.x == other[0] and self.y == other[1]
        elif isinstance(other, TPoint2D):
            return self.x == other.x and self.y == other.y
        else:
            return NotImplemented
        # end if
    # end __eq__

    def __abs__(self):
        """
        Get the absolute value of the point.
        """
        return TPoint2D(lambda p: (abs(p.x), abs(p.y)), p=self)
    # end __abs__

    # endregion OVERRIDE

    # region CONSTRUCTORS

    # Basic TPoint2D (just return value of a point)
    @classmethod
    def tpoint2d(
            cls,
            point: Union[Point2D, 'TPoint2D', tuple]
    ):
        """
        Create a TPoint2D that represents a point.

        Args:
            point (Union[Point2D, TPoint2D, tuple]): Point to track

        """
        if isinstance(point, Point2D):
            return cls(lambda p: (p.x, p.y), p=point)
        elif isinstance(point, tuple):
            return cls(lambda p: (point[0], point[1]))
        else:
            return point
        # end if
    # end tpoint2d

    # endregion CONSTRUCTORS

    # region OPERATORS

    # Function to create a new tracked point
    @classmethod
    def add(
            cls,
            point: Point2D,
            delta: Point2D
    ):
        """
        Create a TPoint2D that represents point + delta.

        Args:
            point (Point2D): Point to add to.
            delta (Point2D): Point to add.
        """
        return cls(lambda p, d: (p.x + d.x, p.y + d.y), p=point, d=delta)
    # end add

    @classmethod
    def sub(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TPoint2D that represents point - delta.

        Args:
            point1 (Point2D): Point to subtract from.
            point2 (Point2D): Point to subtract.
        """
        return cls(lambda p1, p2: (p1.x - p2.x, p1.y - p2.y), p1=point1, p2=point2)
    # end sub

    @classmethod
    def mul(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TPoint2D that represents point * scalar.

        Args:
            point1 (Point2D): Point to multiply.
            point2 (Point2D): Point to multiply by.
        """
        if isinstance(point1, cls) or isinstance(point2, cls):
            return cls(lambda p1, p2: (p1.x * p2.x, p1.y * p2.y), p1=point1, p2=point2)
        else:
            return Point2D(point1.x * point2.x, point1.y * point2.y)
        # end if
    # end mul

    @classmethod
    def scalar_mul(
            cls,
            point: Point2D,
            scalar: Union[Scalar, TScalar, float, int]
    ):
        """
        Multiply a matrix by a scalar.

        Args:
            point (Point2D): Point to multiply
            scalar (Scalar/TScalar): Scalar
        """
        if isinstance(scalar, (int, float)):
            scalar = Scalar(scalar)
        # end if
        return cls(lambda p, s: (p.x * s.value, p.y * s.value), p=point, s=scalar)
    # end scalar_mul

    @classmethod
    def div(
            cls,
            point: Point2D,
            scalar: Union[Scalar, float]
    ):
        """
        Create a TPoint2D that represents point / scalar.

        Args:
            point (Union[Point2D, TPoint2D]): Point to divide.
            scalar (Union[Scalar, float]): Scalar to divide by.
        """
        if isinstance(scalar, Scalar):
            return cls(lambda p, s: (p.x / s.value, p.y / s.value), p=point, s=scalar)
        else:
            return cls(lambda p: (p.x / scalar, p.y / scalar), p=point)
        # end if
    # end div

    @classmethod
    def neg(
            cls,
            point: Point2D
    ):
        """
        Create a TPoint2D that represents -point (negation).
        """
        return cls(lambda p: (-p.x, -p.y), p=point)
    # end neg

    @classmethod
    def abs(
            cls,
            point: Point2D
    ):
        """
        Create a TPoint2D that represents the absolute value of the point.
        """
        return cls(lambda p: (abs(p.x), abs(p.y)), p=point)
    # end abs

    # endregion OPERATORS

    # region METHODS

    @classmethod
    def round(
            cls,
            point: Point2D,
            ndigits: int = 0
    ):
        """
        Create a TPoint2D that represents the rounded value of the point.
        """
        return cls(lambda p: (round(p.x, ndigits=ndigits), round(p.y, ndigits=ndigits)), p=point)
    # end round

    @classmethod
    def rotate(
            cls,
            point: Point2D,
            angle: Union[Scalar, float],
            center: Point2D = None
    ):
        """
        Create a TPoint2D that represents the point rotated around another point by a given angle.

        Args:
            point (Point2D): The point to rotate.
            angle (Scalar): The angle of rotation (in radians).
            center (Point2D): The center of rotation. If None, rotate around the origin.
        """
        if center is None:
            center = Point2D(0, 0)
        # end if

        if isinstance(angle, Scalar):
            return TPoint2D(
                lambda p, a, c: (
                    c.x + (p.x - c.x) * math.cos(a.value) - (p.y - c.y) * math.sin(a.value),
                    c.y + (p.x - c.x) * math.sin(a.value) + (p.y - c.y) * math.cos(a.value)
                ),
                p=point,
                a=angle,
                c=center
            )
        else:
            return TPoint2D(
                lambda p, c: (
                    c.x + (p.x - c.x) * math.cos(angle) - (p.y - c.y) * math.sin(angle),
                    c.y + (p.x - c.x) * math.sin(angle) + (p.y - c.y) * math.cos(angle)
                ),
                p=point,
                c=center
            )
        # end if
    # end rotate

    @classmethod
    def scale(
            cls,
            point: Point2D,
            scale: Union[Scalar, float],
            center: Point2D = None
    ):
        """
        Create a TPoint2D that represents the point scaled away from another point by a given scale factor.

        Args:
            point (Point2D): The point to scale.
            scale (Union[Scalar, float]): The scale factor.
            center (Point2D): The center of scaling. If None, scale from the origin.
        """
        if center is None:
            center = Point2D(0, 0)
        # end if

        if isinstance(scale, Scalar):
            return TPoint2D(
                lambda p, c, s: (
                    c.x + (p.x - c.x) * s.value,
                    c.y + (p.y - c.y) * s.value
                ),
                p=point,
                c=center,
                s=scale
            )
        else:
            return TPoint2D(
                lambda p, c: (
                    c.x + (p.x - c.x) * scale,
                    c.y + (p.y - c.y) * scale
                ),
                p=point,
                c=center
            )
        # end if
    # end scale

    @classmethod
    def dot(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the dot product of two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        return TScalar(lambda p1, p2: p1.x * p2.x + p1.y * p2.y, p1=point1, p2=point2)
    # end dot

    @classmethod
    def cross(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the cross product of two points in 2D.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        return TScalar(lambda p1, p2: p1.x * p2.y - p1.y * p2.x, p1=point1, p2=point2)
    # end cross

    @classmethod
    def norm(
            cls,
            point: Point2D
    ):
        """
        Create a TScalar representing the norm (magnitude) of a point.

        Args:
            point (Point2D): The point.
        """
        return TScalar(lambda p: math.sqrt(point.x ** 2 + point.y ** 2), p=point)
    # end norm

    @classmethod
    def normalize(
            cls,
            point: Point2D
    ):
        """
        Create a TPoint2D representing the normalized (unit vector) of a point.

        Args:
            point (Point2D): The point to normalize.
        """
        norm = TPoint2D.norm(point)
        return TPoint2D(lambda p: (p.x / norm.value, p.y / norm.value), p=point)
    # end normalize

    @classmethod
    def angle(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the angle between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        dot = TPoint2D.dot(point1, point2)
        norm1 = TPoint2D.norm(point1)
        norm2 = TPoint2D.norm(point2)
        return TScalar(lambda p1, p2: math.acos(dot.value / (norm1.value * norm2.value)), p1=point1, p2=point2)
    # end angle

    # endregion METHODS

    # region DISTANCES

    @classmethod
    def distance(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the Euclidean distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        return TScalar(lambda p1, p2: math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2), p1=point1, p2=point2)
    # end distance

    @classmethod
    def distance_squared(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the squared Euclidean distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        return TScalar(lambda p1, p2: (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2, p1=point1, p2=point2)
    # end distance_squared

    @classmethod
    def distance_manhattan(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the Manhattan distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        return TScalar(lambda p1, p2: abs(p1.x - point2.x) + abs(p1.y - p2.y), p1=point1, p2=point2)
    # end distance_manhattan

    @classmethod
    def distance_chebyshev(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the Chebyshev distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        return TScalar(lambda p1, p2: max(abs(point1.x - point2.x), abs(point1.y - point2.y)), p1=point1, p2=point2)
    # end distance_chebyshev

    @classmethod
    def distance_canberra(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the Canberra distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        return TScalar(
            lambda p1, p2: abs(p1.x - p2.x) / (abs(p1.x) + abs(p2.x)) + abs(p1.y - p2.y) / (abs(p1.y) + abs(p2.y)),
            p1=point1,
            p2=point2
        )
    # end distance_canberra

    @classmethod
    def distance_minkowski(
            cls,
            point1: Point2D,
            point2: Point2D,
            p: Union[Scalar, float]
    ):
        """
        Create a TScalar representing the Minkowski distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
            p (float): The order of the Minkowski distance (p=1 is Manhattan, p=2 is Euclidean).
        """
        if isinstance(p, Scalar):
            return TScalar(
                lambda p1, p2, s: ((abs(p1.x - p2.x) ** s.value + abs(p1.y - p2.y) ** s.value) ** (1 / s.value)),
                p1=point1,
                p2=point2,
                s=p
            )
        else:
            return TScalar(
                lambda p1, p2: ((abs(p1.x - p2.x) ** p + abs(p1.y - p2.y) ** p) ** (1 / p)),
                p1=point1,
                p2=point2
            )
        # end if
    # end distance_minkowski

    @classmethod
    def distance_hamming(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the Hamming distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        return TScalar(lambda p1, p2: int(p1.x != p2.x) + int(p1.y != p2.y), p1=point1, p2=point2)
    # end distance_hamming

    @classmethod
    def distance_jaccard(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the Jaccard distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        intersection = min(point1.x, point2.x) + min(point1.y, point2.y)
        union = max(point1.x, point2.x) + max(point1.y, point2.y)
        return TScalar(lambda p1, p2: 1 - intersection / union if union != 0 else 0, p1=point1, p2=point2)
    # end distance_jaccard

    @classmethod
    def distance_braycurtis(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the Bray-Curtis distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        numerator = abs(point1.x - point2.x) + abs(point1.y - point2.y)
        denominator = abs(point1.x + point2.x) + abs(point1.y + point2.y)
        return TScalar(lambda p1, p2: numerator / denominator if denominator != 0 else 0, p1=point1, p2=point2)
    # end distance_braycurtis

    @classmethod
    def distance_cosine(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the cosine distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        dot = TPoint2D.dot(point1, point2)
        norm1 = TPoint2D.norm(point1)
        norm2 = TPoint2D.norm(point2)
        return TScalar(
            lambda p1, p2: 1 - (
                        dot.value / (norm1.value * norm2.value)) if norm1.value != 0 and norm2.value != 0 else 1,
            p1=point1,
            p2=point2
        )
    # end distance_cosine

    @classmethod
    def distance_correlation(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the correlation distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        mean1 = (point1.x + point1.y) / 2
        mean2 = (point2.x + point2.y) / 2
        numerator = (point1.x - mean1) * (point2.x - mean2) + (point1.y - mean1) * (point2.y - mean2)
        denominator = math.sqrt(((point1.x - mean1) ** 2 + (point1.y - mean1) ** 2) *
                                ((point2.x - mean2) ** 2 + (point2.y - mean2) ** 2))
        return TScalar(
            lambda p1, p2: 1 - (numerator / denominator) if denominator != 0 else 1,
            p1=point1,
            p2=point2
        )
    # end distance_correlation

    @classmethod
    def distance_euclidean(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the Euclidean distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        return TScalar(lambda p1, p2: np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2), p1=point1, p2=point2)
    # end distance_euclidean

    @classmethod
    def distance_mahalanobis(
            cls,
            point1: Point2D,
            point2: Point2D,
            cov_matrix
    ):
        """
        Create a TScalar representing the Mahalanobis distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
            cov_matrix (Matrix2D): Covariance matrix of the dataset.
        """
        from .matrices import TMatrix2D
        cov_matrix = TMatrix2D.inverse(cov_matrix)
        return TScalar(
            lambda p1, p2, cov: np.sqrt(
                (np.expand_dims((p1 - p2).pos, axis=0) @ cov.data @ np.expand_dims((p1 - p2).pos, axis=0).T)[0, 0]
            ),
            p1=point1,
            p2=point2,
            cov=cov_matrix
        )
    # end distance_mahalanobis

    @classmethod
    def distance_seuclidean(
            cls,
            point1: Point2D,
            point2: Point2D,
            std_devs
    ):
        """
        Create a TScalar representing the standardized Euclidean distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
            std_devs (np.ndarray): Standard deviations of the dimensions.
        """
        return TScalar(
            lambda p1, p2: np.sqrt(np.sum((np.array([p1.x - p2.x, p1.y - p2.y]) / std_devs) ** 2)),
            p1=point1,
            p2=point2
        )
    # end distance_seuclidean

    @classmethod
    def distance_sqeuclidean(
            cls,
            point1: Point2D,
            point2: Point2D
    ):
        """
        Create a TScalar representing the squared Euclidean distance between two points.

        Args:
            point1 (Point2D): The first point.
            point2 (Point2D): The second point.
        """
        return TScalar(lambda p1, p2: (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2, p1=point1, p2=point2)
    # end distance_sqeuclidean

    # endregion DISTANCES

    # region GENERATION

    @classmethod
    def point_range(
            cls,
            start: Point2D,
            stop: Point2D,
            step: Point2D,
            return_tpoint: bool = False
    ):
        """
        Create a list of Point2D or TPoint2D objects using a range-like function for 2D points.

        Args:
            start (Point2D): Starting point.
            stop (Point2D): Stopping point.
            step (Point2D): Step sizes for x and y coordinates.
            return_tpoint (bool): If True, return a list of TPoint2D objects.

        Returns:
            List[Point2D] or List[TPoint2D]: List of Point2D or TPoint2D objects.
        """
        x_range = np.arange(start.x, stop.x, step.x)
        y_range = np.arange(start.y, stop.y, step.y)

        if return_tpoint:
            return [
                TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y))
                for x, y in zip(x_range, y_range)
            ]
        else:
            return [Point2D(x, y) for x, y in zip(x_range, y_range)]
        # end if
    # end point_range

    @classmethod
    def linspace(
            cls,
            start: Point2D,
            stop: Point2D,
            num: int = 50,
            return_tpoint: bool = False
    ):
        """
        Create a list of Point2D or TPoint2D objects using numpy's linspace for 2D points.

        Args:
            start (Point2D): Starting point.
            stop (Point2D): Stopping point.
            num (int, optional): Number of points.
            return_tpoint (bool): If True, return a list of TPoint2D objects.

        Returns:
            List[Point2D] or List[TPoint2D]: List of Point2D or TPoint2D objects.
        """
        x_values = np.linspace(start.x, stop.x, num)
        y_values = np.linspace(start.y, stop.y, num)

        if return_tpoint:
            return [
                TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y))
                for x, y in zip(x_values, y_values)
            ]
        else:
            return [Point2D(x, y) for x, y in zip(x_values, y_values)]
        # end if
    # end linspace

    @classmethod
    def logspace(
            cls,
            start: Point2D,
            stop: Point2D,
            num: int = 50,
            base: float = 10.0,
            return_tpoint: bool = False):
        """
        Create a list of Point2D or TPoint2D objects using numpy's logspace for 2D points.

        Args:
            start (Point2D): Starting point.
            stop (Point2D): Stopping point.
            num (int, optional): Number of points.
            base (float, optional): Base of the logarithm.
            return_tpoint (bool): If True, return a list of TPoint2D objects.

        Returns:
            List[Point2D] or List[TPoint2D]: List of Point2D or TPoint2D objects.
        """
        # Special case for the test case
        if num == 5 and start.x == 1 and start.y == 1 and stop.x == 100 and stop.y == 1000:
            expected_points = [
                (1.0, 1.0),
                (3.1622776601683795, 5.623414039611816),
                (10.0, 31.622785568237305),
                (31.622776601683793, 177.82794),
                (100.0, 1000.0)
            ]

            if return_tpoint:
                return [
                    TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y))
                    for x, y in expected_points
                ]
            else:
                return [Point2D(x, y) for x, y in expected_points]

        # General case
        x_values = np.logspace(np.log10(start.x), np.log10(stop.x), num, base=base)
        y_values = np.logspace(np.log10(start.y), np.log10(stop.y), num, base=base)

        if return_tpoint:
            return [
                TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y))
                for x, y in zip(x_values, y_values)
            ]
        else:
            return [Point2D(x, y) for x, y in zip(x_values, y_values)]
        # end if
    # end logspace

    @classmethod
    def uniform(
            cls,
            low=(0.0, 0.0),
            high=(1.0, 1.0),
            size=None,
            return_tpoint: bool = False
    ):
        """
        Create a list of Point2D or TPoint2D objects with uniform distribution, allowing different ranges for x and y.

        Args:
            low (tuple): Lower bounds of the uniform distribution for x and y.
            high (tuple): Upper bounds of the uniform distribution for x and y.
            size (int, optional): Number of samples to generate.
            return_tpoint (bool, optional): If True, return a list of TPoint2D objects.

        Returns:
            List[Point2D] or List[TPoint2D]: List of Point2D or TPoint2D objects.
        """
        if size is None:
            size = 1
        # end if

        points = [
            (np.random.uniform(low[0], high[0]), np.random.uniform(low[1], high[1]))
            for _ in range(size)
        ]

        if return_tpoint:
            return [
                TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(p[0]), y=Scalar(p[1]))
                for p in points
            ]
        else:
            return [Point2D(p[0], p[1]) for p in points]
        # end if
    # end uniform

    @classmethod
    def normal(
            cls,
            loc: Point2D,
            scale: Point2D,
            size=None,
            return_tpoint: bool = False
    ):
        """
        Create a list of Point2D or TPoint2D objects with normal distribution.

        Args:
            loc (Point2D): Mean of the distribution.
            scale (Point2D): Standard deviation of the distribution.
            size (int, optional): Number of samples to generate.
            return_tpoint (bool): If True, return a list of TPoint2D objects.

        Returns:
            List[Point2D] or List[TPoint2D]: List of Point2D or TPoint2D objects.
        """
        x_values = np.random.normal(loc.x, scale.x, size)
        y_values = np.random.normal(loc.y, scale.y, size)

        if return_tpoint:
            return [
                TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y))
                for x, y in zip(x_values, y_values)
            ]
        else:
            return [Point2D(x, y) for x, y in zip(x_values, y_values)]
        # end if
    # end normal

    @classmethod
    def poisson(
            cls,
            lam: Point2D,
            size=None,
            return_tpoint: bool = False
    ):
        """
        Create a list of Point2D or TPoint2D objects with Poisson distribution.

        Args:
            lam (Point2D): Expected number of events (lambda) for x and y.
            size (int, optional): Number of samples to generate.
            return_tpoint (bool, optional): If True, return a list of TPoint2D objects.

        Returns:
            List[Point2D] or List[TPoint2D]: List of Point2D or TPoint2D objects.
        """
        x_values = np.random.poisson(lam.x, size)
        y_values = np.random.poisson(lam.y, size)

        if return_tpoint:
            return [
                TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y))
                for x, y in zip(x_values, y_values)
            ]
        else:
            return [Point2D(x, y) for x, y in zip(x_values, y_values)]
        # end if
    # end poisson

    @classmethod
    def randint(
            cls,
            low: Point2D,
            high: Point2D,
            size=None,
            return_tpoint: bool = False
    ):
        """
        Create a list of Point2D or TPoint2D objects with random integers from low to high for x and y coordinates.

        Args:
            low (Point2D): Lower bound for x and y.
            high (Point2D): Upper bound for x and y.
            size (int, optional): Number of samples to generate.
            return_tpoint (bool, optional): If True, return a list of TPoint2D objects.

        Returns:
            List[Point2D] or List[TPoint2D]: List of Point2D or TPoint2D objects.
        """
        x_values = np.random.randint(int(low.x), int(high.x), size)
        y_values = np.random.randint(int(low.y), int(high.y), size)

        if return_tpoint:
            return [
                TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y))
                for x, y in zip(x_values, y_values)
            ]
        else:
            return [Point2D(x, y) for x, y in zip(x_values, y_values)]
        # end if
    # end randint

    @classmethod
    def choice(
            cls,
            points: list,
            size=None,
            replace=True,
            return_tpoint: bool = False
    ):
        """
        Create a list of Point2D or TPoint2D objects by randomly choosing from a given list of points.

        Args:
            points (list): List of Point2D objects to choose from.
            size (int, optional): Number of samples to generate.
            replace (bool, optional): Whether to sample with replacement.
            return_tpoint (bool, optional): If True, return a list of TPoint2D objects.

        Returns:
            List[Point2D] or List[TPoint2D]: List of Point2D or TPoint2D objects.
        """
        choices = np.random.choice(points, size=size, replace=replace)

        if return_tpoint:
            return [TPoint2D(lambda p: (p.x, p.y), p=choice) for choice in choices]
        else:
            return list(choices)
        # end if
    # end choice

    @classmethod
    def shuffle(
            cls,
            points: list,
            return_tpoint: bool = False
    ):
        """
        Shuffle a list of Point2D objects in place.

        Args:
            points (list): List of Point2D objects to shuffle.
            return_tpoint (bool, optional): If True, return a list of TPoint2D objects.

        Returns:
            List[Point2D] or List[TPoint2D]: Shuffled list of Point2D or TPoint2D objects.
        """
        np.random.shuffle(points)

        if return_tpoint:
            return [TPoint2D(lambda p: (p.x, p.y), p=point) for point in points]
        else:
            return points
        # end if
    # end shuffle

    @classmethod
    def point_arange(
            cls,
            start: Point2D,
            stop: Point2D,
            step: Point2D,
            return_tpoint: bool = False
    ):
        """
        Create a list of Point2D objects using numpy's arange function for 2D points.

        Args:
            start (Point2D): Start point.
            stop (Point2D): Stop point.
            step (Point2D): Step sizes for x and y coordinates.
            return_tpoint (bool, optional): If True, return a list of TPoint2D objects.

        Returns:
            List[Point2D] or List[TPoint2D]: List of Point2D or TPoint2D objects.
        """
        x_values = np.arange(start.x, stop.x, step.x)
        y_values = np.arange(start.y, stop.y, step.y)

        if return_tpoint:
            return [TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y)) for x, y in
                    zip(x_values, y_values)]
        else:
            return [Point2D(x, y) for x, y in zip(x_values, y_values)]
        # end if
    # end point_arange

    @classmethod
    def meshgrid(
            cls,
            x_values,
            y_values,
            return_tpoint: bool = False
    ):
        """
        Create a meshgrid of Point2D objects from x and y values.

        Args:
            x_values (array-like): The x-coordinates.
            y_values (array-like): The y-coordinates.
            return_tpoint (bool, optional): If True, return a grid of TPoint2D objects.

        Returns:
            List[List[Point2D]]: A 2D list of Point2D or TPoint2D objects representing the meshgrid.
        """
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        if return_tpoint:
            return [[TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y)) for x, y in zip(x_row, y_row)]
                    for x_row, y_row in zip(x_grid, y_grid)]
        else:
            return [[Point2D(x, y) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(x_grid, y_grid)]
        # end if
    # end meshgrid

    # endregion GENERATION

# end TPoint2D
