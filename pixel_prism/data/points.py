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
from typing import Any, Union
import numpy as np
from pixel_prism.animate import MovableMixin
from .data import Data
from .scalar import Scalar, TScalar
from .eventmixin import EventMixin
from .events import ObjectChangedEvent


# A generic point
class Point(Data, EventMixin, MovableMixin):
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
        EventMixin.__init__(self)
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
            dtype (type): Data type of the point
        """
        super().__init__(readonly=readonly)
        self._pos = np.array([x, y], dtype=dtype)

        # Movable
        self.start_position = None

        # List of event listeners (per events)
        self.event_listeners = {
            "on_change": [] if on_change is None else [on_change]
        }
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

    # Movable position
    @property
    def movable_position(self) -> Any:
        """
        Get the position of the object.

        Returns:
            any: Position of the object
        """
        return self.pos
    # end movable_position

    @movable_position.setter
    def movable_position(self, value: Any):
        """
        Set the position of the object.

        Args:
            value (any): Position of the object
        """
        self.pos = value
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
        self._pos[0] = x.value if type(x) is Scalar else x
        self._pos[1] = y.value if type(y) is Scalar else y
        self.dispatch_event("on_change", ObjectChangedEvent(self, x=self._pos[0], y=self._pos[1]))
    # end set

    def get(self):
        """
        Get the coordinates of the point.

        Returns:
            np.array: Array containing the X and Y coordinates of the point
        """
        return self._pos[0], self._pos[1]
    # end get

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

    # region MOVABLE

    # Initialize position
    def init_move(
            self,
            relative: bool = False
    ):
        """
        Initialize the move animation.

        Args:
            relative (bool): If the move is relative to the current position
        """
        self.start_position = None
    # end init_move

    # Start animation
    def start_move(
            self,
            start_value: Any,
            relative: bool = False,
            *args,
            **kwargs
    ):
        """
        Start the move animation.

        Args:
            start_value (any): The start position of the object
            relative (bool): If the move is relative to the current position
        """
        self.start_position = self.pos.copy()
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
        self.movable_position = self.start_position * (1 - interpolated_t) + end_value.movable_position * interpolated_t
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
        pass
    # end end_move

    # Finish animation
    def finish_move(self, relative: bool = False, *args, **kwargs):
        """
        Finish the move animation.
        """
        pass
    # end finish_move

    # endregion MOVABLE

    # region OVERRIDE

    # Return a string representation of the point.
    def __str__(self):
        """
        Return a string representation of the point.
        """
        return f"Point2D(x={self.x}, y={self.y}, closed={self.data_closed})"
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
            return add_t(self, other)
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
            return sub_t(self, other)
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
            return sub_t(other, self)
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
            return mul_t(self, other)
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
            return mul_t(self, other)
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

    # region CLASS_METHODS

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

    # endregion CLASS_METHODS

# end Point2D


class Point3D(Point):
    """
    A class to represent a point in 3D space.
    """

    def __init__(self, x=0, y=0, z=0, readonly: bool = False, dtype=np.float32):
        """
        Initialize the point with its coordinates.

        Args:
            x (float): X-coordinate of the point
            y (float): Y-coordinate of the point
            z (float): Z-coordinate of the poin
            readonly (bool): If the point is read-only
            dtype (type): Data type of the point
        """
        super().__init__(readonly=readonly)
        self.pos = np.array([x, y, z], dtype=dtype)
    # end __init__

    # region PROPERTIES

    @property
    def x(self):
        """
        Get the X-coordinate of the point.

        Returns:
            float: X-coordinate of the point
        """
        return self.pos[0]
    # end x

    @x.setter
    def x(self, value):
        """
        Set the X-coordinate of the point.
        """
        self.pos[0] = value
    # end x

    @property
    def y(self):
        """
        Get the Y-coordinate of the point.

        Returns:
            float: Y-coordinate of the point
        """
        return self.pos[1]
    # end y

    @y.setter
    def y(self, value):
        """
        Set the Y-coordinate of the point.
        """
        self.pos[1] = value
    # end y

    @property
    def z(self):
        """
        Get the Z-coordinate of the point.

        Returns:
            float: Z-coordinate of the point
        """
        return self.pos[2]
    # end z

    @z.setter
    def z(self, value):
        """
        Set the Z-coordinate of the point.
        """
        self.pos[2] = value
    # end z

    # endregion PROPERTIES

    # region PUBLIC

    def set(self, pos):
        """
        Set the coordinates of the point.

        Args:
            pos (np.array): Tuple containing the X, Y, and Z coordinates of the point
        """
        self.check_closed()
        self.pos = pos
    # end set

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

    # endregion OVERRIDE

# end Point3D


# TPoint2D class
class TPoint2D(Point2D):
    """
    A class that tracks transformations applied to a Point2D and updates dynamically.
    """

    def __init__(self, transform_func, on_change=None, **points):
        """
        Initialize the tracked point.

        Args:
            transform_func (function): The transformation function to apply to the point
            on_change (function): Function to call when the point changes
            **points (Any): Points to track
        """
        self._points = points
        self._transform_func = transform_func

        # Initialize with the transformed point's position
        x, y = self._transform_func(**self._points)
        super().__init__(x, y)

        # Listen to changes in the original point
        if on_change is not None:
            for point in self._points.values():
                point.add_event_listener("on_change", on_change)
                point.add_event_listener("on_change", self._on_point_changed)
            # end for
        # end if
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

    def add_event_listener(self, event_name, listener):
        """
        Add an event listener to the data object.

        Args:
            event_name (str): Event to listen for
            listener (function): Listener function
        """
        for point in self.points.values():
            point.add_event_listener(event_name, listener)
        # end for
    # end add_event_listener

    def remove_event_listener(self, event_name, listener):
        """
        Remove an event listener from the data object.

        Args:
            event_name (str): Event to remove listener from
            listener (function): Listener function to remove
        """
        # Unregister from all sources
        for point in self.points.values():
            point.remove_event_listener(event_name, listener)
        # end for
    # end remove_event_listener

    # endregion PUBLIC

    # region EVENT

    def _on_point_changed(self, event):
        """
        Update the point when a source point changes.
        """
        x, y = self.get()
        self._pos[0] = x
        self._pos[1] = y
        self.dispatch_event("on_change", ObjectChangedEvent(self, x=self.x, y=self.y))
    # end _on_point_changed

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
            return add_t(self, other)
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
            return sub_t(self, other)
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
            return sub_t(other, self)
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

# end TPoint2D


# Basic TPoint2D (just return value of a point)
def tpoint2d(
        point: Union[Point2D, TPoint2D, tuple]
):
    """
    Create a TPoint2D that represents a point.
    """
    if isinstance(point, Point2D):
        return TPoint2D(lambda p: (p.x, p.y), p=point)
    elif isinstance(point, tuple):
        return TPoint2D(lambda p: (point[0], point[1]))
    else:
        return point
    # end if
# end tpoint2d


# region OPERATORS


# Function to create a new tracked point
def add_t(
        point: Union[Point2D, TPoint2D],
        delta: Union[Point2D, TPoint2D]
):
    """
    Create a TPoint2D that represents point + delta.

    Args:
        point (Union[Point2D, TPoint2D]): Point to add to.
        delta (Union[Point2D, TPoint2D]): Point to add.
    """
    return TPoint2D(lambda p, d: (p.x + d.x, p.y + d.y), p=point, d=delta)
# end add_t


def sub_t(
        point: Union[Point2D, TPoint2D],
        delta: Union[Point2D, TPoint2D]
):
    """
    Create a TPoint2D that represents point - delta.

    Args:
        point (Union[Point2D, TPoint2D]): Point to subtract from.
        delta (Union[Point2D, TPoint2D]): Point to subtract.
    """
    return TPoint2D(lambda p: (p.x - delta.x, p.y - delta.y), p=point)
# end sub_t


def mul_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D],
):
    """
    Create a TPoint2D that represents point * scalar.

    Args:
        point1 (Union[Point2D, TPoint2D]): Point to multiply.
        point2 (Union[Point2D, TPoint2D]): Point to multiply by.
    """
    if isinstance(point1, TPoint2D) or isinstance(point2, TPoint2D):
        return TPoint2D(lambda p1, p2: (p1.x * p2.x, p1.y * p2.y), p1=point1, p2=point2)
    else:
        return Point2D(point1.x * point2.x, point1.y * point2.y)
    # end if
# end mul_t

def scalar_mul_t(
        point: [Point2D, TPoint2D],
        scalar: [Scalar, TScalar, float, int]
) -> TPoint2D:
    """
    Multiply a matrix by a scalar.

    Args:
        point (Point2D): Point to multiply
        scalar (Scalar/TScalar): Scalar
    """
    if isinstance(scalar, (int, float)):
        scalar = Scalar(scalar)
    # end if
    return TPoint2D(lambda p, s: (p.x * s.value, p.y * s.value), p=point, s=scalar)
# end scalar_mul_t

def div_t(
        point: Union[Point2D, TPoint2D],
        scalar: Union[Scalar, float]
):
    """
    Create a TPoint2D that represents point / scalar.

    Args:
        point (Union[Point2D, TPoint2D]): Point to divide.
        scalar (Union[Scalar, float]): Scalar to divide by.
    """
    if isinstance(scalar, Scalar):
        return TPoint2D(lambda p, s: (p.x / s.value, p.y / s.value), p=point, s=scalar)
    else:
        return TPoint2D(lambda p: (p.x / scalar, p.y / scalar), p=point)
    # end if
# end div_t


def neg_t(
        point: Union[Point2D, TPoint2D]
):
    """
    Create a TPoint2D that represents -point (negation).
    """
    return TPoint2D(lambda p: (-p.x, -p.y), p=point)
# end neg_t


def abs_t(
        point: Union[Point2D, TPoint2D]
):
    """
    Create a TPoint2D that represents the absolute value of the point.
    """
    return TPoint2D(lambda p: (abs(p.x), abs(p.y)), p=point)
# end abs_t


# endregion OPERATORS


# region METHODS


def round_t(
        point: Union[Point2D, TPoint2D],
        ndigits: int = 0
):
    """
    Create a TPoint2D that represents the rounded value of the point.
    """
    return TPoint2D(lambda p: (round(p.x, ndigits=ndigits), round(p.y, ndigits=ndigits)), p=point)
# end round_t


def rotate_t(
        point: Union[Point2D, TPoint2D],
        angle: Union[Scalar, float],
        center: Union[Point2D, TPoint2D] = None
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
# end rotate_t


def scale_t(
        point: Union[Point2D, TPoint2D],
        scale: Union[Scalar, float],
        center: Union[Point2D, TPoint2D] = None
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
# end scale_t


def dot_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the dot product of two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda p1, p2: p1.x * p2.x + p1.y * p2.y, p1=point1, p2=point2)
# end dot_t


def cross_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the cross product of two points in 2D.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda p1, p2: p1.x * p2.y - p1.y * p2.x, p1=point1, p2=point2)
# end cross_t


def norm_t(
        point: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the norm (magnitude) of a point.

    Args:
        point (Point2D): The point.
    """
    return TScalar(lambda p: math.sqrt(point.x ** 2 + point.y ** 2), p=point)
# end norm_t


def normalize_t(
        point: Union[Point2D, TPoint2D]
):
    """
    Create a TPoint2D representing the normalized (unit vector) of a point.

    Args:
        point (Point2D): The point to normalize.
    """
    norm = norm_t(point)
    return TPoint2D(lambda p: (p.x / norm.value, p.y / norm.value), p=point)
# end normalize_t


def angle_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the angle between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    dot = dot_t(point1, point2)
    norm1 = norm_t(point1)
    norm2 = norm_t(point2)
    return TScalar(lambda p1, p2: math.acos(dot.value / (norm1.value * norm2.value)), p1=point1, p2=point2)
# end angle_t


# endregion METHODS


# region DISTANCES


def distance_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the Euclidean distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda p1, p2: math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2), p1=point1, p2=point2)
# end distance_t


def distance_squared_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the squared Euclidean distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda p1, p2: (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2, p1=point1, p2=point2)
# end distance_squared_t


def distance_manhattan_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the Manhattan distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda p1, p2: abs(p1.x - point2.x) + abs(p1.y - p2.y), p1=point1, p2=point2)
# end distance_manhattan_t


def distance_chebyshev_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the Chebyshev distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda p1, p2: max(abs(point1.x - point2.x), abs(point1.y - point2.y)), p1=point1, p2=point2)
# end distance_chebyshev_t


def distance_canberra_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
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
# end distance_canberra_t


def distance_minkowski_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D],
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
# end distance_minkowski_t


def distance_hamming_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the Hamming distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda p1, p2: int(p1.x != p2.x) + int(p1.y != p2.y), p1=point1, p2=point2)
# end distance_hamming_t


def distance_jaccard_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
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
# end distance_jaccard_t


def distance_braycurtis_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
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
# end distance_braycurtis_t


def distance_cosine_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the cosine distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    dot = dot_t(point1, point2)
    norm1 = norm_t(point1)
    norm2 = norm_t(point2)
    return TScalar(
        lambda p1, p2: 1 - (dot.value / (norm1.value * norm2.value)) if norm1.value != 0 and norm2.value != 0 else 1,
        p1=point1,
        p2=point2
    )
# end distance_cosine_t


def distance_correlation_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
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
# end distance_correlation_t


def distance_euclidean_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the Euclidean distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda p1, p2: np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2), p1=point1, p2=point2)
# end distance_euclidean_t


def distance_mahalanobis_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D],
        cov_matrix
):
    """
    Create a TScalar representing the Mahalanobis distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
        cov_matrix (np.ndarray): Covariance matrix of the dataset.
    """
    return TScalar(
        lambda p1, p2, cov: np.sqrt(np.array([p1.x - p2.x, p1.y - p2.y]).T @ np.linalg.inv(cov) @ np.array([p1.x - p2.x, p1.y - p2.y])),
        p1=point1,
        p2=point2,
        cov=cov_matrix
    )
# end distance_mahalanobis_t


def distance_seuclidean_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D],
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
# end distance_seuclidean_t


def distance_sqeuclidean_t(
        point1: Union[Point2D, TPoint2D],
        point2: Union[Point2D, TPoint2D]
):
    """
    Create a TScalar representing the squared Euclidean distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda p1, p2: (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2, p1=point1, p2=point2)
# end distance_sqeuclidean_t


# endregion DISTANCES


# region GENERATION


def point_range(
        start: Union[Point2D, TPoint2D],
        stop: Union[Point2D, TPoint2D],
        step: Union[Point2D, TPoint2D],
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


def linspace(
        start: Union[Point2D, TPoint2D],
        stop: Union[Point2D, TPoint2D],
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


def logspace(
        start: Union[Point2D, TPoint2D],
        stop: Union[Point2D, TPoint2D],
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


def uniform(
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


def normal(
        loc: Union[Point2D, TPoint2D],
        scale: Union[Point2D, TPoint2D],
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


def poisson(
        lam: Union[Point2D, TPoint2D],
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


def randint(
        low: Union[Point2D, TPoint2D],
        high: Union[Point2D, TPoint2D],
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
    x_values = np.random.randint(low.x, high.x, size)
    y_values = np.random.randint(low.y, high.y, size)

    if return_tpoint:
        return [
            TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y))
            for x, y in zip(x_values, y_values)
        ]
    else:
        return [Point2D(x, y) for x, y in zip(x_values, y_values)]
    # end if
# end randint


def choice(
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


def shuffle(
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


def point_arange(
        start: Union[Point2D, TPoint2D],
        stop: Union[Point2D, TPoint2D],
        step: Union[Point2D, TPoint2D],
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
        return [TPoint2D(lambda x, y: (x.value, y.value), x=Scalar(x), y=Scalar(y)) for x, y in zip(x_values, y_values)]
    else:
        return [Point2D(x, y) for x, y in zip(x_values, y_values)]
    # end if
# end point_arange


def meshgrid(
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

