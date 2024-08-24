#
# This file contains the Point2D class, which is a simple class that
# represents a point in 2D space.

# Imports
import numpy as np
from pixel_prism.animate.able import MovableMixin

from .data import Data
from .scalar import Scalar
from .eventmixin import EventMixin


# A generic point
class Point(Data, EventMixin, MovableMixin):
    """
    A generic point class.
    """
    pass
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
            dtype=np.float32
    ):
        """
        Initialize the point with its coordinates.

        Args:
            x (float): X-coordinate of the point
            y (float): Y-coordinate of the point
        """
        super().__init__()
        self._pos = np.array([x, y], dtype=dtype)

        # List of event listeners (per events)
        self.event_listeners = {
            "on_change": [] if on_change is None else [on_change]
        }
    # end __init__

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

    def set(self, x, y):
        """
        Set the coordinates of the point.

        Args:
            x (float or Scalar): X-coordinate of the point
            y (float or Scalar): Y-coordinate of the point
        """
        self._pos[0] = x.value if type(x) is Scalar else x
        self._pos[1] = y.value if type(y) is Scalar else y
        self.dispatch_event("on_change", x=self._pos[0], y=self._pos[1])
    # end set

    def get(self):
        """
        Get the coordinates of the point.

        Returns:
            np.array: Array containing the X and Y coordinates of the point
        """
        return self._pos[0], self._pos[1]
    # end get

    def copy(self):
        """
        Return a copy of the point.
        """
        return Point2D(x=self.x, y=self.y, dtype=self._pos.dtype)
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

    # region OVERRIDE

    # Return a string representation of the point.
    def __str__(self):
        """
        Return a string representation of the point.
        """
        return f"Point2D(x={self.x}, y={self.y})"
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
        if isinstance(other, Point2D):
            return Point2D(self.x + other.x, self.y + other.y)
        elif isinstance(other, (int, float)):
            return Point2D(self.x + other, self.y + other)
        elif isinstance(other, tuple):
            return Point2D(self.x + other[0], self.y + other[1])
        else:
            return NotImplemented
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
        if isinstance(other, Point2D):
            return Point2D(self.x - other.x, self.y - other.y)
        elif isinstance(other, (int, float)):
            return Point2D(self.x - other, self.y - other)
        elif isinstance(other, tuple):
            return Point2D(self.x - other[0], self.y - other[1])
        else:
            return NotImplemented
        # end if
    # end __sub__

    def __rsub__(self, other):
        """
        Subtract two points.

        Args:
            other (Point2D): Point to subtract from this point or scalar value to subtract from the point.
        """
        return self.__sub__(other)
    # end __rsub__

    def __mul__(self, other):
        """
        Multiply the point by a scalar value.

        Args:
            other (int, float): Scalar value to multiply the point by.
        """
        if isinstance(other, (int, float)):
            return Point2D(self.x * other, self.y * other)
        elif isinstance(other, Point2D):
            return Point2D(self.x * other.x, self.y * other.y)
        elif isinstance(other, tuple):
            return Point2D(self.x * other[0], self.y * other[1])
        else:
            return NotImplemented
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
        if isinstance(other, (int, float)):
            return Point2D(self.x / other, self.y / other)
        elif isinstance(other, Point2D):
            return Point2D(self.x / other.x, self.y / other.y)
        elif isinstance(other, tuple):
            return Point2D(self.x / other[0], self.y / other[1])
        else:
            return NotImplemented
        # end if
    # end __truediv__

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

    def __init__(self, x=0, y=0, z=0, dtype=np.float32):
        """
        Initialize the point with its coordinates.

        Args:
            x (float): X-coordinate of the point
            y (float): Y-coordinate of the point
            z (float): Z-coordinate of the point
        """
        super().__init__()
        self.pos = np.array([x, y, z], dtype=dtype)
    # end __init__

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

    def set(self, pos):
        """
        Set the coordinates of the point.

        Args:
            pos (np.array): Tuple containing the X, Y, and Z coordinates of the point
        """
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

    # Return a string representation of the point.
    def __repr__(self):
        """
        Return a string representation of the point.
        """
        return f"Point3D(x={self.x}, y={self.y}, z={self.z})"
    # end __repr__

# end Point3D

