#
# This file contains the Point2D class, which is a simple class that
# represents a point in 2D space.
import math
from typing import Any

# Imports
import numpy as np
from pixel_prism.animate.able import MovableMixin

from .data import Data
from .scalar import Scalar
from .eventmixin import EventMixin
from .events import ObjectChangedEvent


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
            origin = Point2D.null()
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


# TPoint2D class
class TPoint2D(Point2D):
    """
    A class that tracks transformations applied to a Point2D and updates dynamically.
    """

    def __init__(self, point: Point2D, transform_func):
        self.original_point = point
        self.transform_func = transform_func

        # Initialize with the transformed point's position
        transformed_point = self.transform_func(self.original_point)
        super().__init__(transformed_point.x, transformed_point.y)

        # Subscribe to changes on the original point
        self.original_point.add_event_listener("on_change", self.update_position)
    # end __init__

    # region PUBLIC

    def update_position(self, _=None):
        """
        Update the position based on the original point's new position.
        """
        transformed_point = self.transform_func(self.original_point)
        super().set(transformed_point.x, transformed_point.y)
    # end update_position

    # endregion PUBLIC

# end TPoint2D


# Function to create a new tracked point
def add_t(point: Point2D, delta: Point2D):
    """
    Create a TPoint2D that represents point + delta.
    """
    return TPoint2D(point, lambda p: Point2D(p.x + delta.x, p.y + delta.y))
# end add_t


def sub_t(point: Point2D, delta: Point2D):
    """
    Create a TPoint2D that represents point - delta.
    """
    return TPoint2D(point, lambda p: Point2D(p.x - delta.x, p.y - delta.y))
# end sub_t


def mul_t(point: Point2D, scalar: float):
    """
    Create a TPoint2D that represents point * scalar.
    """
    return TPoint2D(point, lambda p: Point2D(p.x * scalar, p.y * scalar))
# end mul_t


def div_t(point: Point2D, scalar: float):
    """
    Create a TPoint2D that represents point / scalar.
    """
    return TPoint2D(point, lambda p: Point2D(p.x / scalar, p.y / scalar))
# end div_t


def neg_t(point: Point2D):
    """
    Create a TPoint2D that represents -point (negation).
    """
    return TPoint2D(point, lambda p: Point2D(-p.x, -p.y))
# end neg_t


def abs_t(point: Point2D):
    """
    Create a TPoint2D that represents the absolute value of the point.
    """
    return TPoint2D(point, lambda p: Point2D(abs(p.x), abs(p.y)))
# end abs_t


def round_t(point: Point2D, ndigits=0):
    """
    Create a TPoint2D that represents the rounded value of the point.
    """
    return TPoint2D(point, lambda p: Point2D(round(p.x, ndigits=ndigits), round(p.y, ndigits=ndigits)))
# end round_t


def rotate_t(point: Point2D, angle: Scalar, center: Point2D = None):
    """
    Create a TPoint2D that represents the point rotated around another point by a given angle.

    Args:
        point (Point2D): The point to rotate.
        angle (Scalar): The angle of rotation (in radians).
        center (Point2D): The center of rotation. If None, rotate around the origin.
    """
    if center is None:
        center = Point2D(0, 0)

    return TPoint2D(point, lambda p: Point2D(
        center.x + (p.x - center.x) * math.cos(angle.value) - (p.y - center.y) * math.sin(angle.value),
        center.y + (p.x - center.x) * math.cos(angle.value) + (p.y - center.y) * math.cos(angle.value)
    ))
# end rotate_t


def scale_t(point: Point2D, scale: Scalar, center: Point2D = None):
    """
    Create a TPoint2D that represents the point scaled away from another point by a given scale factor.

    Args:
        point (Point2D): The point to scale.
        scale (Scalar): The scale factor.
        center (Point2D): The center of scaling. If None, scale from the origin.
    """
    if center is None:
        center = Point2D(0, 0)

    return TPoint2D(point, lambda p: Point2D(
        center.x + (p.x - center.x) * scale.value,
        center.y + (p.y - center.y) * scale.value
    ))
# end scale_t


def dot_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the dot product of two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda: point1.x * point2.x + point1.y * point2.y)
# end dot_t


def cross_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the cross product of two points in 2D.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda: point1.x * point2.y - point1.y * point2.x)
# end cross_t


def norm_t(point: Point2D):
    """
    Create a TScalar representing the norm (magnitude) of a point.

    Args:
        point (Point2D): The point.
    """
    return TScalar(lambda: math.sqrt(point.x ** 2 + point.y ** 2))
# end norm_t


def normalize_t(point: Point2D):
    """
    Create a TPoint2D representing the normalized (unit vector) of a point.

    Args:
        point (Point2D): The point to normalize.
    """
    norm = norm_t(point)
    return TPoint2D(point, lambda p: Point2D(p.x / norm.value, p.y / norm.value))
# end normalize_t


def angle_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the angle between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    dot = dot_t(point1, point2)
    norm1 = norm_t(point1)
    norm2 = norm_t(point2)
    return TScalar(lambda: math.acos(dot.value / (norm1.value * norm2.value)))
# end angle_t


def distance_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the Euclidean distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda: math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2))
# end distance_t


def distance_squared_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the squared Euclidean distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda: (point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
# end distance_squared_t


def distance_manhattan_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the Manhattan distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda: abs(point1.x - point2.x) + abs(point1.y - point2.y))
# end distance_manhattan_t


def distance_chebyshev_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the Chebyshev distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda: max(abs(point1.x - point2.x), abs(point1.y - point2.y)))
# end distance_chebyshev_t


def distance_canberra_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the Canberra distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda: abs(point1.x - point2.x) / (abs(point1.x) + abs(point2.x)) +
                             abs(point1.y - point2.y) / (abs(point1.y) + abs(point2.y)))
# end distance_canberra_t


def distance_minkowski_t(point1: Point2D, point2: Point2D, p: float):
    """
    Create a TScalar representing the Minkowski distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
        p (float): The order of the Minkowski distance (p=1 is Manhattan, p=2 is Euclidean).
    """
    return TScalar(lambda: ((abs(point1.x - point2.x) ** p + abs(point1.y - point2.y) ** p) ** (1 / p)))
# end distance_minkowski_t


def distance_hamming_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the Hamming distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda: int(point1.x != point2.x) + int(point1.y != point2.y))
# end distance_hamming_t


def distance_jaccard_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the Jaccard distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    intersection = min(point1.x, point2.x) + min(point1.y, point2.y)
    union = max(point1.x, point2.x) + max(point1.y, point2.y)
    return TScalar(lambda: 1 - intersection / union if union != 0 else 0)
# end distance_jaccard_t


def distance_braycurtis_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the Bray-Curtis distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    numerator = abs(point1.x - point2.x) + abs(point1.y - point2.y)
    denominator = abs(point1.x + point2.x) + abs(point1.y + point2.y)
    return TScalar(lambda: numerator / denominator if denominator != 0 else 0)
# end distance_braycurtis_t


def distance_cosine_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the cosine distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    dot = dot_t(point1, point2)
    norm1 = norm_t(point1)
    norm2 = norm_t(point2)
    return TScalar(lambda: 1 - (dot.value / (norm1.value * norm2.value)) if norm1.value != 0 and norm2.value != 0 else 1)
# end distance_cosine_t


def distance_correlation_t(point1: Point2D, point2: Point2D):
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
    return TScalar(lambda: 1 - (numerator / denominator) if denominator != 0 else 1)
# end distance_correlation_t


def distance_haversine_t(point1: Point2D, point2: Point2D, radius=6371.0):
    """
    Create a TScalar representing the Haversine distance between two geographic points.

    Args:
        point1 (Point2D): The first point (latitude, longitude).
        point2 (Point2D): The second point (latitude, longitude).
        radius (float): Radius of the sphere (default is Earth's radius in km).
    """
    lat1, lon1 = math.radians(point1.y), math.radians(point1.x)
    lat2, lon2 = math.radians(point2.y), math.radians(point2.x)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return TScalar(lambda: radius * c)
# end distance_haversine_t


def distance_euclidean_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the Euclidean distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda: np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2))
# end distance_euclidean_t


def distance_mahalanobis_t(point1: Point2D, point2: Point2D, cov_matrix):
    """
    Create a TScalar representing the Mahalanobis distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
        cov_matrix (np.ndarray): Covariance matrix of the dataset.
    """
    delta = np.array([point1.x - point2.x, point1.y - point2.y])
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    return TScalar(lambda: np.sqrt(delta.T @ inv_cov_matrix @ delta))
# end distance_mahalanobis_t


def distance_seuclidean_t(point1: Point2D, point2: Point2D, std_devs):
    """
    Create a TScalar representing the standardized Euclidean distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
        std_devs (np.ndarray): Standard deviations of the dimensions.
    """
    delta = np.array([point1.x - point2.x, point1.y - point2.y])
    return TScalar(lambda: np.sqrt(np.sum((delta / std_devs) ** 2)))
# end distance_seuclidean_t


def distance_sqeuclidean_t(point1: Point2D, point2: Point2D):
    """
    Create a TScalar representing the squared Euclidean distance between two points.

    Args:
        point1 (Point2D): The first point.
        point2 (Point2D): The second point.
    """
    return TScalar(lambda: (point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
# end distance_sqeuclidean_t









