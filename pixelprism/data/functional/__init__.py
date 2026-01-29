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

"""
Functional operations for pixel_prism.math_old types.

This module provides functional versions of operations like add, sub, mul, div, etc.
for various math_old types like TScalar, TMatrix2D, TPoint2D, etc.

Example:
<<<<<<<< HEAD:pixelprism/math_old/functional/__init__.py
    >>> import pixelprism.math_old.functional as F
========
    >>> import pixelprism.data.functional as F
>>>>>>>> dev:pixelprism/data/functional/__init__.py
    >>> a = Scalar(1)
    >>> b = Scalar(2)
    >>> c = F.add(a, b)  # c is a TScalar with value 3
"""

from typing import Union, Any

from ..scalar import Scalar, TScalar
from ..matrices import Matrix2D, TMatrix2D
from ..points import Point2D, TPoint2D, Point3D


def add(a: Any, b: Any) -> Any:
    """
    Add two objects.

    Args:
        a: First object
        b: Second object

    Returns:
        The result of adding a and b, with the appropriate type
    """
    # Scalar addition
    if isinstance(a, (Scalar, int, float)) and isinstance(b, (Scalar, int, float)):
        return TScalar.add(a, b)
    # Matrix addition
    elif isinstance(a, Matrix2D) and isinstance(b, Matrix2D):
        return TMatrix2D.add(a, b)
    # Point2D addition
    elif isinstance(a, Point2D) and isinstance(b, Point2D):
        return TPoint2D.add(a, b)
    # Point3D addition is not supported yet
    elif isinstance(a, Point3D) and isinstance(b, Point3D):
        raise NotImplementedError("Addition for Point3D is not implemented yet")
    else:
        raise TypeError(f"Unsupported types for addition: {type(a)} and {type(b)}")
    # end if
# end add


def sub(a: Any, b: Any) -> Any:
    """
    Subtract b from a.

    Args:
        a: First object
        b: Second object

    Returns:
        The result of subtracting b from a, with the appropriate type
    """
    # Scalar subtraction
    if isinstance(a, (Scalar, int, float)) and isinstance(b, (Scalar, int, float)):
        return TScalar.sub(a, b)

    # Matrix subtraction
    elif isinstance(a, Matrix2D) and isinstance(b, Matrix2D):
        return TMatrix2D.sub(a, b)

    # Point2D subtraction
    elif isinstance(a, Point2D) and isinstance(b, Point2D):
        return TPoint2D.sub(a, b)

    # Point3D subtraction is not supported yet
    elif isinstance(a, Point3D) and isinstance(b, Point3D):
        raise NotImplementedError("Subtraction for Point3D is not implemented yet")

    else:
        raise TypeError(f"Unsupported types for subtraction: {type(a)} and {type(b)}")


def mul(a: Any, b: Any) -> Any:
    """
    Multiply two objects.

    Args:
        a: First object
        b: Second object

    Returns:
        The result of multiplying a and b, with the appropriate type
    """
    # Scalar multiplication
    if isinstance(a, (Scalar, int, float)) and isinstance(b, (Scalar, int, float)):
        return TScalar.mul(a, b)

    # Matrix multiplication
    elif isinstance(a, Matrix2D) and isinstance(b, Matrix2D):
        return TMatrix2D.mul(a, b)

    # Point2D multiplication
    elif isinstance(a, Point2D) and isinstance(b, Point2D):
        return TPoint2D.mul(a, b)

    # Point3D multiplication is not supported yet
    elif isinstance(a, Point3D) and isinstance(b, Point3D):
        raise NotImplementedError("Multiplication for Point3D is not implemented yet")

    # Scalar * Matrix
    elif isinstance(a, (Scalar, int, float)) and isinstance(b, Matrix2D):
        return TMatrix2D.scalar_mul(b, a)

    # Matrix * Scalar
    elif isinstance(a, Matrix2D) and isinstance(b, (Scalar, int, float)):
        return TMatrix2D.scalar_mul(a, b)

    # Scalar * Point2D
    elif isinstance(a, (Scalar, int, float)) and isinstance(b, Point2D):
        return TPoint2D.scalar_mul(b, a)

    # Point2D * Scalar
    elif isinstance(a, Point2D) and isinstance(b, (Scalar, int, float)):
        return TPoint2D.scalar_mul(a, b)

    # Scalar * Point3D is not supported yet
    elif isinstance(a, (Scalar, int, float)) and isinstance(b, Point3D):
        raise NotImplementedError("Scalar multiplication for Point3D is not implemented yet")

    # Point3D * Scalar is not supported yet
    elif isinstance(a, Point3D) and isinstance(b, (Scalar, int, float)):
        raise NotImplementedError("Scalar multiplication for Point3D is not implemented yet")

    else:
        raise TypeError(f"Unsupported types for multiplication: {type(a)} and {type(b)}")


def div(a: Any, b: Any) -> Any:
    """
    Divide a by b.

    Args:
        a: First object
        b: Second object

    Returns:
        The result of dividing a by b, with the appropriate type
    """
    # Scalar division
    if isinstance(a, (Scalar, int, float)) and isinstance(b, (Scalar, int, float)):
        return TScalar.div(a, b)

    # Point2D division by scalar
    elif isinstance(a, Point2D) and isinstance(b, (Scalar, int, float)):
        return TPoint2D.div(a, b)

    # Point3D division by scalar is not supported yet
    elif isinstance(a, Point3D) and isinstance(b, (Scalar, int, float)):
        raise NotImplementedError("Division for Point3D is not implemented yet")

    else:
        raise TypeError(f"Unsupported types for division: {type(a)} and {type(b)}")


def floor(a: Any) -> Any:
    """
    Apply the floor function to an object.

    Args:
        a: The object to apply the floor function to

    Returns:
        The result of applying the floor function to a, with the appropriate type
    """
    # Scalar floor
    if isinstance(a, (Scalar, int, float)):
        return TScalar.floor(a if isinstance(a, Scalar) else Scalar(a))

    else:
        raise TypeError(f"Unsupported type for floor: {type(a)}")


def ceil(a: Any) -> Any:
    """
    Apply the ceiling function to an object.

    Args:
        a: The object to apply the ceiling function to

    Returns:
        The result of applying the ceiling function to a, with the appropriate type
    """
    # Scalar ceil
    if isinstance(a, (Scalar, int, float)):
        return TScalar.ceil(a if isinstance(a, Scalar) else Scalar(a))

    else:
        raise TypeError(f"Unsupported type for ceil: {type(a)}")


def abs(a: Any) -> Any:
    """
    Apply the absolute value function to an object.

    Args:
        a: The object to apply the absolute value function to

    Returns:
        The result of applying the absolute value function to a, with the appropriate type
    """
    # Scalar abs
    if isinstance(a, (Scalar, int, float)):
        # For scalar values, we can use a different approach
        # First, ensure we have a Scalar
        if not isinstance(a, Scalar):
            a = Scalar(a)

        # Create a conditional expression: if a >= 0 then a else -a
        # We can implement this using existing operations
        # Check if a is negative
        is_negative = a.value < 0
        # If a is negative, return neg(a), otherwise return a
        if is_negative:
            return neg(a)
        else:
            return a

    # Point2D abs
    elif isinstance(a, Point2D):
        return TPoint2D.abs(a)

    # Point3D abs is not supported yet
    elif isinstance(a, Point3D):
        raise NotImplementedError("Absolute value for Point3D is not implemented yet")

    else:
        raise TypeError(f"Unsupported type for abs: {type(a)}")


def neg(a: Any) -> Any:
    """
    Apply the negation function to an object.

    Args:
        a: The object to apply the negation function to

    Returns:
        The result of applying the negation function to a, with the appropriate type
    """
    # Scalar neg
    if isinstance(a, (Scalar, int, float)):
        # For scalar values, we can use the sub method: neg(x) = 0 - x
        # First, ensure we have a Scalar
        if not isinstance(a, Scalar):
            a = Scalar(a)

        # Use 0 - a to compute the negation
        zero = Scalar(0)
        return TScalar.sub(zero, a)

    # Point2D neg
    elif isinstance(a, Point2D):
        return TPoint2D.neg(a)

    # Point3D neg is not supported yet
    elif isinstance(a, Point3D):
        raise NotImplementedError("Negation for Point3D is not implemented yet")

    else:
        raise TypeError(f"Unsupported type for neg: {type(a)}")


def mm(a: Matrix2D, b: Matrix2D) -> TMatrix2D:
    """
    Perform matrix-matrix multiplication.

    Args:
        a: First matrix
        b: Second matrix

    Returns:
        The result of matrix-matrix multiplication
    """
    return TMatrix2D.mm(a, b)


def transpose(a: Matrix2D) -> TMatrix2D:
    """
    Transpose a matrix.

    Args:
        a: The matrix to transpose

    Returns:
        The transposed matrix
    """
    return TMatrix2D.transpose(a)


def inverse(a: Matrix2D) -> TMatrix2D:
    """
    Compute the inverse of a matrix.

    Args:
        a: The matrix to invert

    Returns:
        The inverted matrix
    """
    return TMatrix2D.inverse(a)
