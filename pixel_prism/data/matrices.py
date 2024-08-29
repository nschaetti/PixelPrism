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


# Imports
from typing import Union
import numpy as np
from .data import Data
from .scalar import TScalar
from .points import Point2D, TPoint2D
from .eventmixin import EventMixin


class Matrix2D(Data, EventMixin):
    """
    A class to represent a 2D matrix.
    """

    def __init__(self, matrix=None, on_change=None):
        """
        Initialize the 2D matrix.

        Args:
            matrix (list or np.array): Initial matrix values, defaults to identity matrix if None
            on_change (callable): Function to call when the matrix changes
        """
        super().__init__()
        if matrix is None:
            matrix = np.identity(3)
        # end if
        self._matrix = np.array(matrix, dtype=np.float32)

        # List of event listeners (per events)
        self.event_listeners = {
            "on_change": [] if on_change is None else [on_change]
        }
    # end __init__

    @property
    def matrix(self):
        """
        Get the matrix.
        """
        return self._matrix
    # end matrix

    @matrix.setter
    def matrix(self, value):
        """
        Set the matrix and trigger event listeners.
        """
        self.set(value)
    # end matrix

    def set(self, value):
        """
        Set the matrix and trigger event listeners.
        """
        self._matrix = np.array(value, dtype=np.float32)
        self.dispatch_event("on_change", self._matrix)
    # end set

    def get(self):
        """
        Get the matrix.
        """
        return self._matrix
    # end get

    def copy(self):
        """
        Return a copy of the matrix.
        """
        return Matrix2D(self._matrix.copy())
    # end copy

    def __str__(self):
        """
        Return a string representation of the matrix.
        """
        return str(self._matrix)
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the matrix.
        """
        return f"Matrix2D(matrix={self._matrix})"
    # end __repr__

    # Operator overloads
    def __add__(self, other):
        """
        Add two matrices.
        """
        if isinstance(other, Matrix2D):
            return Matrix2D(self._matrix + other._matrix)
        return Matrix2D(self._matrix + other)
    # end __add__

    def __sub__(self, other):
        """
        Subtract two matrices.
        """
        if isinstance(other, Matrix2D):
            return Matrix2D(self._matrix - other._matrix)
        return Matrix2D(self._matrix - other)

    # end __sub__

    def __mul__(self, other):
        """
        Multiply the matrix by another matrix or a scalar.
        """
        if isinstance(other, Matrix2D):
            return Matrix2D(np.dot(self._matrix, other._matrix))
        return Matrix2D(self._matrix * other)

    # end __mul__

    def __truediv__(self, other):
        """
        Divide the matrix by a scalar.
        """
        return Matrix2D(self._matrix / other)

    # end __truediv__

    def __eq__(self, other):
        """
        Check if two matrices are equal.
        """
        if isinstance(other, Matrix2D):
            return np.array_equal(self._matrix, other._matrix)
        return False

    # end __eq__

    def __ne__(self, other):
        """
        Check if two matrices are not equal.
        """
        return not self.__eq__(other)
    # end __ne__

# end Matrix2D


class TMatrix2D(Matrix2D):
    """
    A class to represent a 2D matrix that is dynamically computed based on other Matrix2D objects.
    """

    def __init__(self, func, *sources):
        """
        Initialize the TMatrix2D.

        Args:
            func (function): A function that computes the matrix dynamically.
            sources (Matrix2D): Matrix2D objects that this TMatrix2D depends on.
        """
        self.func = func
        self.sources = sources

        # Initialize the base class with the computed matrix
        initial_value = self.func()
        super().__init__(initial_value)

        # Attach listeners to the source matrices
        for source in self.sources:
            source.add_event_listener("on_change", self._on_source_changed)
        # end for
    # end __init__

    def _on_source_changed(self, *args, **kwargs):
        """
        Update the matrix when a source Matrix2D changes.
        """
        new_value = self.func()
        self.set(new_value)
    # end _on_source_changed

    # Override set to prevent manual setting
    def set(self, value):
        """
        Prevent manual setting of the matrix. It should be computed only.
        """
        raise AttributeError("Cannot set matrix directly on TMatrix2D. It's computed based on other Matrix2D objects.")
    # end set

    def get(self):
        """
        Get the current computed matrix.
        """
        return self.func()
    # end get

# end TMatrix2D


def add_t(matrix1: Matrix2D, matrix2: Matrix2D) -> TMatrix2D:
    return TMatrix2D(lambda: matrix1.get() + matrix2.get(), matrix1, matrix2)
# end add_t

def sub_t(matrix1: Matrix2D, matrix2: Matrix2D) -> TMatrix2D:
    return TMatrix2D(lambda: matrix1.get() - matrix2.get(), matrix1, matrix2)
# end sub_t

def mul_t(matrix1: Matrix2D, matrix2: Matrix2D) -> TMatrix2D:
    return TMatrix2D(lambda: np.dot(matrix1.get(), matrix2.get()), matrix1, matrix2)
# end mul_t

def scalar_mul_t(matrix: Matrix2D, scalar: TScalar) -> TMatrix2D:
    return TMatrix2D(lambda: matrix.get() * scalar.get(), matrix, scalar)
# end scalar_mul_t

def transpose_t(matrix: Matrix2D) -> TMatrix2D:
    return TMatrix2D(lambda: np.transpose(matrix.get()), matrix)
# end transpose_t

def inverse_t(matrix: Matrix2D) -> TMatrix2D:
    return TMatrix2D(lambda: np.linalg.inv(matrix.get()), matrix)
# end inverse_t

def rotate_point_t(matrix: Matrix2D, point: Union[Point2D, TPoint2D]) -> TPoint2D:
    return TPoint2D(point, lambda p: np.dot(matrix.get(), p.pos))
# end rotate_point_t

def determinant_t(matrix: Matrix2D) -> TScalar:
    return TScalar(lambda: np.linalg.det(matrix.get()), matrix)
# end determinant_t

def trace_t(matrix: Matrix2D) -> TScalar:
    return TScalar(lambda: np.trace(matrix.get()), matrix)
# end trace_t





