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
from .scalar import Scalar
from .events import ObjectChangedEvent
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
        self._data = np.array(matrix, dtype=np.float32)

        # List of event listeners (per events)
        self.event_listeners = {
            "on_change": [] if on_change is None else [on_change]
        }
    # end __init__

    # region PROPERTIES

    @property
    def data(self):
        """
        Get the matrix.
        """
        return self._data
    # end data

    @data.setter
    def data(self, value):
        """
        Set the matrix and trigger event listeners.
        """
        self.set(value)
    # end data

    # endregion PROPERTIES

    # region PUBLIC

    def set(self, value):
        """
        Set the matrix and trigger event listeners.
        """
        self._data = np.array(value, dtype=np.float32)
        self.dispatch_event("on_change", self._data)
    # end set

    def get(self):
        """
        Get the matrix.
        """
        return self._data
    # end get

    def copy(self):
        """
        Return a copy of the matrix.
        """
        return Matrix2D(self._data.copy())
    # end copy

    # endregion PUBLIC

    # region OVERRIDE

    def __str__(self):
        """
        Return a string representation of the matrix.
        """
        return str(self.data)
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the matrix.
        """
        return f"Matrix2D(matrix={self.data})"
    # end __repr__

    # Get element of the matrix
    def __getitem__(self, key):
        """
        Get an element of the matrix.
        """
        return self._data[key]
    # end __getitem__

    # Set element of the matrix
    def __setitem__(self, key, value):
        """
        Set an element of the matrix.
        """
        self._data[key] = value
        self.dispatch_event("on_change", self._data)
    # end __setitem__

    # Operator overloads
    def __add__(self, other):
        """
        Add two matrices.
        """
        # float, int
        if isinstance(other, float) or isinstance(other, int):
            # Matrix2D + scalar = Matrix2D
            return Matrix2D(self.data + other)
        # Scalar, TScalar
        elif isinstance(other, TScalar):
            # Matrix2D + TScalar = TMatrix2D
            return TMatrix2D(lambda s, o: s.data + o.value, s=self, o=other)
        elif isinstance(other, Scalar):
            # Matrix2D + Scalar = Matrix2D
            return Matrix2D(self.data + other.value)
        # Point2D, TPoint2D
        elif isinstance(other, Point2D):
            # Matrix2D + Point2D = not defined
            raise ValueError("Cannot add a matrix and a point.")
        elif isinstance(other, TPoint2D):
            # Matrix2D + TPoint2D = not defined
            raise ValueError("Cannot add a matrix and a point.")
        # Matrix2D
        elif isinstance(other, TMatrix2D):
            # Matrix2D + TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m, o: m.data + o.data, m=self, o=other)
        elif isinstance(other, Matrix2D):
            # Matrix2D + Matrix2D = Matrix2D
            return Matrix2D(self.data + other.data)
        elif isinstance(other, np.ndarray):
            # Matrix2D + np.ndarray = Matrix2D
            return Matrix2D(self.data + other)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Matrix2D' and '{}'".format(type(other)))
        # end if
    # end __add__

    def __radd__(self, other):
        """
        Add two matrices.
        """
        return self.__add__(other)
    # end __radd__

    def __sub__(self, other):
        """
        Subtract two matrices.
        """
        # float, int
        if isinstance(other, float) or isinstance(other, int):
            # Matrix2D - scalar = Matrix2D
            return Matrix2D(self.data - other)
        # Scalar, TScalar
        elif isinstance(other, TScalar):
            # Matrix2D - TScalar = TMatrix2D
            return TMatrix2D(lambda s, o: s.data - o.value, s=self, o=other)
        elif isinstance(other, Scalar):
            # Matrix2D - Scalar = Matrix2D
            return Matrix2D(self.data - other.value)
        # Point2D, TPoint2D
        elif isinstance(other, Point2D):
            # Matrix2D - Point2D = not defined
            raise ValueError("Cannot substract a matrix and a point.")
        elif isinstance(other, TPoint2D):
            # Matrix2D - TPoint2D = not defined
            raise ValueError("Cannot substract a matrix and a point.")
        # Matrix2D
        elif isinstance(other, TMatrix2D):
            # Matrix2D - TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m, o: m.data - o.data, m=self, o=other)
        elif isinstance(other, Matrix2D):
            # Matrix2D - Matrix2D = Matrix2D
            return Matrix2D(self.data - other.data)
        elif isinstance(other, np.ndarray):
            # Matrix2D - np.ndarray = Matrix2D
            return Matrix2D(self.data - other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Matrix2D' and '{}'".format(type(other)))
        # end if
    # end __sub__

    def __rsub__(self, other):
        """
        Subtract two matrices.
        """
        # float, int
        if isinstance(other, float) or isinstance(other, int):
            # scalar - Matrix2D = Matrix2D
            return Matrix2D(other - self.data)
        # Scalar, TScalar
        elif isinstance(other, TScalar):
            # TScalar - TScalar = TMatrix2D
            return TMatrix2D(lambda s, o: o.value - s.data, s=self, o=other)
        elif isinstance(other, Scalar):
            # Scalar - Scalar = Matrix2D
            return Matrix2D(other.value - self.data)
        # Point2D, TPoint2D
        elif isinstance(other, Point2D):
            # Point2D - Point2D = not defined
            raise ValueError("Cannot substract a matrix and a point.")
        elif isinstance(other, TPoint2D):
            # TPoint2D - TPoint2D = not defined
            raise ValueError("Cannot substract a matrix and a point.")
        # Matrix2D
        elif isinstance(other, TMatrix2D):
            # TMatrix2D - TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m, o: o.data - m.data, m=self, o=other)
        elif isinstance(other, Matrix2D):
            # Matrix2D - Matrix2D = Matrix2D
            return Matrix2D(other.data - self.data)
        elif isinstance(other, np.ndarray):
            # np.ndarray - Matrix2D = Matrix2D
            return Matrix2D(other - self.data)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Matrix2D' and '{}'".format(type(other)))
        # end if
    # end __rsub__

    def __mul__(self, other):
        """
        Multiply the matrix by another matrix or a scalar.
        """
        # float, int
        if isinstance(other, float) or isinstance(other, int):
            # Matrix2D * scalar = Matrix2D
            return Matrix2D(self.data * other)
        # Scalar, TScalar
        elif isinstance(other, TScalar):
            # Matrix2D * TScalar = TMatrix2D
            return TMatrix2D(lambda s, o: s.data * o.value, s=self, o=other)
        elif isinstance(other, Scalar):
            # Matrix2D * Scalar = Matrix2D
            return Matrix2D(self.data * other.value)
        # Point2D, TPoint2D
        elif isinstance(other, Point2D):
            # Matrix2D * Point2D = not defined
            raise ValueError("Cannot add a matrix and a point.")
        elif isinstance(other, TPoint2D):
            # Matrix2D * TPoint2D = not defined
            raise ValueError("Cannot add a matrix and a point.")
        # Matrix2D
        elif isinstance(other, TMatrix2D):
            # Matrix2D * TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m, o: m.data * o.data, m=self, o=other)
        elif isinstance(other, Matrix2D):
            # Matrix2D * Matrix2D = Matrix2D
            return Matrix2D(self.data * other.data)
        elif isinstance(other, np.ndarray):
            # Matrix2D * np.ndarray = Matrix2D
            return Matrix2D(self.data * other)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Matrix2D' and '{}'".format(type(other)))
        # end if
    # end __mul__

    def __rmul__(self, other):
        """
        Multiply the matrix by another matrix or a scalar.
        """
        return self.__mul__(other)
    # end __rmul__

    def __matmul__(self, other):
        """
        Matrix-matrix multiplication.
        """
        if isinstance(other, Matrix2D):
            # Matrix2D @ Matrix2D = Matrix2D
            return Matrix2D(np.matmul(self.data, other.data))
        else:
            raise TypeError("Unsupported operand type(s) for @: 'Matrix2D' and '{}'".format(type(other)))
        # end if
    # end __matmul__

    # Override rmatmul
    def __rmatmul__(self, other):
        """
        Matrix-matrix multiplication.
        """
        if isinstance(other, Matrix2D):
            # Matrix2D @ Matrix2D = Matrix2D
            return Matrix2D(np.matmul(other.data, self.data))
        else:
            raise TypeError("Unsupported operand type(s) for @: 'Matrix2D' and '{}'".format(type(other)))
        # end if
    # end __rmatmul__

    def __truediv__(self, other):
        """
        Divide the matrix by a scalar.
        """
        # float, int
        if isinstance(other, float) or isinstance(other, int):
            # Matrix2D / scalar = Matrix2D
            return Matrix2D(self.data / other)
        # Scalar, TScalar
        elif isinstance(other, TScalar):
            # Matrix2D / TScalar = TMatrix2D
            return TMatrix2D(lambda s, o: s.data / o.value, s=self, o=other)
        elif isinstance(other, Scalar):
            # Matrix2D / Scalar = Matrix2D
            return Matrix2D(self.data / other.value)
        # Point2D, TPoint2D
        elif isinstance(other, Point2D):
            # Matrix2D / Point2D = not defined
            raise ValueError("Cannot add a matrix and a point.")
        elif isinstance(other, TPoint2D):
            # Matrix2D / TPoint2D = not defined
            raise ValueError("Cannot add a matrix and a point.")
        # Matrix2D
        elif isinstance(other, TMatrix2D):
            # Matrix2D / TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m, o: m.data / o.data, m=self, o=other)
        elif isinstance(other, Matrix2D):
            # Matrix2D / Matrix2D = Matrix2D
            return Matrix2D(self.data / other.data)
        elif isinstance(other, np.ndarray):
            # Matrix2D / np.ndarray = Matrix2D
            return Matrix2D(self.data / other)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Matrix2D' and '{}'".format(type(other)))
        # end if
    # end __truediv__

    # Override rtruediv
    def __rtruediv__(self, other):
        """
        Divide the matrix by a scalar.
        """
        return other.__truediv__(self)
    # end __rtruediv__

    def __eq__(self, other):
        """
        Check if two matrices are equal.
        """
        if isinstance(other, Matrix2D):
            return np.array_equal(self._data, other._data)
        # end if
        return False
    # end __eq__

    def __ne__(self, other):
        """
        Check if two matrices are not equal.
        """
        return not self.__eq__(other)
    # end __ne__

    # endregion OVERRIDE

# end Matrix2D


class TMatrix2D(Matrix2D):
    """
    A class to represent a 2D matrix that is dynamically computed based on other Matrix2D objects.
    """

    def __init__(self, transform_func, on_change=None, **matrices):
        """
        Initialize the TMatrix2D.

        Args:
            transform_func (callable): Function to compute the matrix
            on_change (callable): Function to call when the matrix changes
            **matrices (Matrix2D/TMatrix2D/Scalar/TScalar/Point2D/TPoint2D): Source matrices
        """
        self._matrices = matrices
        self._transform_func = transform_func

        # Initialize the base class with the computed matrix
        data = self._transform_func(**self._matrices)
        super().__init__(data)

        # Attach listeners to the source matrices
        if on_change is not None:
            for m in self.matrices:
                m.add_event_listener("on_change", on_change)
                m.add_event_listener("on_change", self._on_source_changed)
            # end for
        # end if
    # end __init__

    # region PROPERTIES

    @property
    def matrices(self):
        """
        Get the source matrices.
        """
        return self._matrices
    # end matrices

    @property
    def transform_func(self):
        """
        Get the transformation function.
        """
        return self._transform_func
    # end transform_func

    @property
    def data(self):
        """
        Get the current computed matrix.
        """
        m_data = self.get()
        self._data = m_data
        return m_data
    # end data

    # endregion PROPERTIES

    # region PUBLIC

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
        return self._transform_func(**self._matrices)
    # end get

    def add_event_listener(self, event_name, listener):
        """
        Add an event listener to the data object.

        Args:
            event_name (str): Event to listen for
            listener (function): Listener function
        """
        for m in self.matrices.values():
            m.add_event_listener(event_name, listener)
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
        for m in self.matrices.values():
            m.remove_event_listener(event_name, listener)
        # end for
    # end remove_event_listener

    # endregion PUBLIC

    # region EVENTS

    def _on_source_changed(self, event):
        """
        Update the matrix when a source Matrix2D changes.
        """
        new_value = self.get()
        self._data = new_value
        self.dispatch_event("on_change", ObjectChangedEvent(self, data=self.data, source=event.source))
    # end _on_source_changed

    # endregion EVENTS

    # region OVERRIDE

    def __str__(self):
        """
        Return a string representation of the matrix.
        """
        return f"TMatrix2D(transform_func={self.transform_func}, matrices={self.matrices})"
    # end __str__

    def __repr__(self):
        """
        Return a string representation of the matrix.
        """
        return f"TMatrix2D(transform_func={self.transform_func}, matrices={self.matrices})"
    # end __repr__

    # Get element of the matrix
    def __getitem__(self, key):
        """
        Get an element of the matrix.
        """
        m_data = self.get()
        self._data = m_data
        return m_data[key]
    # end __getitem__

    # Set element of the matrix
    def __setitem__(self, key, value):
        """
        Set an element of the matrix.
        """
        raise AttributeError("Cannot set matrix directly on TMatrix2D. It's computed based on other Matrix2D objects.")
    # end __setitem__

    # Override add
    def __add__(self, other):
        """
        Add two matrices.
        """
        if isinstance(other, (Matrix2D, TMatrix2D)):
            # TMatrix2D + TMatrix2D = TMatrix2D
            # TMatrix2D + Matrix2D = TMatrix2D
            return add_t(self, other)
        elif isinstance(other, np.ndarray):
            # TMatrix2D + np.ndarray = TMatrix2D
            return add_t(self, Matrix2D(other))
        elif isinstance(other, (int, float)):
            # TMatrix2D + scalar = TMatrix2D
            return TMatrix2D(lambda m, a: m.data + a, m=self, a=other)
        elif isinstance(other, (Scalar, TScalar)):
            # TMatrix2D + TScalar = TMatrix2D
            # TMatrix2D + Scalar = TMatrix2D
            return TMatrix2D(lambda m, s: m.data + s.value, m=self, s=other)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'TMatrix2D' and '{}'".format(type(other)))
        # end if
    # end __add__

    # Override radd
    def __radd__(self, other):
        """
        Add two matrices.
        """
        return self.__add__(other)
    # end __radd__

    # Override sub
    def __sub__(self, other):
        """
        Subtract two matrices.
        """
        if isinstance(other, (Matrix2D, TMatrix2D)):
            # TMatrix2D - TMatrix2D = TMatrix2D
            # TMatrix2D - Matrix2D = TMatrix2D
            return sub_t(self, other)
        elif isinstance(other, np.ndarray):
            # TMatrix2D - np.ndarray = TMatrix2D
            return sub_t(self, Matrix2D(other))
        elif isinstance(other, (int, float)):
            # TMatrix2D - scalar = TMatrix2D
            return TMatrix2D(lambda m, a: m.data - a, m=self, a=other)
        elif isinstance(other, (Scalar, TScalar)):
            # TMatrix2D - TScalar = TMatrix2D
            # TMatrix2D - Scalar = TMatrix2D
            return TMatrix2D(lambda m, s: m.data - s.value, m=self, s=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'TMatrix2D' and '{}'".format(type(other)))
        # end if
    # end __sub__

    # Override rsub
    def __rsub__(self, other):
        """
        Subtract two matrices.
        """
        if isinstance(other, (Matrix2D, TMatrix2D)):
            # TMatrix2D - TMatrix2D = TMatrix2D
            # Matrix2D - TMatrix2D = TMatrix2D
            return sub_t(other, self)
        elif isinstance(other, np.ndarray):
            # np.ndarray - TMatrix2D = TMatrix2D
            return sub_t(Matrix2D(other), self)
        elif isinstance(other, (int, float)):
            # scalar - TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m, a: a - m.data, m=self, a=other)
        elif isinstance(other, (Scalar, TScalar)):
            # TScalar - TMatrix2D = TMatrix2D
            # Scalar - TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m, s: s.value - m.data, m=self, s=other)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'TMatrix2D' and '{}'".format(type(other)))
        # end if
    # end __rsub__

    # Override mul
    def __mul__(self, other):
        """
        Multiply the matrix by another matrix or a scalar.
        """
        if isinstance(other, (Matrix2D, TMatrix2D)):
            # TMatrix2D * TMatrix2D = TMatrix2D
            # TMatrix2D * Matrix2D = TMatrix2D
            return mul_t(self, other)
        elif isinstance(other, np.ndarray):
            # TMatrix2D * np.ndarray = TMatrix2D
            return mul_t(self, Matrix2D(other))
        elif isinstance(other, (int, float)):
            # TMatrix2D * scalar = TMatrix2D
            return scalar_mul_t(self, Scalar(other))
        elif isinstance(other, (Scalar, TScalar)):
            # TMatrix2D * TScalar = TMatrix2D
            # TMatrix2D * Scalar = TMatrix2D
            return scalar_mul_t(self, other)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'TMatrix2D' and '{}'".format(type(other)))
        # end if
    # end __mul__

    # Override rmul
    def __rmul__(self, other):
        """
        Multiply the matrix by another matrix or a scalar.
        """
        self.__mul__(other)
    # end __rmul__

    def __matmul__(self, other):
        """
        Matrix-matrix multiplication.
        """
        if isinstance(other, (Matrix2D, TMatrix2D)):
            # TMatrix2D @ TMatrix2D = TMatrix2D
            # TMatrix2D @ Matrix2D = TMatrix2D
            return mm_t(self, other)
        else:
            raise TypeError("Unsupported operand type(s) for @: 'Matrix2D' and '{}'".format(type(other)))
        # end if
    # end __matmul__

    # Override rmatmul
    def __rmatmul__(self, other):
        """
        Matrix-matrix multiplication.
        """
        if isinstance(other, (Matrix2D, TMatrix2D)):
            # Matrix2D @ TMatrix2D = TMatrix2D
            # TMatrix2D @ TMatrix2D = TMatrix2D
            return mm_t(other, self)
        else:
            raise ValueError("Unsupported operand type(s) for @: 'Matrix2D' and '{}'".format(type(other)))
        # end if
    # end __rmatmul__

    # Override truediv
    def __truediv__(self, other):
        """
        Divide the matrix by a scalar.
        """
        if isinstance(other, (Matrix2D, TMatrix2D)):
            # TMatrix2D / TMatrix2D = TMatrix2D
            # TMatrix2D / Matrix2D = TMatrix2D
            return TMatrix2D(lambda m1, m2: m1.data / m2.data, m1=self, m2=other)
        elif isinstance(other, np.ndarray):
            # TMatrix2D / np.ndarray = TMatrix2D
            return TMatrix2D(lambda m, a: m.data / a, m=self, a=other)
        elif isinstance(other, (int, float)):
            # TMatrix2D / scalar = TMatrix2D
            return TMatrix2D(lambda m, a: m.data / a, m=self, a=other)
        elif isinstance(other, (TScalar, Scalar)):
            # TMatrix2D / TScalar = TMatrix2D
            # TMatrix2D / Scalar = TMatrix2D
            return TMatrix2D(lambda m, s: m.data / s.value, m=self, s=other)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'TMatrix2D' and '{}'".format(type(other)))
        # end if
    # end __truediv__

    # Override rtruediv
    def __rtruediv__(self, other):
        """
        Divide the matrix by a scalar.
        """
        if isinstance(other, (Matrix2D, TMatrix2D)):
            # TMatrix2D / TMatrix2D = TMatrix2D
            # Matrix2D / TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m1, m2: m1.data / m2.data, m1=other, m2=self)
        elif isinstance(other, np.ndarray):
            # np.ndarray / TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m, a: a / m.data, m=self, a=other)
        elif isinstance(other, (int, float)):
            # scalar / TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m, a: a / m.data, m=self, a=other)
        elif isinstance(other, (TScalar, Scalar)):
            # TScalar / TMatrix2D = TMatrix2D
            # Scalar / TMatrix2D = TMatrix2D
            return TMatrix2D(lambda m, s: s.value / m.data, m=self, s=other)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'TMatrix2D' and '{}'".format(type(other)))
        # end if
    # end __rtruediv__

    # Override eq
    def __eq__(self, other):
        """
        Check if two matrices are equal.
        """
        if isinstance(other, Matrix2D):
            return np.array_equal(self.data, other.data)
        elif isinstance(other, TMatrix2D):
            return np.array_equal(self.data, other.data)
        elif isinstance(other, np.ndarray):
            return np.array_equal(self.data, other)
        else:
            raise TypeError("Unsupported operand type(s) for ==: 'TMatrix2D' and '{}'".format(type(other)))
        # end if
    # end __eq__

    # Override ne
    def __ne__(self, other):
        """
        Check if two matrices are not equal.
        """
        return not self.__eq__(other)
    # end __ne__

    # Override abs
    def __abs__(self):
        """
        Compute the absolute value of the matrix.
        """
        return TMatrix2D(lambda m: np.abs(m.data), m=self)
    # end __abs__

    # endregion OVERRIDE

# end TMatrix2D


# Basic TMatrix2D (just return value of a matrix)
def tmatrix2d(matrix: Union[Matrix2D, TMatrix2D, np.ndarray]) -> TMatrix2D:
    """
    Return a TMatrix2D that just returns the value of the matrix.
    """
    if isinstance(matrix, np.ndarray):
        return TMatrix2D(lambda m: m.data, m=Matrix2D(matrix))
    elif isinstance(matrix, Matrix2D):
        return TMatrix2D(lambda m: m.data, m=matrix)
    else:
        return matrix
    # end if
# end tmatrix2d


# region OPERATORS


# Addition
def add_t(
        matrix1: Union[Matrix2D, TMatrix2D],
        matrix2: Union[Matrix2D, TMatrix2D]
) -> TMatrix2D:
    """
    Add two matrices.

    Args:
        matrix1 (Matrix2D/TMatrix2D): First matrix
        matrix2 (Matrix2D/TMatrix2D): Second matrix
    """
    return TMatrix2D(lambda m1, m2: m1.get() + m2.get(), m1=matrix1, m2=matrix2)
# end add_t

def sub_t(
        matrix1: Union[Matrix2D, TMatrix2D],
        matrix2: Union[Matrix2D, TMatrix2D]
) -> TMatrix2D:
    """
    Subtract two matrices.

    Args:
        matrix1 (Matrix2D/TMatrix2D): First matrix
        matrix2 (Matrix2D/TMatrix2D): Second matrix
    """
    return TMatrix2D(lambda m1, m2: m1.data - m2.data, m1=matrix1, m2=matrix2)
# end sub_t

def mul_t(
        matrix1: [Matrix2D, TMatrix2D],
        matrix2: [Matrix2D, TMatrix2D]
) -> TMatrix2D:
    """
    Multiply two matrices.

    Args:
        matrix1 (Matrix2D/TMatrix2D): First matrix
        matrix2 (Matrix2D/TMatrix2D): Second matrix
    """
    return TMatrix2D(lambda m1, m2: np.multiply(m1.data, m2.data), m1=matrix1, m2=matrix2)
# end mul_t

# Matrix-matrix multiplication
def mm_t(matrix1: [Matrix2D, TMatrix2D], matrix2: [Matrix2D, TMatrix2D]) -> TMatrix2D:
    """
    Multiply two matrices.

    Args:
        matrix1 (Matrix2D/TMatrix2D): First matrix
        matrix2 (Matrix2D/TMatrix2D): Second matrix
    """
    return TMatrix2D(lambda m1, m2: np.matmul(m1.data, m2.data), m1=matrix1, m2=matrix2)
# end mm_t

def scalar_mul_t(
        matrix: [Matrix2D, TMatrix2D],
        scalar: [Scalar, TScalar, float, int]
) -> TMatrix2D:
    """
    Multiply a matrix by a scalar.

    Args:
        matrix (Matrix2D/TMatrix2D): Matrix
        scalar (Scalar/TScalar): Scalar
    """
    if isinstance(scalar, (int, float)):
        scalar = Scalar(scalar)
    # end if
    return TMatrix2D(lambda m, s: m.data * s.value, m=matrix, s=scalar)
# end scalar_mul_t

def transpose_t(
        matrix: Union[Matrix2D, TMatrix2D]
) -> TMatrix2D:
    """
    Transpose a matrix.

    Args:
        matrix (Matrix2D/TMatrix2D): Matrix to transpose
    """
    return TMatrix2D(lambda m1: np.transpose(m1.data), m1=matrix)
# end transpose_t

def inverse_t(
        matrix: Matrix2D
) -> TMatrix2D:
    """
    Inverse a matrix.

    Args:
        matrix (Matrix2D): Matrix to invert
    """
    return TMatrix2D(lambda m: np.linalg.inv(m.data), m=matrix)
# end inverse_t

# Matrix-vector multiplication
def mv_t(
        matrix: Union[Matrix2D, TMatrix2D],
        point: Union[Point2D, TPoint2D]
) -> TPoint2D:
    """
    Multiply a matrix by a vector.

    Args:
        matrix (Matrix2D): Matrix
        point (Point2D/TPoint2D): Point
    """
    return TPoint2D(lambda m, p: np.dot(m.data, p.pos), m=matrix, p=point)
# end mv_t

# Determinant
def determinant_t(
        matrix: Union[Matrix2D, TMatrix2D]
) -> TScalar:
    """
    Compute the determinant of a matrix.

    Args:
        matrix (Matrix2D): Matrix
    """
    return TScalar(lambda m: np.linalg.det(m.data), m=matrix)
# end determinant_t

# Trace of a matrix
def trace_t(
        matrix: Union[Matrix2D, TMatrix2D]
) -> TScalar:
    """
    Compute the trace of a matrix.

    Args:
        matrix (Matrix2D/TMatrix2D): Matrix
    """
    return TScalar(lambda m: np.trace(m.data), m=matrix)
# end trace_t


# endregion OPERATORS

