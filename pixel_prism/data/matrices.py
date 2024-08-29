

# Imports
import numpy as np
from .data import Data
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
    A class to represent a 2D matrix that tracks transformations.
    """

    def __init__(self, matrix=None, on_change=None):
        """
        Initialize the tracking 2D matrix.

        Args:
            matrix (list or np.array): Initial matrix values, defaults to identity matrix if None
            on_change (callable): Function to call when the matrix changes
        """
        super().__init__(matrix=matrix, on_change=on_change)
    # end __init__

    def add_t(self, other):
        """
        Add two matrices and return a tracked result.

        Args:
            other (Matrix2D or np.array): Matrix to add
        """
        result = super().__add__(other)
        return TMatrix2D(result.matrix)
    # end add_t

    def sub_t(self, other):
        """
        Subtract two matrices and return a tracked result.

        Args:
            other (Matrix2D or np.array): Matrix to subtract
        """
        result = super().__sub__(other)
        return TMatrix2D(result.matrix)
    # end sub_t

    def mul_t(self, other):
        """
        Multiply the matrix by another matrix or a scalar and return a tracked result.

        Args:
            other (Matrix2D or scalar): Matrix or scalar to multiply
        """
        result = super().__mul__(other)
        return TMatrix2D(result.matrix)
    # end mul_t

    def div_t(self, other):
        """
        Divide the matrix by a scalar and return a tracked result.

        Args:
            other (scalar): Scalar to divide by
        """
        result = super().__truediv__(other)
        return TMatrix2D(result.matrix)
    # end div_t

    def transpose_t(self):
        """
        Return the transpose of the matrix as a tracked result.
        """
        result = self._matrix.T
        return TMatrix2D(result)
    # end transpose_t

    def inverse_t(self):
        """
        Return the inverse of the matrix as a tracked result.
        """
        result = np.linalg.inv(self._matrix)
        return TMatrix2D(result)
    # end inverse_t

# end TMatrix2D



