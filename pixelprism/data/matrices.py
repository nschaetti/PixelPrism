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
from .data import Data
from .events import Event, EventType
from .scalar import TScalar
from .points import Point2D, TPoint2D


class Matrix2D(Data):
    """
    A class to represent a 2D matrix.
    """

    def __init__(
            self,
            matrix=None,
            on_change=None,
            readonly: bool = False,
            dtype: np.dtype = np.float32,
    ):
        """
        Initialize the 2D matrix.

        Args:
            matrix (list or np.array): Initial matrix values, defaults to identity matrix if None
            on_change (callable): Function to call when the matrix changes
        """
        Data.__init__(self, readonly=readonly)

        # Identity by default
        if matrix is None:
            matrix = np.identity(3)
        # end if

        # Create data
        self._data = np.array(matrix, dtype=dtype)

        # On object changed
        self._on_change = Event()

        # Add on_change
        self._on_change += on_change
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

    # On change
    @property
    def on_change(self) -> Event:
        """
        On object changed.
        """
        return self._on_change
    # end on_change

    # endregion PROPERTIES

    # region PUBLIC

    def set(self, value):
        """
        Set the matrix and trigger event listeners.
        """
        self._data = np.array(value, dtype=np.float32)
        self._trigger_on_change()
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

    # To list
    def to_list(self):
        """
        Convert the scalar to a list.
        """
        return self._data.tolist()
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

    # endregion PUBLIC

    # region PRIVATE

    # Trigger on change
    def _trigger_on_change(self):
        """
        Trigger the on_change callback.
        """
        self._on_change.trigger(self, event_type=EventType.MATRIX_CHANGED, matrix=self._data)
    # end _trigger_on_change

    # endregion PRIVATE

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
        self._trigger_on_change()
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

    # region CLASS_METHODS

    # Identity matrix
    @classmethod
    def identity(
            cls,
            on_change=None
    ):
        """
        Return an identity matrix.
        """
        return cls(
            matrix=np.identity(2),
            on_change=on_change
        )
    # end identity

    # Scale matrix
    @classmethod
    def stretching(
            cls,
            sx: Union[float, Scalar, TScalar],
            sy: Union[float, Scalar, TScalar],
            on_change=None
    ):
        """
        Return a scale matrix.
        """
        return cls(
            matrix=np.array([
                [sx, 0],
                [0, sy],
            ]),
            on_change=on_change
        )
    # end stretching

    # endregion CLASS_METHODS

    # region OPERATORS

    # Addition
    @classmethod
    def add(
            cls,
            matrix1,
            matrix2
    ):
        """
        Add two matrices.

        Args:
            matrix1 (Matrix2D, TMatrix2D, np.ndarray): First matrix
            matrix2 (Matrix2D, TMatrix2D, np.ndarray): Second matrix
        """
        if isinstance(matrix2, np.ndarray):
            return Matrix2D(matrix1.data + matrix2)
        # end if
        return Matrix2D(matrix1.data + matrix2.data)
    # end add

    @classmethod
    def sub(
            cls,
            matrix1,
            matrix2
    ):
        """
        Subtract two matrices.

        Args:
            matrix1 (Matrix2D): First matrix
            matrix2 (Matrix2D): Second matrix
        """
        return Matrix2D(matrix1.data - matrix2.data)
    # end sub

    @classmethod
    def mul(
            cls,
            matrix1,
            matrix2
    ):
        """
        Multiply two matrices.

        Args:
            matrix1 (Matrix2D): First matrix
            matrix2 (Matrix2D): Second matrix
        """
        return TMatrix2D(lambda m1, m2: np.multiply(m1.data, m2.data), m1=matrix1, m2=matrix2)

    # end mul

    # Matrix-matrix multiplication
    @classmethod
    def mm(
            cls,
            matrix1,
            matrix2,
    ):
        """
        Multiply two matrices.

        Args:
            matrix1 (Matrix2D): First matrix
            matrix2 (Matrix2D): Second matrix
        """
        return TMatrix2D(lambda m1, m2: np.matmul(m1.data, m2.data), m1=matrix1, m2=matrix2)

    # end mm

    @classmethod
    def scalar_mul(
            cls,
            matrix,
            scalar: Union[Scalar, TScalar, float, int]
    ):
        """
        Multiply a matrix by a scalar.

        Args:
            matrix (Matrix2D): Matrix
            scalar (Scalar): Scalar
        """
        if isinstance(scalar, (int, float)):
            scalar = Scalar(scalar)
        # end if
        return TMatrix2D(lambda m, s: m.data * s.value, m=matrix, s=scalar)

    # end scalar_mul

    @classmethod
    def transpose(
            cls,
            matrix
    ):
        """
        Transpose a matrix.

        Args:
            matrix (Matrix2D): Matrix to transpose
        """
        return TMatrix2D(lambda m1: np.transpose(m1.data), m1=matrix)

    # end transpose

    @classmethod
    def inverse(
            cls,
            matrix
    ):
        """
        Inverse a matrix.

        Args:
            matrix (Matrix2D): Matrix to invert
        """
        return TMatrix2D(lambda m: np.linalg.inv(m.data), m=matrix)

    # end inverse

    # Matrix-vector multiplication
    @classmethod
    def mv(
            cls,
            matrix,
            point: Point2D,
    ) -> TPoint2D:
        """
        Multiply a matrix by a vector.

        Args:
            matrix (Matrix2D): Matrix
            point (Point2D): Point
        """
        return TPoint2D(lambda m, p: np.dot(m.data, p.pos), m=matrix, p=point)

    # end mv

    # Determinant
    @classmethod
    def determinant(
            cls,
            matrix
    ) -> TScalar:
        """
        Compute the determinant of a matrix.

        Args:
            matrix (Matrix2D): Matrix
        """
        return TScalar(lambda m: np.linalg.det(m.data), m=matrix)

    # end determinant

    # Trace of a matrix
    @classmethod
    def trace(
            cls,
            matrix
    ) -> TScalar:
        """
        Compute the trace of a matrix.

        Args:
            matrix (Matrix2D): Matrix
        """
        return TScalar(lambda m: np.trace(m.data), m=matrix)
    # end trace

    # endregion OPERATORS

# end Matrix2D


class TMatrix2D(Matrix2D):
    """
    A class to represent a 2D matrix that is dynamically computed based on other Matrix2D objects.
    """

    def __init__(
            self,
            transform_func,
            on_change=None,
            **matrices
    ):
        """
        Initialize the TMatrix2D.

        Args:
            transform_func (callable): Function to compute the matrix
            on_change (callable): Function to call when the matrix changes
            **matrices (Matrix2D/TMatrix2D/Scalar/TScalar/Point2D/TPoint2D): Source matrices
        """
        # Properties
        self._matrices = matrices
        self._transform_func = transform_func

        # Initialize the base class with the computed matrix
        data = self._transform_func(**self._matrices)
        super().__init__(data)

        # Listen to sources
        for _, s in self._matrices.items():
            if hasattr(s, "on_change"):
                s.on_change.subscribe(self._on_source_changed)
            # end if
        # end for

        # Attach listeners to the source matrices
        self._on_change += on_change
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

    # endregion PUBLIC

    # region EVENTS

    def _on_source_changed(self, sender, event_type, **kwargs):
        """
        Update the matrix when a source Matrix2D changes.
        """
        new_value = self.get()
        self._data = new_value
        self._trigger_on_change()
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
            return TMatrix2D.add(self, other)
        elif isinstance(other, np.ndarray):
            # TMatrix2D + np.ndarray = TMatrix2D
            return TMatrix2D.add(self, Matrix2D(other))
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
            return TMatrix2D.sub(self, other)
        elif isinstance(other, np.ndarray):
            # TMatrix2D - np.ndarray = TMatrix2D
            return TMatrix2D.sub(self, Matrix2D(other))
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
            return TMatrix2D.sub(other, self)
        elif isinstance(other, np.ndarray):
            # np.ndarray - TMatrix2D = TMatrix2D
            return TMatrix2D.sub(Matrix2D(other), self)
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
            return TMatrix2D.mul(self, other)
        elif isinstance(other, np.ndarray):
            # TMatrix2D * np.ndarray = TMatrix2D
            return TMatrix2D.mul(self, Matrix2D(other))
        elif isinstance(other, (int, float)):
            # TMatrix2D * scalar = TMatrix2D
            return TMatrix2D.scalar_mul(self, Scalar(other))
        elif isinstance(other, (Scalar, TScalar)):
            # TMatrix2D * TScalar = TMatrix2D
            # TMatrix2D * Scalar = TMatrix2D
            return TMatrix2D.scalar_mul(self, other)
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
            return TMatrix2D.mm(self, other)
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
            return TMatrix2D.mm(other, self)
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

    # region CLASS_METHODS

    # Identity matrix
    @classmethod
    def identity(
            cls,
            on_change=None
    ):
        """
        Return an identity matrix.
        """
        matrix = Matrix2D.identity()
        return matrix, cls(lambda m: m.data, m=matrix)
    # end identity

    # Stretching matrix
    @classmethod
    def stretching(
            cls,
            sx: Union[float, Scalar, TScalar],
            sy: Union[float, Scalar, TScalar],
            *args,
            **kwargs
    ):
        """
        Return a stretching matrix.
        """
        # Scalar, TScalar
        if isinstance(sx, (int, float)):
            sx = Scalar(sx)
        # end if

        # Scalar, TScalar
        if isinstance(sy, (int, float)):
            sy = Scalar(sy)
        # end if

        return cls(
            lambda scale_x, scale_y: np.array([[scale_x.value, 0], [0, scale_y.value]]),
            scale_x=sx,
            scale_y=sy
        )
    # end stretching

    # Shear matrix
    @classmethod
    def shearing(
            cls,
            shx: Union[float, Scalar, TScalar],
            shy: Union[float, Scalar, TScalar],
    ):
        """
        Return a shear matrix.

        Args:
            shx (float/Scalar/TScalar): Shear factor in x-direction
            shy (float/Scalar/TScalar): Shear factor in y-direction
        """
        # Scalar, TScalar
        if isinstance(shx, (int, float)):
            shx = Scalar(shx)
        # end if

        # Scalar, TScalar
        if isinstance(shy, (int, float)):
            shy = Scalar(shy)
        # end if

        return cls(
            lambda shear_x, shear_y: np.array([[1, shear_x.value], [shear_y.value, 1]]),
            shear_x=shx,
            shear_y=shy
        )
    # end shearing

    # Rotation matrix
    @classmethod
    def rotation(
            cls,
            angle: Union[float, Scalar, TScalar]
    ):
        """
        Return a rotation matrix.

        Args:
            angle (float/Scalar/TScalar): Rotation angle in radians
        """
        # Scalar, TScalar
        if isinstance(angle, (int, float)):
            angle = Scalar(angle)
        # end if

        return cls(
            lambda theta: np.array([
                [np.cos(theta.value), -np.sin(theta.value)],
                [np.sin(theta.value), np.cos(theta.value)]
            ]),
            theta=angle
        )
    # end rotation

    # Basic TMatrix2D (just return value of a matrix)
    @classmethod
    def tmatrix2d(
            cls,
            matrix: Union[Matrix2D, np.ndarray]
    ):
        """
        Return a TMatrix2D that just returns the value of the matrix.

        Args:
            matrix (Matrix2D): Matrix to be converted to TMatrix2D

        Returns:
            TMatrix2D
        """
        if isinstance(matrix, np.ndarray):
            return TMatrix2D(lambda m: m.data, m=Matrix2D(matrix))
        elif isinstance(matrix, Matrix2D):
            return TMatrix2D(lambda m: m.data, m=matrix)
        else:
            return matrix
        # end if
    # end tmatrix2d

    # endregion CLASS_METHODS

    # region OPERATORS

    # Addition
    @classmethod
    def add(
            cls,
            matrix1: Matrix2D,
            matrix2: Matrix2D
    ):
        """
        Add two matrices.

        Args:
            matrix1 (Matrix2D): First matrix
            matrix2 (Matrix2D): Second matrix
        """
        return TMatrix2D(lambda m1, m2: m1.get() + m2.get(), m1=matrix1, m2=matrix2)
    # end add

    @classmethod
    def sub(
            cls,
            matrix1: Matrix2D,
            matrix2: Matrix2D
    ):
        """
        Subtract two matrices.

        Args:
            matrix1 (Matrix2D): First matrix
            matrix2 (Matrix2D): Second matrix
        """
        return TMatrix2D(lambda m1, m2: m1.data - m2.data, m1=matrix1, m2=matrix2)
    # end sub

    @classmethod
    def mul(
            cls,
            matrix1: Matrix2D,
            matrix2: Matrix2D
    ):
        """
        Multiply two matrices.

        Args:
            matrix1 (Matrix2D): First matrix
            matrix2 (Matrix2D): Second matrix
        """
        return TMatrix2D(lambda m1, m2: np.multiply(m1.data, m2.data), m1=matrix1, m2=matrix2)
    # end mul

    # Matrix-matrix multiplication
    @classmethod
    def mm(
            cls,
            matrix1: Matrix2D,
            matrix2: Matrix2D,
    ):
        """
        Multiply two matrices.

        Args:
            matrix1 (Matrix2D): First matrix
            matrix2 (Matrix2D): Second matrix
        """
        return TMatrix2D(lambda m1, m2: np.matmul(m1.data, m2.data), m1=matrix1, m2=matrix2)
    # end mm

    @classmethod
    def scalar_mul(
            cls,
            matrix: Matrix2D,
            scalar: Union[Scalar, TScalar, float, int]
    ):
        """
        Multiply a matrix by a scalar.

        Args:
            matrix (Matrix2D): Matrix
            scalar (Scalar): Scalar
        """
        if isinstance(scalar, (int, float)):
            scalar = Scalar(scalar)
        # end if
        return TMatrix2D(lambda m, s: m.data * s.value, m=matrix, s=scalar)
    # end scalar_mul

    @classmethod
    def transpose(
            cls,
            matrix: Matrix2D
    ):
        """
        Transpose a matrix.

        Args:
            matrix (Matrix2D): Matrix to transpose
        """
        return TMatrix2D(lambda m1: np.transpose(m1.data), m1=matrix)
    # end transpose

    @classmethod
    def inverse(
            cls,
            matrix: Matrix2D
    ):
        """
        Inverse a matrix.

        Args:
            matrix (Matrix2D): Matrix to invert
        """
        return TMatrix2D(lambda m: np.linalg.inv(m.data), m=matrix)
    # end inverse

    # Matrix-vector multiplication
    @classmethod
    def mv(
            cls,
            matrix: Matrix2D,
            point: Point2D,
    ) -> TPoint2D:
        """
        Multiply a matrix by a vector.

        Args:
            matrix (Matrix2D): Matrix
            point (Point2D): Point
        """
        return TPoint2D(lambda m, p: np.dot(m.data, p.pos), m=matrix, p=point)
    # end mv

    # Determinant
    @classmethod
    def determinant(
            cls,
            matrix: Matrix2D
    ) -> TScalar:
        """
        Compute the determinant of a matrix.

        Args:
            matrix (Matrix2D): Matrix
        """
        return TScalar(lambda m: np.linalg.det(m.data), m=matrix)
    # end determinant

    # Trace of a matrix
    @classmethod
    def trace(
            cls,
            matrix: Matrix2D
    ) -> TScalar:
        """
        Compute the trace of a matrix.

        Args:
            matrix (Matrix2D): Matrix
        """
        return TScalar(lambda m: np.trace(m.data), m=matrix)
    # end trace

    # endregion OPERATORS

# end TMatrix2D

