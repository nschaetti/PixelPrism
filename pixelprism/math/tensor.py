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
# Copyright (C) 2025 Pixel Prism
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


# Imports
from typing import List, Dict, Union, Any, Optional, Tuple, Callable, Sequence
from typing import TypeAlias
import numpy as np

from .dtype import DType, AnyDType
from .shape import Shape


__all__ = [
    "Tensor",
    "DataType",
    "concatenate",
    "hstack",
    "vstack",
    "pow",
    "square",
    "sqrt",
    "cbrt",
    "reciprocal",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "deg2rad",
    "rad2deg",
    "absolute",
    "abs",
    "sign",
    "floor",
    "ceil",
    "trunc",
    "rint",
    "round",
    "clip",
    "einsum",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_not",
    "logical_and",
    "logical_or",
    "logical_xor",
    "any",
    "all",
]


NumberType = int | float | np.number | bool | complex
NumerListType: TypeAlias = list[Union[NumberType, "NumerListType"]]
DataType: TypeAlias = Union[NumberType, NumerListType, np.ndarray]


# Tensor class
def _numpy_dtype_to_dtype(dtype: np.dtype) -> DType:
    """Convert numpy dtype to pixelprism dtype.

    Parameters
    ----------
    dtype: np.dtype
        Numpy dtype.

    Returns
    -------
    DType
        Converted dtype.
    """
    if dtype == np.dtype(np.float32):
        return DType.FLOAT32
    elif dtype == np.dtype(np.float64):
        return DType.FLOAT64
    elif dtype == np.dtype(np.int32):
        return DType.INT32
    elif dtype == np.dtype(np.int64):
        return DType.INT64
    else:
        raise ValueError(f"Unsupported numpy dtype: {dtype}")
    # end if
# end def _numpy_dtype_to_dtype


def _convert_data_to_numpy_array(
        data: Union[List[float], np.ndarray],
        dtype: Optional[AnyDType] = None
) -> np.ndarray:
    """Convert data to numpy array."""
    if dtype:
        dtype = _convert_dtype_to_numpy(dtype)
    # end if
    if isinstance(data, np.ndarray):
        if dtype is not None:
            if data.dtype != dtype:
                data = data.astype(dtype)
            # end if
        return data
    else:
        return np.array(data, dtype=dtype)
    # end if
# end def _convert_data


def _convert_dtype_to_numpy(dtype: AnyDType) -> np.dtype:
    """Convert dtype to numpy dtype."""
    if isinstance(dtype, DType):
        return dtype.to_numpy()
    elif isinstance(dtype, np.dtype):
        return dtype
    elif dtype == float:
        return np.float32
    elif dtype == int:
        return np.int32
    elif isinstance(dtype, type):
        return np.dtype(dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    # end if
# end def _convert_dtype_to_numpy


def _get_dtype(
        data: Union[bool, int, float, List[bool | int | float], np.ndarray],
        dtype: Optional[DType] = None
) -> np.dtype:
    """Get dtype from data."""
    if dtype is not None:
        dtype = dtype.to_numpy()
    elif isinstance(data, np.ndarray):
        dtype = data.dtype
    elif isinstance(data, np.generic):
        dtype = data.dtype
    elif isinstance(data, float):
        dtype = np.float32
    elif isinstance(data, int):
        dtype = np.int32
    elif isinstance(data, bool):
        dtype = np.bool_
    elif isinstance(data, list):
        raise ValueError("Cannot infer dtype from list of values, specify dtype explicitly.")
    else:
        raise TypeError(f"Unsupported or unknown data type when creating tensor: {type(data)}")
    # end if
    return dtype
# end if


class Tensor:
    """
    Declare a tensor as a class.
    """

    __array_priority__ = 1000  # ensure numpy prefers Tensor overrides

    def __init__(
            self,
            *,
            data: DataType,
            dtype: Optional[DType] = None,
            mutable: bool = True
    ):
        """
        """
        # Super
        self._dtype = DType.from_numpy(_get_dtype(data, dtype))
        self._data = _convert_data_to_numpy_array(data=data, dtype=dtype)
        self._shape = Shape(dims=self._data.shape)
        self._mutable = mutable
    # end __init__

    # region PROPERTIES

    @property
    def value(self) -> np.ndarray:
        """Get the value of the tensor."""
        return self._data
    # end def value

    @property
    def dtype(self) -> DType:
        """Get the dtype of the tensor."""
        return self._dtype
    # end def dtype

    @property
    def shape(self) -> Shape:
        """Get the shape of the tensor."""
        return self._shape
    # end def shape

    @property
    def dims(self) -> Tuple[int, ...]:
        """Get the dimensions of the tensor."""
        return self._shape.dims
    # end def dims

    @property
    def mutable(self) -> bool:
        """Get the mutable status of the tensor."""
        return self._mutable
    # end def mutable

    @property
    def is_mutable(self) -> bool:
        """Get the mutable status of the tensor."""
        return self._mutable
    # end def is_mutable

    @property
    def ndim(self) -> int:
        """Get the dimension of the tensor."""
        return self._shape.rank
    # end def dim

    @property
    def rank(self) -> int:
        """Get the dimension of the tensor."""
        return self._shape.rank
    # end def rank

    @property
    def size(self) -> int:
        """Get the size of the tensor."""
        return self._shape.size
    # end def size

    @property
    def n_elements(self) -> int:
        """Get the number of elements in the tensor."""
        return self._shape.size
    # end def n_elements

    # endregion PROPERTIES

    # region PUBLIC

    def set(self, data: Union[List[float], np.ndarray]) -> None:
        """
        """
        if isinstance(data, list):
            data = _convert_data_to_numpy_array(data, dtype=self.dtype.to_numpy())
        # end if
        data = self.dtype.convert_numpy(data)
        self._data = data
    # end set

    def copy(
            self,
            mutable: Optional[bool] = None,
            copy_data: bool = True
    ) -> 'Tensor':
        """
        """
        return Tensor(
            data=self._data.copy() if copy_data else self._data,
            mutable=mutable if mutable is not None else self._mutable
        )
    # end copy

    def item(self) -> Union[int, float]:
        """Return the first element of the tensor as a scalar."""
        if self.rank == 0:
            return self._data.item()
        else:
            raise ValueError("Cannot convert a tensor to a single integer value.")
        # end if
    # end def item

    def astype(self, dtype: DType) -> 'Tensor':
        """Convert the tensor to a different dtype."""
        return Tensor(data=self._data.astype(dtype.to_numpy()), dtype=dtype)
    # end def as_type

    def astype_(self, new_type: DType) -> 'Tensor':
        """Convert the tensor to a different dtype."""
        self._dtype = new_type
        self._data = self._dtype.convert_numpy(self._data)
        return self
    # end def as_type_

    def as_bool(self) -> 'Tensor':
        """Convert the tensor to a boolean tensor."""
        return Tensor(data=self._data.astype(np.bool_))
    # end def as_bool

    def as_float(self) -> 'Tensor':
        """Convert the tensor to a float tensor."""
        return Tensor(data=self._data.astype(np.float32))
    # end def as_float

    def as_int(self) -> 'Tensor':
        """Convert the tensor to an integer tensor."""
        return Tensor(data=self._data.astype(np.int32))
    # end def as_int

    def as_immutable(self) -> 'Tensor':
        """Convert the tensor to an immutable tensor."""
        return Tensor(data=self._data.copy(), mutable=False)
    # end def as_immutable

    def as_mutable(self) -> 'Tensor':
        """Convert the tensor to a mutable tensor."""
        return Tensor(data=self._data.copy(), mutable=True)
    # end def as_mutable

    def reshape(self, shape: Shape) -> 'Tensor':
        """Reshape the tensor."""
        return Tensor(data=self._data.reshape(shape.dims))
    # end def reshape

    def tolist(self):
        return self._data.tolist()
    # end def tolist

    # endregion PUBLIC

    # region PRIVATE

    def _coerce_operand(self, other: Union["Tensor", DataType, np.ndarray]) -> np.ndarray:
        """Convert operands to numpy arrays for arithmetic."""
        if isinstance(other, Tensor):
            return other.value
        return np.asarray(other)
    # end def _coerce_operand

    def _binary_op(self, other, op) -> 'Tensor':
        """Apply a numpy binary operator and wrap in a Tensor."""
        other_arr = self._coerce_operand(other)
        result = op(self._data, other_arr)
        return Tensor(data=np.asarray(result))
    # end def _binary_op

    def _binary_op_reverse(self, other, op) -> 'Tensor':
        """Apply a numpy binary operator with operands reversed."""
        other_arr = self._coerce_operand(other)
        result = op(other_arr, self._data)
        return Tensor(data=np.asarray(result))
    # end def _binary_op_reverse

    def _unary_op(self, op: Callable[[np.ndarray], np.ndarray]) -> 'Tensor':
        """Apply a numpy unary operator and wrap in a Tensor."""
        result = op(self._data)
        return Tensor(data=np.asarray(result))
    # end def _unary_op

    # endregion PRIVATE

    # region OVERRIDE

    def __add__(self, other) -> 'Tensor':
        """Elementwise addition."""
        return self._binary_op(other, np.add)
    # end __add__

    def __radd__(self, other) -> 'Tensor':
        """Elementwise reverse addition."""
        return self._binary_op_reverse(other, np.add)
    # end __radd__

    def __sub__(self, other) -> 'Tensor':
        """Elementwise subtraction."""
        return self._binary_op(other, np.subtract)
    # end __sub__

    def __rsub__(self, other) -> 'Tensor':
        """Elementwise reverse subtraction."""
        return self._binary_op_reverse(other, np.subtract)
    # end __rsub__

    def __mul__(self, other) -> 'Tensor':
        """Elementwise multiplication."""
        return self._binary_op(other, np.multiply)
    # end __mul__

    def __rmul__(self, other) -> 'Tensor':
        """Elementwise reverse multiplication."""
        return self._binary_op_reverse(other, np.multiply)
    # end __rmul__

    def __truediv__(self, other) -> 'Tensor':
        """Elementwise division."""
        return self._binary_op(other, np.divide)
    # end __truediv__

    def __rtruediv__(self, other) -> 'Tensor':
        """Elementwise reverse division."""
        return self._binary_op_reverse(other, np.divide)
    # end __rtruediv__

    def __pow__(self, other) -> 'Tensor':
        """Elementwise power."""
        return self._binary_op(other, np.power)
    # end __pow__

    def __rpow__(self, other) -> 'Tensor':
        """Elementwise reverse power."""
        return self._binary_op_reverse(other, np.power)
    # end __rpow__

    def __matmul__(self, other) -> 'Tensor':
        """Matrix multiplication."""
        other_arr = self._coerce_operand(other)
        result = np.matmul(self._data, other_arr)
        return Tensor(data=np.asarray(result))
    # end __matmul__

    def __rmatmul__(self, other) -> 'Tensor':
        """Reverse matrix multiplication."""
        other_arr = self._coerce_operand(other)
        result = np.matmul(other_arr, self._data)
        return Tensor(data=np.asarray(result))
    # end __rmatmul__

    def __neg__(self) -> 'Tensor':
        """Elementwise negation."""
        return Tensor(data=np.negative(self._data))
    # end __neg__

    # Override the integer conversion
    def __int__(self):
        """
        """
        if self.rank == 0:
            return int(self._data.item())
        else:
            raise TypeError("Cannot convert a tensor to a single integer value.")
        # end if
    # end __int__

    # Override the float conversion
    def __float__(self):
        """
        """
        if self.rank == 0:
            return float(self._data.item())
        else:
            raise TypeError("Cannot convert a tensor to a single integer value.")
        # end if
    # end __float__

    def __str__(self):
        """
        Convert this Scalar to a string.

        This method is called when a string representation of the Tensor is needed,
        such as when using the str() function or print().

        Returns:
            str: A string representation of the Tensor value.
        """
        return f"tensor({str(self._data.tolist())}, dtype={self._dtype}, shape={self._shape}, mutable={self._mutable})"
    # end __str__

    def __repr__(self):
        """
        Return a string representation of this Scalar for debugging.

        This method is called when a developer-friendly representation of the Tensor
        is needed, such as in the Python interactive shell or debugger.

        Returns:
            str: A string representation that includes the class name and value.
        """
        return self.__str__()
    # end __repr__

    def __getitem__(self, key):
        return Tensor(data=self._data[key])
    # end def __getitem__

    def __setitem__(self, key, value):
        self._data[key] = value
    # end def __setitem__

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return np.array_equal(self.value, other.value)
        elif isinstance(other, np.ndarray):
            return np.array_equal(self.value, other)
        elif isinstance(other, (int, float)):
            return np.array_equal(self.value, np.asarray(other))
        elif isinstance(other, list):
            return np.array_equal(self.value, np.asarray(other))
        else:
            return False
        # end if
    # end def __eq__

    # endregion OVERRIDE

    # region MATH BASE

    def pow(self, exponent: Union['Tensor', DataType, np.ndarray]) -> 'Tensor':
        """Elementwise power equivalent to numpy.power."""
        return self._binary_op(exponent, np.power)
    # end def pow

    def square(self) -> 'Tensor':
        """Elementwise square."""
        return self._unary_op(np.square)
    # end def square

    def sqrt(self) -> 'Tensor':
        """Elementwise square root."""
        return self._unary_op(np.sqrt)
    # end def sqrt

    def cbrt(self) -> 'Tensor':
        """Elementwise cubic root."""
        return self._unary_op(np.cbrt)
    # end def cbrt

    def reciprocal(self) -> 'Tensor':
        """Elementwise reciprocal."""
        return self._unary_op(np.reciprocal)
    # end def reciprocal

    def exp(self) -> 'Tensor':
        """Elementwise natural exponential."""
        return self._unary_op(np.exp)
    # end def exp

    def exp2(self) -> 'Tensor':
        """Elementwise base-2 exponential."""
        return self._unary_op(np.exp2)
    # end def exp2

    def expm1(self) -> 'Tensor':
        """Elementwise exp(x) - 1 with better precision for small x."""
        return self._unary_op(np.expm1)
    # end def expm1

    def log(self) -> 'Tensor':
        """Elementwise natural logarithm."""
        return self._unary_op(np.log)
    # end def log

    def log2(self) -> 'Tensor':
        """Elementwise base-2 logarithm."""
        return self._unary_op(np.log2)
    # end def log2

    def log10(self) -> 'Tensor':
        """Elementwise base-10 logarithm."""
        return self._unary_op(np.log10)
    # end def log10

    def log1p(self) -> 'Tensor':
        """Elementwise log(1 + x) with higher accuracy for small x."""
        return self._unary_op(np.log1p)
    # end def log1p

    def absolute(self) -> 'Tensor':
        """Elementwise absolute value."""
        return self._unary_op(np.absolute)
    # end def absolute

    def abs(self) -> 'Tensor':
        """Alias for absolute to mirror numpy."""
        return self.absolute()
    # end def abs

    def __abs__(self) -> 'Tensor':
        """Support Python's built-in abs()."""
        return self.absolute()
    # end def __abs__

    # endregion MATH BASE

    # region MATH COMPARISON

    def equal(self, other: 'Tensor') -> 'Tensor':
        """Elementwise equality."""
        return self._binary_op(other, np.equal)
    # end def equal

    def not_equal(self, other: 'Tensor') -> 'Tensor':
        """Elementwise inequality."""
        return self._binary_op(other, np.not_equal)
    # end def not_equal

    def greater_equal(self, other: 'Tensor') -> 'Tensor':
        """Elementwise greater-than-or-equal-to."""
        return self._binary_op(other, np.greater_equal)
    # end def greater_equal

    def greater(self, other: 'Tensor') -> 'Tensor':
        """Elementwise greater-than."""
        return self._binary_op(other, np.greater)
    # end def greater

    def less_equal(self, other: 'Tensor') -> 'Tensor':
        """Elementwise less-than-or-equal-to."""
        return self._binary_op(other, np.less_equal)
    # end def less_equal

    def less(self, other: 'Tensor') -> 'Tensor':
        """Elementwise less-than."""
        return self._binary_op(other, np.less)
    # end def less

    def logical_not(self) -> 'Tensor':
        """Elementwise logical inversion."""
        return self._unary_op(np.logical_not)
    # end def logical_not

    def logical_and(self, other: 'Tensor') -> 'Tensor':
        """Elementwise logical conjunction."""
        return self._binary_op(other, np.logical_and)
    # end def logical_and

    def logical_or(self, other: 'Tensor') -> 'Tensor':
        """Elementwise logical disjunction."""
        return self._binary_op(other, np.logical_or)
    # end def logical_or

    def logical_xor(self, other: 'Tensor') -> 'Tensor':
        """Elementwise logical exclusive-or."""
        return self._binary_op(other, np.logical_xor)
    # end def logical_xor

    # endregion MATH COMPARISON

    # region MATH TRIGO

    def sin(self) -> 'Tensor':
        """Elementwise sine."""
        return self._unary_op(np.sin)
    # end def sin

    def cos(self) -> 'Tensor':
        """Elementwise cosine."""
        return self._unary_op(np.cos)
    # end def cos

    def tan(self) -> 'Tensor':
        """Elementwise tangent."""
        return self._unary_op(np.tan)
    # end def tan

    def arcsin(self) -> 'Tensor':
        """Elementwise inverse sine."""
        return self._unary_op(np.arcsin)
    # end def arcsin

    def arccos(self) -> 'Tensor':
        """Elementwise inverse cosine."""
        return self._unary_op(np.arccos)
    # end def arccos

    def arctan(self) -> 'Tensor':
        """Elementwise inverse tangent."""
        return self._unary_op(np.arctan)
    # end def arctan

    def arctan2(
            self,
            other: Union["Tensor", DataType, np.ndarray]
    ) -> 'Tensor':
        """Elementwise two-argument arctangent."""
        return self._binary_op(other, np.arctan2)
    # end def arctan2

    def sinh(self) -> 'Tensor':
        """Elementwise hyperbolic sine."""
        return self._unary_op(np.sinh)
    # end def sinh

    def cosh(self) -> 'Tensor':
        """Elementwise hyperbolic cosine."""
        return self._unary_op(np.cosh)
    # end def cosh

    def tanh(self) -> 'Tensor':
        """Elementwise hyperbolic tangent."""
        return self._unary_op(np.tanh)
    # end def tanh

    def arcsinh(self) -> 'Tensor':
        """Elementwise inverse hyperbolic sine."""
        return self._unary_op(np.arcsinh)
    # end def arcsinh

    def arccosh(self) -> 'Tensor':
        """Elementwise inverse hyperbolic cosine."""
        return self._unary_op(np.arccosh)
    # end def arccosh

    def arctanh(self) -> 'Tensor':
        """Elementwise inverse hyperbolic tangent."""
        return self._unary_op(np.arctanh)
    # end def arctanh

    def deg2rad(self) -> 'Tensor':
        """Convert angles from degrees to radians."""
        return self._unary_op(np.deg2rad)
    # end def deg2rad

    def rad2deg(self) -> 'Tensor':
        """Convert angles from radians to degrees."""
        return self._unary_op(np.rad2deg)
    # end def rad2deg

    # endregion MATH TRIGO

    # region MATH DISCRETE

    def sign(self) -> 'Tensor':
        """Elementwise sign indicator."""
        return self._unary_op(np.sign)
    # end def sign

    def floor(self) -> 'Tensor':
        """Elementwise floor."""
        return self._unary_op(np.floor)
    # end def floor

    def ceil(self) -> 'Tensor':
        """Elementwise ceiling."""
        return self._unary_op(np.ceil)
    # end def ceil

    def trunc(self) -> 'Tensor':
        """Elementwise truncate towards zero."""
        return self._unary_op(np.trunc)
    # end def trunc

    def rint(self) -> 'Tensor':
        """Elementwise rounding to nearest integer."""
        return self._unary_op(np.rint)
    # end def rint

    def round(self, decimals: int = 0) -> 'Tensor':
        """Elementwise rounding with configurable precision."""
        return Tensor(data=np.array(np.round(self._data, decimals=decimals)))
    # end def round

    def clip(
            self,
            min_value: Optional[Union['Tensor', DataType, np.ndarray]] = None,
            max_value: Optional[Union['Tensor', DataType, np.ndarray]] = None
    ) -> 'Tensor':
        """Clip tensor values between min_value and max_value."""
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be provided.")
        min_arr = self._coerce_operand(min_value) if min_value is not None else None
        max_arr = self._coerce_operand(max_value) if max_value is not None else None
        result = np.clip(self._data, min_arr, max_arr)
        return Tensor(data=np.asarray(result))
    # end def clip

    # endregion MATH DISCRETE

    # region MATH LINEAR

    def matmul(self, other: Union["Tensor", DataType, np.ndarray]) -> 'Tensor':
        """Matrix multiplication."""
        return self._binary_op(other, np.matmul)
    # end def matmul

    def einsum(self, equation: str, *operands: Union['Tensor', DataType, np.ndarray]) -> 'Tensor':
        """Compute the tensor product of the given operands using the given equation."""
        operands = [_as_numpy_operand(self)] + [_as_numpy_operand(o) for o in operands]
        result = np.einsum(equation, *operands)
        return Tensor(data=np.asarray(result))
    # end def einsum

    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1) -> 'Tensor':
        """Compute the trace of a matrix."""
        result = np.trace(self._data, offset=offset, axis1=axis1, axis2=axis2)
        return Tensor(data=np.asarray(result, dtype=self._dtype.to_numpy()))
    # end def trace

    def transpose(self, axes: Optional[List[int]] = None) -> 'Tensor':
        """Transpose the tensor."""
        result = np.transpose(self._data, axes=axes)
        return Tensor(data=np.asarray(result))
    # end def transpose

    def inverse(self) -> 'Tensor':
        """Compute the inverse of a square matrix."""
        result = np.linalg.inv(self._data)
        return Tensor(data=np.asarray(result))
    # end def inverse

    def det(self) -> 'Tensor':
        """Compute the determinant of a square matrix."""
        result = np.linalg.det(self._data)
        return Tensor(data=np.asarray(result))
    # end def det

    def norm(self, ord: Union[int, float] = 2) -> 'Tensor':
        """Compute the norm of the tensor."""
        result = np.linalg.norm(self._data, ord=ord)
    # end def norm

    # endregion MATH LINEAR

    # region MATH REDUCTION

    def sum(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the sum of the tensor along the given axis."""
        result = np.sum(self._data, axis=axis)
        return Tensor(data=np.asarray(result, dtype=self._dtype.to_numpy()))
    # end def sum

    def mean(self, axis:  Optional[int] = None) -> 'Tensor':
        """Compute the mean of the tensor along the given axis."""
        result = np.mean(self._data, axis=axis)
        return Tensor(data=np.asarray(result, dtype=self._dtype.to_numpy()))
    # end def mean

    def std(self, axis: Optional[int] = None, ddof: int = 0) -> 'Tensor':
        """Compute the standard deviation of the tensor along the given axis."""
        result = np.std(self._data, axis=axis, ddof=ddof)
        return Tensor(data=np.asarray(result, dtype=self._dtype.to_numpy()))
    # end def std

    def median(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the median of the tensor along the given axis."""
        result = np.median(self._data, axis=axis)
        return Tensor(data=np.asarray(result, dtype=self._dtype.to_numpy()))
    # end def median

    def q1(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the first quartile of the tensor along the given axis."""
        result = np.percentile(self._data, 25, axis=axis)
        return Tensor(data=np.asarray(result, dtype=self._dtype.to_numpy()))
    # end def q1

    def q3(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the third quartile of the tensor along the given axis."""
        result = np.percentile(self._data, 75, axis=axis)
        return Tensor(data=np.asarray(result, dtype=self._dtype.to_numpy()))
    # end def q3

    def max(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the maximum of the tensor along the given axis."""
        result = np.max(self._data, axis=axis)
        return Tensor(data=np.asarray(result, dtype=self._dtype.to_numpy()))
    # end def max

    def min(self, axis: Optional[int] = None) -> 'Tensor':
        """Compute the minimum of the tensor along the given axis."""
        result = np.min(self._data, axis=axis)
        return Tensor(data=np.asarray(result, dtype=self._dtype.to_numpy()))
    # end def min

    def any(self) -> 'Tensor':
        """Return True if any element evaluates to True."""
        result = np.any(self._data)
        return Tensor(data=np.asarray(result, dtype=np.bool_))
    # end def any

    def all(self) -> 'Tensor':
        """Return True if all elements evaluate to True."""
        result = np.all(self._data)
        return Tensor(data=np.asarray(result, dtype=np.bool_))
    # end def all

    # endregion MATH REDUCTION

    # region RESHAPE

    def flatten(self) -> 'Tensor':
        """Flatten the tensor into a 1D array."""
        return Tensor(data=np.reshape(self._data, (-1,)))
    # end def flatten

    def concatenate(self, *others: "Tensor", axis: Optional[int] = 0) -> 'Tensor':
        """Concatenate the current tensor with additional tensors."""
        tensors: Tuple["Tensor", ...] = (self,) + tuple(others)
        return concatenate(tensors, axis=axis)
    # end def concatenate

    def hstack(self, *others: "Tensor") -> 'Tensor':
        """Concatenate tensors along axis 1."""
        tensors: Tuple["Tensor", ...] = (self,) + tuple(others)
        return hstack(tensors)
    # end def hstack

    def vstack(self, *others: "Tensor") -> 'Tensor':
        """Concatenate tensors along axis 0."""
        tensors: Tuple["Tensor", ...] = (self,) + tuple(others)
        return vstack(tensors)
    # end def vstack

    # endregion RESHAPE

    # region STATIC

    @staticmethod
    def from_numpy(data: np.ndarray, dtype: Optional[DType] = None) -> 'Tensor':
        """Create a tensor from numpy array."""
        return Tensor(data=data, dtype=dtype)
    # end def from_numpy

    @staticmethod
    def zeros(shape: Shape, dtype: Optional[DType] = None) -> 'Tensor':
        """Create a tensor of zeros."""
        np_dtype = _convert_dtype_to_numpy(dtype) if dtype else None
        data = np.zeros(shape=shape.dims, dtype=np_dtype)
        return Tensor(data=data, dtype=dtype)
    # end def zeros

    @staticmethod
    def from_list(data: List[float], dtype: DType) -> 'Tensor':
        """Create a tensor from list."""
        return Tensor(data=data, dtype=dtype)
    # end def from_list

    # endregion STATIC

# end Tensor


def _require_tensor(value: Any) -> Tensor:
    """Ensure the provided value is a Tensor instance."""
    if not isinstance(value, Tensor):
        raise TypeError("Tensor math functions expect Tensor inputs.")
    return value
# end def _require_tensor


def _call_tensor_method(name: str, tensor: Tensor, *args, **kwargs) -> Tensor:
    """Helper to invoke a Tensor math method by name."""
    method = getattr(_require_tensor(tensor), name)
    return method(*args, **kwargs)
# end def _call_tensor_method


def _concatenate_tensors(
        tensors: Sequence[Tensor],
        axis: Optional[int]
) -> Tensor:
    """Internal helper to concatenate tensors consistently."""
    tensors_tuple = tuple(_require_tensor(tensor) for tensor in tensors)
    if not tensors_tuple:
        raise ValueError("Concatenation requires at least one tensor.")
    arrays = [tensor.value for tensor in tensors_tuple]
    result = np.concatenate(arrays, axis=axis)
    return Tensor(data=np.asarray(result))
# end def _concatenate_tensors


def _as_numpy_operand(value: Union["Tensor", DataType, np.ndarray]) -> np.ndarray:
    """Convert Tensor or array-like input to a NumPy array."""
    if isinstance(value, Tensor):
        return value.value
    # end if
    return np.asarray(value)
# end def _as_numpy_operand


def pow(tensor: Tensor, exponent: Union['Tensor', DataType, np.ndarray]) -> Tensor:
    return _call_tensor_method("pow", tensor, exponent)
# end def pow


def square(tensor: Tensor) -> Tensor:
    return _call_tensor_method("square", tensor)
# end def square


def sqrt(tensor: Tensor) -> Tensor:
    return _call_tensor_method("sqrt", tensor)
# end def sqrt


def cbrt(tensor: Tensor) -> Tensor:
    return _call_tensor_method("cbrt", tensor)
# end def cbrt


def reciprocal(tensor: Tensor) -> Tensor:
    return _call_tensor_method("reciprocal", tensor)
# end def reciprocal


def exp(tensor: Tensor) -> Tensor:
    return _call_tensor_method("exp", tensor)
# end def exp


def exp2(tensor: Tensor) -> Tensor:
    return _call_tensor_method("exp2", tensor)
# end def exp2


def expm1(tensor: Tensor) -> Tensor:
    return _call_tensor_method("expm1", tensor)
# end def expm1


def log(tensor: Tensor) -> Tensor:
    return _call_tensor_method("log", tensor)
# end def log


def log2(tensor: Tensor) -> Tensor:
    return _call_tensor_method("log2", tensor)
# end def log2


def log10(tensor: Tensor) -> Tensor:
    return _call_tensor_method("log10", tensor)
# end def log10


def log1p(tensor: Tensor) -> Tensor:
    return _call_tensor_method("log1p", tensor)
# end def log1p

# region TRIGO

def sin(tensor: Tensor) -> Tensor:
    return _call_tensor_method("sin", tensor)
# end def sin


def cos(tensor: Tensor) -> Tensor:
    return _call_tensor_method("cos", tensor)
# end def cos


def tan(tensor: Tensor) -> Tensor:
    return _call_tensor_method("tan", tensor)
# end def tan


def arcsin(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arcsin", tensor)
# end def arcsin


def arccos(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arccos", tensor)
# end def arccos


def arctan(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arctan", tensor)
# end def arctan


def atan2(
        tensor_y: Tensor,
        tensor_x: Union[Tensor, DataType, np.ndarray]
) -> Tensor:
    return tensor_y.arctan2(tensor_x)
# end def atan2


def sinh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("sinh", tensor)
# end def sinh


def cosh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("cosh", tensor)
# end def cosh


def tanh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("tanh", tensor)
# end def tanh


def arcsinh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arcsinh", tensor)
# end def arcsinh


def arccosh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arccosh", tensor)
# end def arccosh


def arctanh(tensor: Tensor) -> Tensor:
    return _call_tensor_method("arctanh", tensor)
# end def arctanh

def deg2rad(tensor: Tensor) -> Tensor:
    return _call_tensor_method("deg2rad", tensor)
# end def deg2rad


def rad2deg(tensor: Tensor) -> Tensor:
    return _call_tensor_method("rad2deg", tensor)
# end def rad2deg

# endregion TRIGO

def absolute(tensor: Tensor) -> Tensor:
    return _call_tensor_method("absolute", tensor)
# end def absolute


def abs(tensor: Tensor) -> Tensor:
    return _call_tensor_method("abs", tensor)
# end def abs


def sign(tensor: Tensor) -> Tensor:
    return _call_tensor_method("sign", tensor)
# end def sign


def floor(tensor: Tensor) -> Tensor:
    return _call_tensor_method("floor", tensor)
# end def floor


def ceil(tensor: Tensor) -> Tensor:
    return _call_tensor_method("ceil", tensor)
# end def ceil


def trunc(tensor: Tensor) -> Tensor:
    return _call_tensor_method("trunc", tensor)
# end def trunc


def rint(tensor: Tensor) -> Tensor:
    return _call_tensor_method("rint", tensor)
# end def rint


def round(tensor: Tensor, decimals: int = 0) -> Tensor:
    return _call_tensor_method("round", tensor, decimals=decimals)
# end def round


def clip(
        tensor: Tensor,
        min_value: Optional[Union['Tensor', DataType, np.ndarray]] = None,
        max_value: Optional[Union['Tensor', DataType, np.ndarray]] = None
) -> Tensor:
    return _call_tensor_method("clip", tensor, min_value=min_value, max_value=max_value)
# end def clip

# region LINEAR_ALGEBRA

def einsum(
        subscripts: str,
        *operands: Union['Tensor', DataType, np.ndarray],
        out: Optional[Union['Tensor', np.ndarray]] = None,
        **kwargs
) -> Tensor:
    """Tensor-based wrapper around numpy.einsum."""
    np_operands = [_as_numpy_operand(op) for op in operands]
    out_tensor: Optional[Tensor] = None
    out_array = None
    if out is not None:
        if isinstance(out, Tensor):
            out_tensor = out
            out_array = out.value
        else:
            out_array = np.asarray(out)
        # end if
    # end if
    result = np.einsum(subscripts, *np_operands, out=out_array, **kwargs)
    if out_tensor is not None:
        return out_tensor
    # end if
    return Tensor(data=np.asarray(result))
# end def einsum


def trace(tensor: Tensor, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
    return _call_tensor_method("trace", tensor, offset=offset, axis1=axis1, axis2=axis2)
# end def trace


def transpose(tensor: Tensor, axes: Optional[List[int]] = None) -> Tensor:
    return _call_tensor_method("transpose", tensor, axes=axes)
# end def transpose


def det(tensor: Tensor) -> Tensor:
    return _call_tensor_method("det", tensor)
# end def det


def inverse(tensor: Tensor) -> Tensor:
    return _call_tensor_method("inverse", tensor)
# end def inverse


def norm(tensor: Tensor, ord: Union[int, float] = 2) -> Tensor:
    return _call_tensor_method("norm", tensor, ord=ord)
# end def norm


# endregion LINEAR_ALGEBRA

def mean(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("mean", tensor, axis=axis)
# end def mean


def std(
        tensor: Tensor,
        axis: Optional[int] = None,
        ddof: int = 0
) -> Tensor:
    return _call_tensor_method("std", tensor, axis=axis, ddof=ddof)
# end def std


def median(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("median", tensor, axis=axis)
# end def median


def q1(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("q1", tensor, axis=axis)
# end def q1


def q3(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("q3", tensor, axis=axis)
# end def q3


def max(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("max", tensor, axis=axis)
# end def max


def min(
        tensor: Tensor,
        axis: Optional[int] = None
) -> Tensor:
    return _call_tensor_method("min", tensor, axis=axis)
# end def min

#
# Reshape
#

# region SHAPE

def flatten(tensor: Tensor) -> Tensor:
    return _call_tensor_method("flatten", tensor)
# end def flatten


def concatenate(
        tensors: Sequence[Tensor],
        axis: Optional[int] = 0
) -> Tensor:
    """Concatenate multiple tensors using NumPy semantics."""
    return _concatenate_tensors(tensors, axis=axis)
# end def concatenate


def hstack(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate tensors along axis 1."""
    return _concatenate_tensors(tensors, axis=1)
# end def hstack


def vstack(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate tensors along axis 0."""
    return _concatenate_tensors(tensors, axis=0)
# end def vstack

# endregion SHAPE

#
# Boolean
#

# region BOOLEAN

def equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("equal", tensor_a, tensor_b)
# end def equal


def not_equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("not_equal", tensor_a, tensor_b)
# end def not_equal


def greater(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("greater", tensor_a, tensor_b)
# end def greater


def greater_equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("greater_equal", tensor_a, tensor_b)
# end def greater_equal


def less(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("less", tensor_a, tensor_b)
# end def less


def less_equal(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("less_equal", tensor_a, tensor_b)
# end def less_equal


def logical_not(tensor: Tensor) -> Tensor:
    return _call_tensor_method("logical_not", tensor)
# end def not


def any(tensor: Tensor) -> Tensor:
    return _call_tensor_method("any", tensor)
# end def any


def all(tensor: Tensor) -> Tensor:
    return _call_tensor_method("all", tensor)
# end def all


def logical_and(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("logical_and", tensor_a, tensor_b)
# end def logical_and


def logical_or(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("logical_or", tensor_a, tensor_b)
# end def logical_or


def logical_xor(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    return _call_tensor_method("logical_xor", tensor_a, tensor_b)
# end def logical_xor

# endregion BOOLEAN
