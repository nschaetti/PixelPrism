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
from typing import List, Dict, Union, Any, Optional, Tuple
import numpy as np
from .math_expr import MathExpr, MathLeaf
from .dtype import DType, AnyDType, DataType
from .shape import Shape


__all__ = [
    "Tensor"
]


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
    elif isinstance(data, float):
        dtype = np.float32
    elif isinstance(data, int):
        dtype = np.int32
    elif isinstance(data, bool):
        dtype = np.bool_
    elif isinstance(data, list):
        raise ValueError("Cannot infer dtype from list of values, specify dtype explicitly.")
    else:
        raise ValueError(f"Unsupported or unknown data type when creating tensor: {type(data)}")
    # end if
    return dtype
# end if


class Tensor(MathLeaf):
    """
    Declare a tensor as a class.
    """

    def __init__(
            self,
            *,
            name: str,
            data: Union[DataType, np.ndarray],
            dtype: Optional[DType] = None,
            mutable: bool = True
    ):
        """
        """
        # Super
        dtype = _get_dtype(data, dtype)
        data = _convert_data_to_numpy_array(data=data, dtype=dtype)
        super(Tensor, self).__init__(
            name=name,
            data=data,
            dtype=_numpy_dtype_to_dtype(data.dtype),
            shape=Shape(dims=data.shape),
            mutable=mutable
        )
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

    def _set(self, data: Union[List[float], np.ndarray]) -> None:
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
            name: str,
            mutable: Optional[bool] = None
    ) -> 'Tensor':
        """
        """
        return Tensor(
            name=name,
            data=self._data,
            mutable=mutable if mutable is not None else self._mutable
        )
    # end copy

    # endregion PUBLIC

    # region OVERRIDE

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
        if self._mutable:
            return f"tensor({self.name}, {str(self._data.tolist())}, dtype={self._dtype}, shape={self._shape})"
        else:
            return f"ctensor({self.name}, {str(self._data.tolist())}, dtype={self._dtype}, shape={self._shape})"
        # end if
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

    # endregion OVERRIDE

# end Tensor




