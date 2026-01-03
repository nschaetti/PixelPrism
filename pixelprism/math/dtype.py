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
"""Data types."""

# Imports
from typing import TypeAlias, Union
from enum import Enum
import numpy as np


__all__ = [
    "DType",
    "NumericType",
    "AnyDType",
    "NestedListType",
    "DataType",
    "ScalarType"
]


# Type numeric
NumericType = float | int | bool | complex | np.number
ScalarType = int | float | np.number | bool | complex
NestedListType: TypeAlias = list[Union[ScalarType, "NestedListType"]]
DataType: TypeAlias = Union[ScalarType, NestedListType]
AnyDType = Union["DType", np.dtype, type[float], type[int], type[bool]]


class DType(Enum):
    """
    Scalar/element dtypes supported by PixelPrism's math system.

    Extend as needed; backends may map these to their native types.
    """

    FLOAT64 = "float64"
    FLOAT32 = "float32"
    INT64 = "int64"
    INT32 = "int32"
    BOOL = "bool"

    # region PROPERTIES

    @property
    def is_float(self) -> bool:
        return self in {DType.FLOAT32, DType.FLOAT64}
    # end def is_float

    @property
    def is_int(self) -> bool:
        return self in {DType.INT32, DType.INT64}
    # end def is_int

    @property
    def is_bool(self) -> bool:
        return self is DType.BOOL
    # end def is_bool

    # endregion PROPERTIES

    # region PUBLIC

    def to_numpy(self) -> np.dtype:
        """Convert to numpy dtype.

        Returns
        -------
        nd.dtype
            Numpy dtype.
        """
        return np.dtype(self.value)
    # end def to_numpy

    def convert_numpy(self, data: np.ndarray) -> np.ndarray:
        """Convert numpy array to this dtype."""
        return data.astype(self.to_numpy())
    # end def convert_numpy

    # endregion PUBLIC

    # region STATIC

    @staticmethod
    def promote(a: "DType", b: "DType") -> "DType":
        """
        Return the promoted dtype for binary ops.
        """
        order = list(DType)
        return a if order.index(a) <= order.index(b) else b
    # end def promote

    @staticmethod
    def from_numpy(dtype: np.dtype) -> "DType":
        """Convert from numpy dtype.

        Returns
        -------
        DType
            Converted dtype.
        """
        return DType(dtype.name)
    # end def from_numpy

    # endregion STATIC

# end class DType

