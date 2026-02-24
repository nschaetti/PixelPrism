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
"""
NumPy-aligned dtype utilities for PixelPrism's algebra system.

This module defines a minimal symbolic dtype enum (``DType``) and helper
functions that map those symbols to concrete NumPy dtypes. Conversion helpers
follow NumPy conventions:

- ``Z`` maps to integral dtypes (default: ``np.int32``).
- ``R`` maps to real floating dtypes (default: ``np.float32``).
- ``C`` maps to complex dtypes (default: ``np.complex128``).
- ``B`` maps to boolean dtypes (``np.bool_``).

Override ``Z_DTYPE`` and ``R_DTYPE`` at the module level to adjust the default
NumPy dtype returned by ``to_numpy`` when ``DType.Z`` or ``DType.R`` is used.
"""

# Imports
from typing import TypeAlias, Union
from enum import Enum
import sys
import numpy as np


__all__ = [
    "DType",
    "TypeLike",
    "Z_DTYPE",
    "R_DTYPE",
    "to_numpy",
    "convert_numpy",
    "copy",
    "create",
    "promote",
    "from_numpy",
]


# Type numeric
TypeLike = Union["DType", np.dtype, type[float], type[int], type[bool], type[complex]]


class DType(Enum):
    """
    Scalar/element dtypes supported by PixelPrism's algebra system.

    Z: integers
    R: real floats
    C: complex
    B: boolean
    """

    Z = "Z"
    R = "R"
    C = "C"
    B = "B"

    def is_integer(self) -> bool:
        return self is DType.Z
    # end def is_integer

    def is_numeric(self) -> bool:
        return self in (DType.Z, DType.R, DType.C)
    # end def is_numeric

    def is_boolean(self) -> bool:
        return self is DType.B
    # end def is_boolean

    def is_complex(self) -> bool:
        return self is DType.C
    # end def is_complex

    def is_real(self) -> bool:
        return self is DType.R
    # end def is_real

    def to_numpy(self) -> np.dtype:
        return to_numpy(self)
    # end def to_numpy

    def copy(self) -> "DType":
        return copy(self)
    # end def copy

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DType) and self.name == other.name
    # end def __eq__

    def __neq__(self, other: object) -> bool:
        return not self.__eq__(other)
    # end def __neq_

    def __repr__(self) -> str:
        if self.name == "Z":
            return "Integers"
        elif self.name == "R":
            return "Reals"
        elif self.name == "C":
            return "Complex"
        elif self.name == "B":
            return "Boolean"
        # end if
        return f"Unknown"
    # end def __repr__

    def __str__(self) -> str:
        return self.__repr__()
    # end def __str__

# end class DType


Z_DTYPE = np.int32
R_DTYPE = np.float32


def _resolve_numpy_dtype(value) -> np.dtype:
    return value if isinstance(value, np.dtype) else np.dtype(value)
# end def _resolve_numpy_dtype


def _get_module_override(name: str, fallback):
    pm = sys.modules.get("pixelprism.math")
    if pm is not None and hasattr(pm, name):
        return getattr(pm, name)
    # end if
    return fallback
# end def _get_module_override


def to_numpy(dtype: TypeLike) -> np.dtype:
    """Convert to a NumPy dtype.

    Parameters
    ----------
    dtype : TypeLike
        Symbolic or concrete dtype to convert.

    Returns
    -------
    numpy.dtype
        Resolved NumPy dtype.

    Raises
    ------
    ValueError
        If the dtype cannot be resolved to a NumPy dtype.
    """
    if isinstance(dtype, DType):
        if dtype is DType.Z:
            return _resolve_numpy_dtype(_get_module_override("Z_DTYPE", Z_DTYPE))
        # end if
        if dtype is DType.R:
            return _resolve_numpy_dtype(_get_module_override("R_DTYPE", R_DTYPE))
        # end if
        if dtype is DType.C:
            return np.dtype(np.complex128)
        # end if
        if dtype is DType.B:
            return np.dtype(np.bool_)
        # end if
    # end if
    if isinstance(dtype, np.dtype):
        return dtype
    # end if
    if dtype is float:
        return _resolve_numpy_dtype(_get_module_override("R_DTYPE", R_DTYPE))
    # end if
    if dtype is int:
        return _resolve_numpy_dtype(_get_module_override("Z_DTYPE", Z_DTYPE))
    # end if
    if dtype is bool:
        return np.dtype(np.bool_)
    # end if
    if dtype is complex:
        return np.dtype(np.complex128)
    # end if
    if isinstance(dtype, type):
        return np.dtype(dtype)
    # end if
    raise ValueError(f"Unsupported dtype: {dtype}")
# end def to_numpy


def convert_numpy(dtype: DType, data: np.ndarray) -> np.ndarray:
    """Convert a NumPy array to the requested dtype.

    Parameters
    ----------
    dtype : DType
        Target symbolic dtype.
    data : numpy.ndarray
        Input array to convert.

    Returns
    -------
    numpy.ndarray
        Array cast to the resolved NumPy dtype.
    """
    return data.astype(to_numpy(dtype))
# end def convert_numpy


def copy(dtype: DType) -> DType:
    """Return a shallow copy of the dtype.

    Parameters
    ----------
    dtype : DType
        DType instance to copy.

    Returns
    -------
    DType
        Copied dtype (identity for enum values).
    """
    return dtype
# end def copy


def create(dtype: TypeLike) -> DType:
    """Create a symbolic dtype from a NumPy or Python type.

    Parameters
    ----------
    dtype : TypeLike
        DType enum, NumPy dtype, or Python scalar type.

    Returns
    -------
    DType
        Symbolic dtype mapping.

    Raises
    ------
    ValueError
        If the input dtype is unsupported.
    """
    if isinstance(dtype, DType):
        return copy(dtype)
    # end if
    if isinstance(dtype, np.dtype):
        return from_numpy(dtype)
    # end if
    if isinstance(dtype, type):
        if dtype is float:
            return DType.R
        # end if
        if dtype is int:
            return DType.Z
        # end if
        if dtype is bool:
            return DType.B
        # end if
        if dtype is complex:
            return DType.C
        # end if
        return from_numpy(np.dtype(dtype))
    # end if
    raise ValueError(f"Unsupported dtype: {dtype}")
# end def create


def promote(a: DType, b: DType) -> DType:
    """Return the promoted dtype for binary ops.

    Parameters
    ----------
    a : DType
        Left operand dtype.
    b : DType
        Right operand dtype.

    Returns
    -------
    DType
        Promoted dtype.
    """
    order = [DType.C, DType.R, DType.Z, DType.B]
    return a if order.index(a) <= order.index(b) else b
# end def promote


def from_numpy(dtype: np.dtype) -> DType:
    """Convert a NumPy dtype to a symbolic dtype.

    Parameters
    ----------
    dtype : numpy.dtype
        NumPy dtype to convert.

    Returns
    -------
    DType
        Symbolic dtype mapping.

    Raises
    ------
    TypeError
        If the NumPy dtype is unsupported.
    """
    dtype = np.dtype(dtype)
    if dtype.kind in {"i", "u"}:
        return DType.Z
    # end if
    if dtype.kind == "f":
        return DType.R
    # end if
    if dtype.kind == "c":
        return DType.C
    # end if
    if dtype.kind == "b":
        return DType.B
    # end if
    raise TypeError(f"Unsupported numpy dtype: {dtype}")
# end def from_numpy
