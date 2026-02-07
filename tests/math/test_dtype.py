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
"""Tests for pixelprism.math dtype helpers."""

# Imports
import numpy as np

from pixelprism.math.dtype import DType, to_numpy, promote, from_numpy


def test_dtype_enum():
    """Ensure the correct values are assigned."""
    assert DType.Z.value == "Z"
    assert DType.R.value == "R"
    assert DType.C.value == "C"
    assert DType.B.value == "B"
# end def test_dtype_enum


def test_dtype_order():
    """Ensure the correct order is assigned."""
    order = list(DType)
    assert order == [DType.Z, DType.R, DType.C, DType.B]
# end def test_dtype_order


def test_dtype_default_numpy_mappings():
    """Ensure default numpy mappings are assigned."""
    assert to_numpy(DType.Z) == np.dtype(np.int32)
    assert to_numpy(DType.R) == np.dtype(np.float32)
    assert to_numpy(DType.C) == np.dtype(np.complex128)
    assert to_numpy(DType.B) == np.dtype(np.bool_)
# end def test_dtype_default_numpy_mappings


def test_dtype_promotion():
    """Ensure the correct promotion rules are applied (B < Z < R < C)."""
    assert promote(DType.Z, DType.R) == DType.R
    assert promote(DType.B, DType.Z) == DType.Z
    assert promote(DType.R, DType.C) == DType.C
# end def test_dtype_promotion


def test_dtype_from_numpy():
    """Ensure numpy dtype mapping is correct."""
    assert from_numpy(np.dtype(np.float64)) == DType.R
    assert from_numpy(np.dtype(np.int64)) == DType.Z
    assert from_numpy(np.dtype(np.complex64)) == DType.C
    assert from_numpy(np.dtype(np.bool_)) == DType.B
# end def test_dtype_from_numpy
