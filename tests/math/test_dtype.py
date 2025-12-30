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
"""Tests for pixelprism.math.Value."""

# Imports
import pytest
from pixelprism.math.dtype import DType


def test_dtype_enum():
    """Ensure the correct values are assigned."""
    assert DType.FLOAT64.value == "float64"
    assert DType.FLOAT32.value == "float32"
    assert DType.INT32.value == "int32"
    assert DType.INT64.value == "int64"
    assert DType.BOOL.value == "bool"
# end def test_dtype_enum


def test_dtype_order():
    """Ensure the correct order is assigned."""
    order = list(DType)
    assert order == [DType.FLOAT64, DType.FLOAT32, DType.INT64, DType.INT32, DType.BOOL]
# end def test_dtype_order


def test_dtype_categories():
    """Ensure the correct categories are assigned."""
    assert DType.FLOAT64.is_float
    assert not DType.INT32.is_float
# end def test_dtype_categories


def test_dtype_promotion():
    """Ensure the correct promotion rules are applied (bool < int32 < int64 < float32 < float64).

    Returns
    -------
    DType :
        Promoted dtype.
    """
    assert DType.promote(DType.INT32, DType.FLOAT32) == DType.FLOAT32
# end def test_dtype_promotion
