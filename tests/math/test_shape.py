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
import pytest

# Imports
from pixelprism.math import Shape, Value, DType


def test_shape_basic():
    """Ensure basic shape functionality works."""
    s = Shape((2, 3))
    assert s.rank == 2
    assert s.n_dims == 2
    assert s[0] == 2
    assert s[1] == 3
# end def test_shape_basic


def test_shape_size():
    """Ensure the size of a shape is correct."""
    s = Shape((2, 3))
    assert s.size == 6
# end def test_shape_size


def test_shape_symbolic_dim():
    """Ensure symbolic dimensions work.
    When we have an unknown dimension, we don't know the number of elements.
    """
    s = Shape((None, 3))
    assert s.size is None
# end def test_shape_symbolic_dim


def test_shape_elementwise_merge():
    a = Shape((None, 3))
    b = Shape((2, 3))
    c = a.merge_elementwise(b)
    assert c == Shape((2, 3))
# end def test_shape_elementwise_merge


def test_shape_matmul():
    a = Shape((2, 3))
    b = Shape((3, 4))
    r = a.matmul_result(b)
    assert r == Shape((2, 4))
# end def test_shape_matmul


def test_shape_batched_matmul():
    a = Shape((1, 2, 2, 3))
    b = Shape((1, 2, 3, 4))
    r = a.matmul_result(b)
    assert r == Shape((1, 2, 2, 4))
# end def test_shape_batched_matmul


def test_shape_concat():
    a = Shape((2, 3))
    b = Shape((2, 4))
    r = a.concat_result(b, axis=1)
    assert r == Shape((2, 7))
# end def test_shape_concat


def test_shape_reshape():
    a = Shape((2, 3))
    b = Shape((6,))
    assert a.can_reshape(b)
# end def test_shape_reshape


def test_shape_reshape_fail():
    a = Shape((2, 3))
    b = Shape((4,))
    assert not a.can_reshape(b)
# end def test_shape_reshape_fail

